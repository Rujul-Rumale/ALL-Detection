"""
train.py
========
Phase 1: Unified Training Script for ALL Leukemia Edge Classifier.
Publication-grade implementation — Research Engineering Plan v1.0.

CHANGELOG FROM v1 (reasons inline throughout):
  - Rich UI removed: was --no_live flag only; plain stdout is faster to parse
    by agentic tools and adds zero overhead per epoch
  - AsymmetricFocalLoss replaced with MildFocalLoss (gamma=0.5)
  - Alpha corrected to [0.75, 0.25]: ALL errors penalised more than HEM
  - current_state nonlocal bug fixed: dashboard CURRENT panel was always blank
  - Patience active from epoch 1 with --min_epochs guard
  - backbone.children() unfreezing replaced with named-prefix table + enum
  - Cosine T_max corrected: cosine_t_max - warmup_epochs
  - Phase LR resets now update scheduler base_lrs so cosine decays correctly
  - phase2_start default moved to 31 (data showed phases 1/1.5 gave ~0.002 AUC gain)
  - CNMCDataset samples stored as numpy arrays (prevents fork COW RAM inflation)
  - torch.load uses weights_only=False on own checkpoints (PyTorch >=2.6 compat)
  - Loss-AUC decoupling detector added (detects ghost training)
  - train_acc logged with [50/50] tag (sampler makes it non-comparable to val_acc)

Usage:
  python training_scripts/train.py --model mnv3l --fold 1 --run_name mnv3l_v1
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import json
import time
import argparse
import logging
import threading
import subprocess
from datetime import datetime
from enum import Enum          # Used for UnfreezePhase — safer than float literals like 1.5

# ── OpenCV thread cap — must happen before any torch/albumentations import ────
# Reason: OpenCV spawns its own thread pool. When DataLoader workers are also
# spawned (num_workers > 0), OpenCV threads inside each worker multiply
# aggressively, causing CPU contention and silent RAM bloat. Setting to 0
# forces OpenCV to run single-threaded inside each worker; PyTorch's own
# thread pool (set via torch.set_num_threads below) handles CPU parallelism.
import cv2
cv2.setNumThreads(0)

# ── ML / data ─────────────────────────────────────────────────────────────────
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import timm
import psutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image

# ── Project metrics (project-local) ──────────────────────────────────────────
from src.utils.training_metrics import (
    compute_all_positive_metrics,
    sweep_all_positive_thresholds,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="ALL Leukemia Edge Classifier Training"
    )

    # ── Model identity ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="mnv3l",
        choices=["mnv3l", "effb0", "effb4", "rn50"],
        # Reason: restricting to known keys ensures TIMM_NAME_MAP lookup
        # never silently falls through to a wrong backbone.
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to a .pth checkpoint to resume from. "
             "Empty string = start fresh.",
    )
    parser.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3],
        # Reason: 3-fold CV defined in cv_splits_3fold.json. Requiring it
        # prevents accidentally training without a split assignment.
    )
    parser.add_argument("--run_name", type=str, required=True)

    # ── Input / batch ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--res", type=int, default=320,
        # Reason: 320 gives a ~2× area increase over the 224 ImageNet default,
        # capturing finer cytological detail without exceeding typical 8 GB
        # VRAM at batch_size=32 with AMP enabled.
    )
    parser.add_argument("--batch_size", type=int, default=32)

    # ── Stopping conditions ───────────────────────────────────────────────────
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Hard ceiling — training never exceeds this + patience epochs "
             "regardless of AUC trajectory.",
    )
    parser.add_argument(
        "--min_epochs", type=int, default=80,
        # Reason: patience cannot fire before this epoch, preventing
        # premature stopping during the productive Phase 2 ramp-up.
        # Set to ~80 because in observed runs AUC is still actively rising
        # through epoch 80 and only plateaus around epoch 94.
        help="Earliest epoch at which early stopping can trigger.",
    )
    parser.add_argument(
        "--patience", type=int, default=25,
        # Reason: 25 epochs of no val_AUC improvement. Previous logic only
        # started counting AFTER args.epochs finished — this meant the model
        # ran 75+ epochs past convergence. Now patience counts from epoch 1
        # (gated by min_epochs). See training loop for details.
    )

    # ── Learning rates ────────────────────────────────────────────────────────
    parser.add_argument(
        "--lr_backbone", type=float, default=1e-5,
        # Reason: 10–100× smaller than lr_head. Pre-trained ImageNet weights
        # are already good; large backbone LR destroys them ("catastrophic
        # forgetting"). Small LR fine-tunes without overwriting.
    )
    parser.add_argument(
        "--lr_head", type=float, default=3e-4,
        # Reason: head is randomly initialised and needs faster convergence.
        # Default 3e-4 applies to MNV3L; other models get 1.5e-4 below
        # because their larger heads are more prone to gradient explosions.
    )

    # ── Warmup and phase schedule ─────────────────────────────────────────────
    parser.add_argument(
        "--warmup_epochs", type=int, default=10,
        # Reason: linear LR warmup stabilises the randomly-initialised head
        # before backbone gradients begin to flow. Without warmup, large
        # initial head gradients can corrupt the first few backbone updates
        # once unfreezing starts.
    )
    parser.add_argument(
        "--phase1_start", type=int, default=11,
        # Reason: immediately after warmup. At this point the head has
        # stabilised enough to accept small backbone gradients safely.
    )
    parser.add_argument(
        "--phase1_5_start", type=int, default=21,
        # Reason: gives Phase 1 (last 2 blocks) 10 epochs to adapt before
        # extending to 4 blocks. Abrupt full-unfreeze from frozen tends to
        # spike val_loss; graduated unfreezing smooths it.
    )
    parser.add_argument(
        "--phase2_start", type=int, default=31,
        # Reason: changed from 41 to 31. Empirical data showed phases 1 and
        # 1.5 (epochs 11–40) produced a combined AUC gain of only 0.002.
        # The bulk of learning happens after full unfreeze. Moving this 10
        # epochs earlier recovers productive compute that was previously wasted
        # in a partial-unfreeze holding pattern.
    )
    parser.add_argument(
        "--cosine_t_max", type=int, default=150,
        # Reason: total epochs over which cosine annealing decays from
        # base LR to eta_min. The *effective* value used in the scheduler
        # is cosine_t_max - warmup_epochs (see scheduler setup below) so
        # that the cosine period begins at warmup end and reaches eta_min
        # exactly at the hard ceiling.
    )

    # ── Regularisation ────────────────────────────────────────────────────────
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        # Reason: L2 penalty on all non-bias parameters. Keeps weights small,
        # reduces overfitting on the ~10k CNMC training images.
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.05,
        # Reason: prevents the model from outputting extreme logits (e.g.
        # probability 0.9999 for ALL). Extreme logits make the focal weight
        # (1-p_t)^gamma approach zero for every sample, amplifying the
        # self-suppression problem. 0.05 is a mild floor.
    )

    # ── Infrastructure ────────────────────────────────────────────────────────
    parser.add_argument(
        "--splits_json", type=str,
        default="cv_splits/cv_splits_3fold.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument(
        "--num_workers", type=int, default=4,
        # Reason: 4 background processes overlap data loading with GPU compute.
        # Higher values yield diminishing returns and increase memory use.
        # 0 = single-process (useful for debugging or when RAM is tight).
    )

    args = parser.parse_args()

    # Adjust head LR for larger models.
    # Reason: EffB0/B4 and ResNet50 have wider heads than MNV3L; 3e-4 can
    # cause gradient spikes in those heads during early epochs.
    if args.model != "mnv3l" and args.lr_head == 3e-4:
        args.lr_head = 1.5e-4

    return args


def setup_logging(output_dir, run_id):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_id}.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    # FileHandler: permanent record for post-hoc analysis.
    logger.addHandler(logging.FileHandler(log_file))
    # StreamHandler: live stdout so agentic runners can tail the process.
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class CNMCDataset(Dataset):
    """
    Dataset for CNMC (Cancer / Normal Microscopic Cells).

    WHY NUMPY ARRAYS INSTEAD OF PYTHON LISTS:
    ------------------------------------------
    Python lists in a Dataset object are shared with persistent DataLoader
    worker processes via os.fork(). Each time a worker reads a list element
    (e.g. self.samples[idx]) it increments that element's Python reference
    count. Because reference counts live inside the object's memory page,
    this write triggers copy-on-write (COW) on that page, causing the OS to
    duplicate the page for the worker process.

    With 4 workers and a 10,000-item list this silently replicates ~4× the
    list RAM after the first epoch, growing further over 175 epochs.
    See: github.com/pytorch/pytorch/issues/13246

    SOLUTION: store paths and labels as numpy arrays. A numpy array's
    reference count lives on the array *object* (a single allocation), not
    on each element. Reading an element does NOT trigger a COW page fault,
    so memory stays flat across all workers for the entire run.
    """

    def __init__(self, image_label_pairs, transform=None):
        paths, labels = zip(*image_label_pairs)
        # dtype=object stores Python str references in a contiguous buffer —
        # safe for arbitrary path lengths, no truncation risk.
        self.paths  = np.array(paths,  dtype=object)
        # int16 is sufficient for class indices (0 or 1); saves memory
        # compared to int32/int64 at no cost.
        self.labels = np.array(labels, dtype=np.int16)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fpath = self.paths[idx]
        # Cast back to Python int: torch.utils.data.default_collate expects
        # Python int or numpy scalar, not np.int16, for label tensors.
        label = int(self.labels[idx])

        # Primary read path: OpenCV is faster than PIL for JPEG/PNG at high res.
        image = cv2.imread(fpath)
        if image is not None:
            # OpenCV loads BGR by default; models expect RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to PIL for formats OpenCV cannot decode (e.g. TIFF,
            # some PNGs with alpha). Ensures no silent None returns from the
            # loader, which would crash the collate step with an unhelpful error.
            image = np.array(Image.open(fpath).convert("RGB"))

        if self.transform:
            # Albumentations API: pass as keyword argument; returns dict.
            image = self.transform(image=image)["image"]

        return image, label


# ═══════════════════════════════════════════════════════════════════════════════
#  Augmentations
# ═══════════════════════════════════════════════════════════════════════════════
def get_transforms(res):
    """
    Returns (train_transform, val_transform) for image resolution `res`.

    AUGMENTATION DESIGN RATIONALE:
    --------------------------------
    CNMC images are bone marrow microscopy slides scanned at varying orientations
    and with staining variation across labs and devices. The augmentation
    pipeline simulates this variation so the model generalises across scanners.

    Spatial augmentations (flip, rotate, affine, elastic):
      - Microscopy images have no canonical orientation — any flip or rotation
        is physically valid. These are the highest-value augmentations.
