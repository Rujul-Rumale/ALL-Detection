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
    logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
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
    ColorJitter (hue=0.02, brightness/contrast/saturation=0.1):
      - Simulates H&E staining variation across labs. Kept mild (0.02 hue)
        to avoid shifting cell colour so far that discriminative features
        (nuclear-to-cytoplasm ratio, chromatin texture) become unreliable.

    GaussianBlur:
      - Simulates out-of-focus scanner regions. Applied at p=0.2 to keep
        the majority of training crops in-focus.

    CoarseDropout (8 holes of size res//10):
      - Forces the model to classify from partial cell views, preventing
        over-reliance on a single morphological feature (e.g. nucleus size).

    Val transform: resize only — no augmentation. Reason: val metrics must
    reflect real-world performance on unmodified images, not augmented ones.
    """
    train_transform = A.Compose([
        # Oversample then crop: gives random spatial sampling without black
        # border padding artefacts from affine transforms.
        A.Resize(res + 32, res + 32),
        A.RandomCrop(res, res),

        # All three flips are valid for microscopy — no canonical "up".
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Affine: mild shear and scale to simulate slide mounting variation.
        A.Affine(shear=(-10, 10), scale=(0.9, 1.1), p=0.5),

        # ElasticTransform: simulates cell membrane deformation from
        # slide preparation. approximate=True uses a faster gaussian kernel
        # approximation; quality difference is negligible at sigma=10.
        A.ElasticTransform(alpha=1, sigma=10, p=0.3, approximate=True),

        # Stain normalisation substitute. True Macenko/Vahadane normalisation
        # is computationally expensive per-sample. ColorJitter at these mild
        # ranges achieves ~70% of the benefit at 0.1% of the cost.
        A.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5
        ),

        # Scanner defocus simulation.
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # Partial occlusion — forces the model to not rely on any single region.
        # NOTE: requires albumentations==1.4.18. The num_holes_range /
        # hole_height_range / hole_width_range API changed in 1.5.x.
        # Pin the version in requirements.txt to avoid silent API breakage.
        A.CoarseDropout(
            num_holes_range=(8, 8),
            hole_height_range=(res // 10, res // 10),
            hole_width_range=(res // 10, res // 10),
            p=0.2,
        ),

        # ImageNet mean/std: used because we initialise from ImageNet weights.
        # Using different stats would misalign input distributions with the
        # pre-trained feature representations.
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        # No random crop: use exact target resolution.
        A.Resize(res, res),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss Function
# ═══════════════════════════════════════════════════════════════════════════════
class MildFocalLoss(nn.Module):
    """
    Focal loss with gamma=0.5, corrected alpha, and label smoothing.

    WHY NOT THE ORIGINAL AsymmetricFocalLoss (gamma=2)?
    -----------------------------------------------------
    The focal term is (1 - p_t)^gamma. Its purpose is to down-weight
    easy (confident) samples so training focuses on hard ones. However
    at gamma=2 this backfires in later training:

      - By epoch 80+, the model classifies most training images with >90%
        confidence.
      - For such a sample: focal_weight = (1 - 0.9)^2 = 0.01
      - That sample contributes only 1% of the gradient of a chance-level
        sample.
      - With ~90% of the batch suppressed this way, the effective gradient
        signal almost vanishes.
      - The reported loss keeps falling (averages over near-zero terms) but
        no meaningful weight updates occur. This is "ghost training" — the
        dashboard reports progress while generalisation is frozen.

    At gamma=0.5:
      - The same sample: (1 - 0.9)^0.5 ≈ 0.32
      - Still down-weighted (easy samples are penalised less), but not
        suppressed. The loss remains informative across all 175 epochs and
        continues to correlate with val_AUC.

    WHY alpha=[0.75, 0.25]?
    ------------------------
    alpha[c] is a per-class penalty multiplier applied on top of the focal
    weight. The original code used alpha=[0.25, 0.75], which penalises HEM
    (class 1) errors 3× more than ALL (class 0) errors.

    For leukemia screening this is clinically backwards:
      - A false negative on ALL (missing a cancer cell) can delay diagnosis
        and worsen patient outcomes.
      - A false positive on HEM (flagging a normal cell as cancer) leads to
        a follow-up test, not a missed diagnosis.

    Correct assignment: alpha[0]=0.75 (penalise ALL errors more),
    alpha[1]=0.25 (tolerate HEM errors more).

    NOTE: WeightedRandomSampler already balances the class distribution;
    alpha provides additional *asymmetric* pressure on top of that balance.
    The two are complementary, not redundant.

    Args:
        weight (Tensor, optional): per-class weight from class_counts,
            passed to CrossEntropyLoss as the base loss.
        gamma (float): focal exponent. 0.5 chosen empirically as the
            highest value that keeps loss informative past epoch 100.
        label_smoothing (float): prevents overconfident logits that would
            push focal weights toward zero even at gamma=0.5.
    """

    def __init__(self, weight=None, gamma=0.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        # reduction="none" because we apply the focal weight per-sample
        # before taking the mean ourselves. If reduction="mean" were used,
        # we would be averaging first then multiplying — wrong order.
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )

    def forward(self, logits, targets):
        # ce_loss shape: [N] — one scalar per sample.
        ce_loss = self.ce(logits, targets)

        # Convert CE loss to probability of the correct class.
        # exp(-CE) = exp(log(p_t)) = p_t.
        # This avoids a separate softmax + gather, keeping the computation
        # numerically stable (no explicit log of a probability).
        pt = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma.
        # High p_t (confident correct prediction) → low weight → sample
        # contributes less to the gradient. Low p_t (hard example) → weight
        # closer to 1 → sample contributes more.
        focal_weight = (1.0 - pt) ** self.gamma

        # Element-wise multiply then reduce.
        return (focal_weight * ce_loss).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric Monitor (running average)
# ═══════════════════════════════════════════════════════════════════════════════
class MetricMonitor:
    """
    Numerically stable running mean for loss tracking across batches.
    Uses sum/count rather than iterative average to avoid floating-point
    drift from repeated (avg + new) / 2 style updates.
    """
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n   # weight by batch size, not batch count
        self.count += n
        self.avg   = self.sum / self.count


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """
    One full pass over the training set with AMP, gradient clipping, and
    lightweight batch-level progress printing.

    Prints a progress line every 20% of batches — enough to confirm the
    process is alive without flooding the log file. An agentic runner that
    tails stdout will see at most 5 progress lines per epoch.
    """
    model.train()
    monitor = MetricMonitor()
    correct = total = 0
    n_batches = len(loader)

    for i, (images, labels) in enumerate(loader, 1):
        # non_blocking=True overlaps the host→device transfer with CPU work
        # in the previous iteration, reducing idle GPU time.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # set_to_none=True releases gradient memory immediately rather than
        # zeroing it, reducing peak memory usage by ~1 tensor per parameter.
        optimizer.zero_grad(set_to_none=True)

        # autocast: automatically casts ops to float16 where safe, keeping
        # accumulations in float32. Roughly 1.5–2× throughput on Ampere GPUs.
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if use_amp:
            # GradScaler: scales loss before backward to prevent float16
            # gradient underflow, then unscales before the optimizer step.
            scaler.scale(loss).backward()
            # unscale_ must happen before clip_grad_norm_ so the gradient
            # norms are in the original (unscaled) magnitude.
            scaler.unscale_(optimizer)
            # Gradient clipping: prevents exploding gradients during early
            # backbone fine-tuning. max_norm=1.0 is a standard safe value.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / MPS fallback path — identical logic without AMP wrappers.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        monitor.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Print every ~20% of batches. max(1, ...) guards against n_batches < 5.
        if i % max(1, n_batches // 5) == 0 or i == n_batches:
            print(
                f"  batch {i}/{n_batches}  loss={monitor.avg:.4f}",
                flush=True,
                # flush=True: important for agentic tools that read stdout line
                # by line. Without it, Python's output buffer may hold lines
                # until the process exits.
            )

    train_acc = correct / total if total > 0 else 0.0
    # NOTE: train_acc is computed on a SYNTHETIC 50/50 distribution because
    # WeightedRandomSampler resamples to class balance. It is NOT comparable
    # to val_acc, which is computed on the natural ~2:1 ALL:HEM distribution.
    # The [50/50] tag added to the log line makes this unmistakable.
    return monitor.avg, train_acc


def validate(model, loader, criterion, device, use_amp=True):
    """
    Validation with 4-way Test-Time Augmentation (TTA).

    TTA RATIONALE:
    --------------
    Microscopy images have no canonical orientation. Averaging predictions
    over the original and three flipped variants reduces variance from
    orientation-sensitive features, typically yielding +0.5–1.5 AUC points
    at zero training cost.

    The four variants are:
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. Both flips (= 180° rotation)

    Loss is computed on the original variant only (logits_orig) for
    comparability with train_loss. Using averaged probs for the loss would
    understate it and break the loss-AUC decoupling detector.

    Returns: (val_loss, acc, auc, sensitivity, specificity, f1)
    """
    model.eval()
    monitor = MetricMonitor()
    all_targets, all_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            variants = [
                images,
                torch.flip(images, [3]),      # horizontal flip (W axis)
                torch.flip(images, [2]),      # vertical flip   (H axis)
                torch.flip(images, [2, 3]),   # both            (= 180°)
            ]

            avg_probs   = torch.zeros(images.size(0), 2, device=device)
            logits_orig = None

            with torch.amp.autocast("cuda", enabled=use_amp):
                for i, v in enumerate(variants):
                    logits = model(v)
                    if i == 0:
                        # Save original logits for loss computation.
                        logits_orig = logits
                    # .float() cast: softmax in float32 prevents precision loss
                    # when averaging probabilities that may be close to 0/1.
                    avg_probs += torch.softmax(logits.float(), dim=1)

            avg_probs /= len(variants)

            # Loss on original only — see docstring above.
            loss = criterion(logits_orig.float(), labels)
            monitor.update(loss.item(), images.size(0))

            all_targets.extend(labels.cpu().numpy())
            # avg_probs[:, 0] = P(ALL) — used as the positive class score
            # for AUC, sensitivity, specificity calculation.
            all_scores.extend(avg_probs[:, 0].cpu().numpy())

    all_targets = np.array(all_targets)
    all_scores  = np.array(all_scores)

    # compute_all_positive_metrics treats class 0 (ALL) as the positive class.
    # threshold=0.5 is the default decision boundary; post-training Youden's J
    # optimisation finds the production threshold separately.
    metrics = compute_all_positive_metrics(
        all_targets, all_scores, threshold=0.5
    )

    return (
        monitor.avg,
        metrics["acc"],
        metrics["auc"],
        metrics["sens"],
        metrics["spec"],
        metrics["f1"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Definition
# ═══════════════════════════════════════════════════════════════════════════════
# Maps CLI --model keys to timm model names.
# Kept as a module-level dict so it is visible to agentic tools inspecting
# the file without running it.
TIMM_NAME_MAP = {
    "mnv3l": "mobilenetv3_large_100",
    "effb0":  "efficientnet_b0",
    "effb4":  "efficientnet_b4",
    "rn50":   "resnet50",
}


class ConstrainedHead(nn.Module):
    """
    Classification head replacing the pre-trained backbone's original head.

    ARCHITECTURE RATIONALE:
    ------------------------
    BN → Dropout(0.4) → Linear(→512) → ReLU → Dropout(0.3) → Linear(→2)

    BatchNorm1d: normalises the backbone feature distribution before the
    linear layers. Pre-trained backbones output features with varying scale
    depending on the last pooling layer. BN removes this scale dependency,
    making the head LR less sensitive to backbone choice.

    Dropout(0.4): strong regularisation before the bottleneck. CNMC has
    ~10k training images — too few for a large linear layer to generalise
    without regularisation.

    Linear(in_features → 512): reduces from the backbone's output dimension
    (e.g. 960 for MNV3L, 1280 for EffB0) to a shared bottleneck size.
    512 was chosen as a round number that fits comfortably in cache for all
    backbone sizes tested.

    ReLU: introduces non-linearity between the two linear layers. Without
    it the two Linear layers collapse to a single affine transform.

    Dropout(0.3): lighter second dropout — the bottleneck is already smaller
    and more regularised.

    Linear(512 → 2): final logit layer. Two output classes: ALL (0), HEM (1).
    """
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),   # inplace=True saves one activation tensor
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def get_model(args):
    """
    Builds backbone + head as nn.Sequential([backbone, head]).

    WHY nn.Sequential INSTEAD OF A CUSTOM MODULE:
    -----------------------------------------------
    The two-element Sequential allows clean indexing: model[0] = backbone,
    model[1] = head. This makes partial unfreezing and per-group optimizer
    params straightforward without subclassing.

    in_features is probed by running a dummy forward pass at build time so
    the head adapts automatically to any backbone without hardcoding output
    dimensions. The probe uses batch_size=2 (not 1) because BatchNorm1d
    raises an error on single-element batches during .train() mode.
    """
    timm_name = TIMM_NAME_MAP[args.model]

    # num_classes=0: removes the backbone's original classifier head,
    # returning the global-pooled feature vector directly.
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

    # Probe output dimension with a dummy pass.
    # .eval() prevents BatchNorm from updating running stats on zeros.
    backbone.eval()
    with torch.no_grad():
        in_features = backbone(torch.zeros(2, 3, 224, 224)).shape[-1]
    backbone.train()

    model = nn.Sequential(backbone, ConstrainedHead(in_features))

    # Return param groups for logging purposes (parameter counts).
    backbone_params = list(model[0].parameters())
    head_params     = list(model[1].parameters())

    return model, backbone_params, head_params, timm_name, in_features


# ═══════════════════════════════════════════════════════════════════════════════
#  Backbone Unfreezing — Enum + Named-Prefix Table
# ═══════════════════════════════════════════════════════════════════════════════
class UnfreezePhase(Enum):
    """
    Enumerates the four backbone freeze states used during training.
    WHY AN ENUM INSTEAD OF INTEGERS OR FLOATS:
    -------------------------------------------
    The original code used phase=1.5 as a float literal to represent the
    "last 4 blocks" state. Float comparisons are fragile (IEEE 754 rounding
    can make 1.5 != 1.5 in edge cases involving arithmetic), and a float
    has no intrinsic meaning — an agentic tool or reviewer cannot know what
    "1.5" means without reading the entire function.

    An Enum is:
      - Type-safe: cannot accidentally pass 1.6 or "1.5"
      - Self-documenting: UnfreezePhase.PARTIAL4 is unambiguous
      - IDE-navigable: jump-to-definition works; float literals don't
    """
    FROZEN   = 0   # all backbone parameters frozen; head only
    PARTIAL2 = 1   # last 2 semantic blocks unfrozen
    PARTIAL4 = 2   # last 4 semantic blocks unfrozen
    FULL     = 3   # entire backbone unfrozen


# Named parameter prefixes for each backbone.
#
# WHY NOT backbone.children() (ORIGINAL BUG):
# --------------------------------------------
# backbone.children() returns the immediate child modules of the timm model
# object. For EfficientNet-B0, these are:
#   conv_stem, bn1, blocks, conv_head, bn2, global_pool
#
# Note: "blocks" is a SINGLE child module containing ALL MBConv stages.
# So list(backbone.children())[-2:] gives [bn2, global_pool] — two utility
# layers — NOT the last two MBConv stages. The intended partial unfreeze was
# never executed; the model was effectively frozen except for BN + pooling.
#
# named_parameters() walks the full parameter tree with dot-separated names
# like "blocks.6.0.conv_dw.weight". Matching against prefix "blocks.6"
# correctly selects exactly MBConv stage 6 across all timm variants.
#
# HOW TO VERIFY/EXTEND THIS TABLE:
#   import timm
#   m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
#   print([n for n, _ in m.named_parameters()][:20])
# This prints the actual parameter names for any timm model. Add entries
# for new models by identifying the last N stage prefixes from this output.
UNFREEZE_PREFIXES = {
    "mobilenetv3_large_100": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "efficientnet_b0": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "efficientnet_b4": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "resnet50": {
        # ResNet uses layer1–layer4 naming instead of blocks.N.
        UnfreezePhase.PARTIAL2: ["layer4", "layer3"],
        UnfreezePhase.PARTIAL4: ["layer4", "layer3", "layer2", "layer1"],
    },
}


def set_backbone_unfreeze_phase(
    model, phase: UnfreezePhase, timm_name: str, logger
):
    """
    Sets requires_grad on backbone parameters according to `phase`.

    Always starts by freezing everything, then selectively unfreezes.
    This ensures idempotency: calling this function twice with the same
    phase produces the same result regardless of prior state.
    """
    backbone = model[0]

    # Step 1: freeze everything. Unconditional — ensures clean state.
    for param in backbone.parameters():
        param.requires_grad = False

    if phase == UnfreezePhase.FROZEN:
        logger.info("Backbone: FROZEN (head-only training)")
        return

    if phase == UnfreezePhase.FULL:
        # Unfreeze all backbone parameters at once.
        for param in backbone.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in backbone.parameters())
        logger.info(f"Backbone: FULL — {n:,} parameters now trainable")
        return

    # PARTIAL2 or PARTIAL4: look up the prefix list for this model.
    prefixes = UNFREEZE_PREFIXES.get(timm_name, {}).get(phase, [])
    if not prefixes:
        # Safety fallback: unknown model or missing entry in the table.
        # Warn loudly so the operator can add the correct prefixes.
        logger.warning(
            f"No prefix table for {timm_name}/{phase.name}. "
            f"Falling back to FULL unfreeze — ADD THIS MODEL TO "
            f"UNFREEZE_PREFIXES to enable proper partial unfreezing."
        )
        for param in backbone.parameters():
            param.requires_grad = True
        return

    # Unfreeze only parameters whose names start with any listed prefix.
    for name, param in backbone.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = True

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(
        f"Backbone: {phase.name} — {n:,} params trainable "
        f"(prefixes: {prefixes})"
    )


def apply_phase_lr(optimizer, backbone_lr, scheduler, logger):
    """
    Sets backbone group LR and resets the cosine scheduler's internal state
    so cosine decay starts from the new LR rather than the original one.

    WHY THIS IS NEEDED (ORIGINAL BUG):
    ------------------------------------
    At each phase transition the original code did:

        optimizer.param_groups[0]["lr"] = args.lr_backbone * 0.5

    But at the END of the same epoch, run_epoch() calls scheduler.step().
    SequentialLR (once past its milestone) delegates to CosineAnnealingLR,
    which recomputes the LR from its internal base_lrs and last_epoch counter
    — OVERWRITING the manual assignment. The phase LR change had zero effect.

    Fix: after setting the LR on the optimizer group, also update:
      - active_sched.base_lrs[0]: the value cosine decays FROM
      - active_sched.last_epoch = 0: restart the cosine period from this LR

    This means cosine will decay from backbone_lr to eta_min starting at the
    phase transition epoch, which is the intended behaviour.
    """
    optimizer.param_groups[0]["lr"] = backbone_lr

    # scheduler.schedulers[-1] is CosineAnnealingLR once the warmup
    # milestone is passed. Index -1 is safe because SequentialLR always
    # has at least one scheduler after the milestone.
    active_sched = scheduler._schedulers[-1]
    if hasattr(active_sched, "base_lrs"):
        active_sched.base_lrs[0] = backbone_lr
        active_sched.last_epoch  = 0

    logger.info(
        f"Backbone LR set to {backbone_lr:.2e}; "
        f"cosine base_lrs and last_epoch reset."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def get_loaders(args):
    """
    Builds train and val DataLoaders from the cross-validation split JSON.

    CLASS IMBALANCE HANDLING:
    --------------------------
    The CNMC dataset has approximately a 2:1 ALL:HEM ratio. Two mechanisms
    are applied together (they are complementary, not redundant):

    1. WeightedRandomSampler: resamples the training set to produce
       approximately 50/50 class batches. This ensures the model sees each
       class equally during training, preventing it from learning to predict
       the majority class always.

    2. MildFocalLoss(weight=...): applies per-class loss weights derived
       from class_counts. Even with a balanced sampler, within each batch
       the model may be more or less confident about each class; the loss
       weight provides additional gradient pressure toward harder classes.

    IMPORTANT: because WeightedRandomSampler produces a 50/50 distribution,
    train_acc reflects performance on an ARTIFICIAL balanced world that does
    NOT exist at inference time. The [50/50] tag on the log line marks this.
    Val_acc reflects the real ~2:1 distribution. The two numbers are NOT
    comparable and the train/val accuracy gap is NOT a reliable measure of
    overfitting — use val_AUC and val_F1 for all convergence decisions.
    """
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    fold_key = f"fold_{args.fold}"
    if fold_key not in splits["folds"]:
        raise ValueError(
            f"Fold '{fold_key}' not found in {args.splits_json}. "
            f"Available keys: {list(splits['folds'].keys())}"
        )

    fold_data   = splits["folds"][fold_key]
    train_pairs = fold_data["train_images"]   # list of [path, label]
    val_pairs   = fold_data["val_images"]

    train_transform, val_transform = get_transforms(args.res)
    train_ds = CNMCDataset(train_pairs, transform=train_transform)
    val_ds   = CNMCDataset(val_pairs,   transform=val_transform)

    # Build sample weights: each sample's weight = 1 / count_of_its_class.
    # This makes ALL samples (minority or majority) equally likely to be drawn
    # per epoch regardless of class frequency.
    labels       = [p[1] for p in train_pairs]
    class_counts = np.bincount(labels)
    sample_weights = np.array([1.0 / class_counts[l] for l in labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        # replacement=True: allows oversampling the minority class beyond
        # its natural count, which is necessary for true class balance.
    )

    # prefetch_factor=2: each worker pre-fetches 2 batches ahead.
    # Reason: GPU processing time ≈ 1–2 batches at res=320, so 2 pre-fetched
    # batches keep the GPU fed without excessive memory buffering.
    prefetch   = 2 if args.num_workers > 0 else None
    persistent = args.num_workers > 0
    # persistent_workers=True: keeps worker processes alive between epochs.
    # Reason: spawning 4 workers takes ~2–3 seconds on some systems. Over
    # 150 epochs this adds 5–7 minutes of pure overhead. Persistent workers
    # avoid this at the cost of slightly higher idle RAM.

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,          # sampler is mutually exclusive with shuffle
        num_workers=args.num_workers,
        pin_memory=True,          # pin_memory: enables faster host→GPU copies
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
        # drop_last=True: discards the last incomplete batch each epoch.
        # Reason: BatchNorm1d in the head raises an error on batch_size=1.
        # Even if the last batch is size > 1, its gradient contribution is
        # disproportionate (smaller batch → noisier gradient estimate).
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,            # no shuffle: val order doesn't matter for metrics
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        # drop_last=False for val: we want metrics over ALL val samples.
    )

    # normed_weights used by MildFocalLoss as the `weight` argument to CE.
    # Formula: total / (n_classes * class_count)
    total_samples = class_counts.sum()
    normed_weights = total_samples / (len(class_counts) * class_counts.astype(float))

    # Class breakdown for the startup log — confirms the split loaded correctly.
    train_all = sum(1 for _, l in train_pairs if l == 0)
    train_hem = sum(1 for _, l in train_pairs if l == 1)
    val_all   = sum(1 for _, l in val_pairs   if l == 0)
    val_hem   = sum(1 for _, l in val_pairs   if l == 1)

    info = {
        "train_total": len(train_pairs), "val_total": len(val_pairs),
        "train_all":   train_all,        "train_hem": train_hem,
        "val_all":     val_all,          "val_hem":   val_hem,
    }

    return (
        train_loader, val_loader,
        torch.tensor(normed_weights, dtype=torch.float32),
        class_counts, info,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  System Monitor Thread
# ═══════════════════════════════════════════════════════════════════════════════
def system_monitor(log_path, stop_event, monitor_state, interval=5):
    """
    Background thread that samples CPU/RAM/GPU/VRAM/temp every `interval`
    seconds and writes to a CSV for post-hoc hardware analysis.

    Runs as a daemon thread so it dies automatically if the main process
    crashes without calling stop_event.set().

    monitor_state is a plain dict updated in-place and read by run_epoch()
    to append hardware info to each epoch log line. No locking is used:
    dict updates in CPython are atomic for simple key/value pairs (GIL
    prevents torn reads/writes on dict.__setitem__).
    """
    with open(log_path, "w") as f:
        f.write(
            "timestamp,cpu_pct,ram_used_gb,ram_total_gb,"
            "gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c\n"
        )

    while not stop_event.is_set():
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram     = psutil.virtual_memory()
            ram_used  = ram.used  / 1024 ** 3
            ram_total = ram.total / 1024 ** 3

            # nvidia-smi subprocess: more reliable than pynvml for reading
            # VRAM and temperature across driver versions.
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,"
                    "memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True,
            )
            parts      = [p.strip() for p in result.stdout.strip().split(",")]
            timestamp  = parts[0]
            gpu_util   = parts[1]
            vram_used  = parts[2]
            vram_total = parts[3]
            gpu_temp   = parts[4]

            with open(log_path, "a") as f:
                f.write(
                    f"{timestamp},{cpu_pct},{ram_used:.2f},{ram_total:.2f},"
                    f"{gpu_util},{vram_used},{vram_total},{gpu_temp}\n"
                )

            # update() replaces all keys atomically under CPython GIL.
            monitor_state.update({
                "cpu_pct":    float(cpu_pct),
                "ram_used":   ram_used,
                "ram_total":  ram_total,
                "gpu_util":   float(gpu_util),
                "vram_used":  float(vram_used),
                "vram_total": float(vram_total),
                "gpu_temp":   float(gpu_temp),
            })
        except Exception:
            # Silently skip failed samples (e.g. nvidia-smi not found on CPU
            # machines). The CSV will have gaps but training continues.
            pass

        stop_event.wait(interval)


def format_time(seconds):
    """Human-readable duration string for ETA and elapsed time logging."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Run identity ──────────────────────────────────────────────────────────
    # Timestamp in run_id prevents collisions when the same run_name is used
    # for multiple experiments on the same day.
    run_id = (
        f"{args.run_name}_fold{args.fold}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = os.path.join(args.output_root, args.run_name)
    logger = setup_logging(output_dir, run_id)

    # ── Device & performance flags ────────────────────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    )

    # CPU thread caps: 8 intra-op, 4 inter-op.
    # Reason: DataLoader workers + PyTorch's own thread pools can over-subscribe
    # CPU cores, causing context-switch overhead that slows both CPU and GPU
    # pipelines. Explicit caps keep total threads predictable.
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)

    # TF32 on Ampere+ GPUs: uses tensor cores for matmul/conv at ~3× the
    # throughput of FP32, with negligible accuracy difference for training.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # cudnn.benchmark: profiles convolution algorithms on the first few batches
    # and selects the fastest for the fixed input shape (res×res).
    # Only beneficial when input shapes are constant — which they are here.
    torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    # AMP only on CUDA: MPS (Apple Silicon) and CPU do not support
    # torch.amp.GradScaler and will error if use_amp=True.

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader,
     loss_weights, class_counts, data_info) = get_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, backbone_params, head_params, timm_name, in_features = get_model(args)
    model = model.to(device)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head     = sum(p.numel() for p in head_params)

    # ── Startup log ───────────────────────────────────────────────────────────
    # Written once at the top of every run. Agentic tools that parse log files
    # can use this block to reconstruct the exact hyperparameter configuration
    # without needing to re-parse CLI args.
    other_folds = [f for f in [1, 2, 3] if f != args.fold]
    folds_train = "+".join(str(f) for f in other_folds)
    # Effective cosine T_max: the warmup period is handled by LinearLR, so
    # CosineAnnealingLR should span only the remaining (non-warmup) epochs.
    cosine_t_max_effective = max(1, args.cosine_t_max - args.warmup_epochs)

    for line in [
        "=" * 70,
        "  ALL Leukemia Edge Classifier — train.py",
        f"  Run ID:            {run_id}",
        f"  Model:             {args.model}  ({timm_name})",
        f"  Fold:              {args.fold} (val) | Folds {folds_train} (train)",
        f"  Resolution:        {args.res}x{args.res}",
        f"  Batch size:        {args.batch_size}",
        f"  Min epochs:        {args.min_epochs}  (patience cannot fire before this)",
        f"  Hard ceiling:      {args.epochs}  (never exceeded regardless of patience)",
        f"  Patience:          {args.patience}",
        f"  LR backbone:       {args.lr_backbone}",
        f"  LR head:           {args.lr_head:.0e}",
        f"  Weight decay:      {args.weight_decay:.0e}",
        f"  Label smoothing:   {args.label_smoothing}",
        f"  Warmup epochs:     {args.warmup_epochs}",
        f"  Phase 1 start:     {args.phase1_start}  (last 2 blocks)",
        f"  Phase 1.5 start:   {args.phase1_5_start}  (last 4 blocks)",
        f"  Phase 2 start:     {args.phase2_start}  (full backbone)",
        f"  Cosine T_max raw:  {args.cosine_t_max}",
        f"  Cosine T_max eff:  {cosine_t_max_effective}  (= raw - warmup)",
        f"  Cosine eta_min:    1e-7",
        f"  AMP:               {use_amp}",
        f"  Device:            {device} ({gpu_name})",
        f"  Workers:           {args.num_workers}",
        f"  Splits JSON:       {args.splits_json}",
        f"  Decouple detector: fires WARNING when val_loss Δ>5e-4 and",
        f"                     val_AUC Δ<0.001 over the last 10 epochs.",
        f"                     This indicates ghost training (loss ≠ AUC).",
        "=" * 70,
        "  Dataset",
        (f"  Train: {data_info['train_total']}  "
         f"(ALL={data_info['train_all']}, HEM={data_info['train_hem']})"),
        (f"  Val:   {data_info['val_total']}  "
         f"(ALL={data_info['val_all']},   HEM={data_info['val_hem']})"),
        (f"  Loss weights:  ALL={loss_weights[0]:.6f}, "
         f"HEM={loss_weights[1]:.6f}  (from class_counts)"),
        "=" * 70,
        "  Model",
        f"  Backbone params: {n_backbone:>10,}",
        f"  Head params:     {n_head:>10,}",
        f"  Total params:    {n_backbone + n_head:>10,}",
        f"  Head:  BN1d -> Drop(0.4) -> Linear(512) -> ReLU -> Drop(0.3) -> Linear(2)",
        "=" * 70,
        "  Loss: MildFocalLoss(gamma=0.5, label_smoothing=0.05)",
        "  Alpha: [0.75, 0.25]  (ALL errors penalised 3x more than HEM)",
        "  NOTE: train_acc is measured on a SYNTHETIC 50/50 distribution.",
        "        WeightedRandomSampler resamples to class balance for training.",
        "        DO NOT compare train_acc to val_acc — different distributions.",
        "        Use val_AUC and val_F1 for all convergence and stopping decisions.",
        "=" * 70,
    ]:
        logger.info(line)
    # Save all args to JSON so any downstream tool can reconstruct the run.
    params_path = os.path.join(output_dir, f"{run_id}_params.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # ── Criterion ─────────────────────────────────────────────────────────────
    # Combine alpha and normed class weights multiplicatively.
    # alpha=[0.75, 0.25] reflects clinical priority (ALL errors are costlier).
    # loss_weights reflects class frequency (normalised 1/class_count).
    # Together they apply both demographic balance and clinical priority.
    alpha = torch.tensor([0.75, 0.25], dtype=torch.float32, device=device)
    combined_weight = alpha * loss_weights.to(device)

    criterion = MildFocalLoss(
        weight=combined_weight,
        gamma=0.5,
        label_smoothing=args.label_smoothing,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Phase 0: backbone fully frozen; only head trains.
    set_backbone_unfreeze_phase(model, UnfreezePhase.FROZEN, timm_name, logger)

    # Two parameter groups with different LRs:
    #   group 0 = backbone (frozen initially; LR will be very small when unfrozen)
    #   group 1 = head (always active; higher LR for faster convergence)
    # filter(requires_grad) on group 0 selects zero params initially (frozen),
    # but the group exists so we can update its params list at phase transitions
    # without rebuilding the optimizer (which would reset momentum buffers).
    optimizer = optim.AdamW(
        [
            {
                "params": filter(
                    lambda p: p.requires_grad, model[0].parameters()
                ),
                "lr": args.lr_backbone,
            },
            {
                "params": model[1].parameters(),
                "lr": args.lr_head,
            },
        ],
        weight_decay=args.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Phase 1 (warmup): linear ramp from 0.5× to 1× LR over warmup_epochs.
    # Reason: the head is randomly initialised; large early gradients can
    # corrupt the first backbone updates when unfreezing begins. Warmup
    # keeps the head LR low until it has begun to converge.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=args.warmup_epochs,
    )

    # Phase 2 (cosine): decays from base LR to eta_min=1e-7 over
    # cosine_t_max_effective steps, which starts immediately after warmup.
    # ORIGINAL BUG: cosine T_max was set to the raw cosine_t_max (e.g. 150).
    # The warmup period consumed the first warmup_epochs steps, so the cosine
    # scheduler only had (150 - 10) = 140 steps to decay — but its internal
    # period was 150, meaning it never reached eta_min within the training run.
    # FIX: set T_max = cosine_t_max - warmup_epochs so cosine reaches eta_min
    # at exactly the hard ceiling epoch.
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max_effective,
        eta_min=1e-7,
    )

    # SequentialLR: runs warmup_sched for warmup_epochs steps, then hands off
    # to cosine_sched for all remaining steps.
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    # GradScaler for AMP. Disabled on CPU (enabled=False = identity operations).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── System monitor ────────────────────────────────────────────────────────
    monitor_state = {
        "cpu_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0,
        "gpu_util": 0.0, "vram_used": 0.0, "vram_total": 0.0, "gpu_temp": 0.0,
    }
    sys_log_path = os.path.join(output_dir, f"{run_id}_system.log")
    stop_event   = threading.Event()
    monitor_thread = threading.Thread(
        target=system_monitor,
        args=(sys_log_path, stop_event, monitor_state),
        daemon=True,
        # daemon=True: thread is killed automatically if main process dies
        # (e.g. OOM, keyboard interrupt). Non-daemon threads block process exit.
    )
    monitor_thread.start()

    # ── Training state ────────────────────────────────────────────────────────
    best_auc        = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, f"{run_id}_best.pth")
    metrics_path    = os.path.join(output_dir, f"{run_id}_metrics.csv")

    with open(metrics_path, "w") as f:
        # Column name "train_acc_50_50" makes the synthetic distribution
        # explicit in any downstream CSV analysis.
        f.write(
            "epoch,train_loss,train_acc_50_50,val_loss,val_acc,"
            "auc,sensitivity,specificity,f1,epoch_time\n"
        )

    best_state    = {}
    current_state = {}   # Must be defined before run_epoch (nonlocal target)
    epoch_times   = []

    # Rolling windows for the loss-AUC decoupling detector.
    # Kept as module-scope lists (closured in run_epoch) rather than globals
    # for cleaner scoping — they are only meaningful within this training run.
    _auc_history  = []
    _loss_history = []

    global_start = time.time()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")

        # weights_only=False REQUIRED for own checkpoints on PyTorch >= 2.6.
        # REASON: PyTorch 2.6 changed the default of weights_only from False
        # to True. The checkpoint saved by this script includes
        # optimizer_state_dict, which contains internal PyTorch objects (e.g.
        # torch.Size, torch.dtype). These are not in the default safe list for
        # weights_only=True, causing an UnpicklingError at load time.
        # This file is written by this script and is a trusted source —
        # weights_only=False is safe here. Do NOT use weights_only=False on
        # third-party or user-provided checkpoints.
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            resumed_epoch = checkpoint["epoch"]
            start_epoch   = resumed_epoch + 1
            logger.info(
                f"Checkpoint was at epoch {resumed_epoch}; "
                f"resuming from epoch {start_epoch}."
            )

            # Align backbone freeze state to match the epoch we are resuming
            # from. Without this, the backbone would start fully frozen on
            # resume even if Phase 2 was already active when the run stopped.
            if resumed_epoch >= args.phase2_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.FULL, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    model[0].parameters()
                )
                logger.info("Resume: aligned to Phase FULL")

            elif resumed_epoch >= args.phase1_5_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.PARTIAL4, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    p for p in model[0].parameters() if p.requires_grad
                )
                logger.info("Resume: aligned to Phase PARTIAL4")

            elif resumed_epoch >= args.phase1_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.PARTIAL2, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    p for p in model[0].parameters() if p.requires_grad
                )
                logger.info("Resume: aligned to Phase PARTIAL2")

            else:
                logger.info("Resume: staying in Phase FROZEN (head only)")

        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state restored from checkpoint.")
            except Exception as e:
                # This can happen if the optimizer param groups changed
                # (e.g. different backbone freeze state). Log and continue;
                # the optimizer will start fresh with the current LR schedule.
                logger.warning(
                    f"Could not restore optimizer state: {e}. "
                    f"Optimizer will start fresh — LR schedule resets."
                )

        if "auc" in checkpoint:
            best_auc = checkpoint["auc"]
            logger.info(f"Resume: best AUC = {best_auc:.4f}")

    # ── Epoch function ────────────────────────────────────────────────────────
    def run_epoch(epoch):
        """
        Runs one complete train + val epoch and updates all training state.

        Declared as a nested function (closure) to avoid passing a large
        number of objects as arguments. All mutable state (best_auc,
        patience_counter, best_state, current_state) is explicitly declared
        nonlocal to avoid the Python scoping bug where assignment inside a
        function creates a LOCAL variable instead of updating the outer one.

        ORIGINAL BUG (current_state):
            nonlocal best_auc, patience_counter, best_state
            # current_state was missing from this list.
            # The line `current_state = {...}` created a new local variable.
            # The outer current_state (read by any external logger/display)
            # was never updated — it always contained the initial empty dict.
        """
        nonlocal best_auc, patience_counter, best_state, current_state

        # ── Phase transitions ──────────────────────────────────────────────
        # Each transition:
        #   1. Updates which backbone parameters have requires_grad=True
        #   2. Rebuilds the backbone optimizer param group to include only
        #      the newly-unfrozen params (previously frozen params had
        #      requires_grad=False and were excluded)
        #   3. Sets the backbone LR and resets the cosine scheduler base
        #      (see apply_phase_lr docstring for why this is necessary)
        if epoch == args.phase1_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.PARTIAL2, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                p for p in model[0].parameters() if p.requires_grad
            )
            # 0.5× backbone LR: last 2 blocks are still fragile at this stage;
            # a gentler LR prevents large weight updates that could undo
            # ImageNet feature representations accumulated over epochs 1–10.
            apply_phase_lr(
                optimizer, args.lr_backbone * 0.5, scheduler, logger
            )

        elif epoch == args.phase1_5_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.PARTIAL4, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                p for p in model[0].parameters() if p.requires_grad
            )
            # 0.75× LR: blocks 3 and 4 are deeper and more general-purpose;
            # slightly higher LR than Phase 1 to allow faster adaptation
            # without destabilising the already-fine-tuned blocks 5 and 6.
            apply_phase_lr(
                optimizer, args.lr_backbone * 0.75, scheduler, logger
            )

        elif epoch == args.phase2_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.FULL, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                model[0].parameters()
            )
            # Full backbone LR: by Phase 2 the head is stable and can absorb
            # full backbone gradients without losing learned representations.
            apply_phase_lr(optimizer, args.lr_backbone, scheduler, logger)

        # ── Train + val ────────────────────────────────────────────────────
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, acc, auc, sens, spec, f1 = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Step scheduler AFTER validation so the LR shown in the log line
        # is the LR that will be used in the NEXT epoch (not the current one).
        scheduler.step()

        # Periodic memory release: prevents VRAM fragmentation accumulating
        # over 150 epochs, which can cause OOM errors in later epochs even
        # when peak per-epoch usage is within limits.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        epoch_time = int(time.time() - t0)
        epoch_times.append(epoch_time)

        # ETA: rolling average over last 5 epochs (more stable than global avg
        # because epoch time changes when backbone is unfrozen mid-training).
        rolling   = epoch_times[-min(5, len(epoch_times)):]
        avg_ep    = sum(rolling) / len(rolling)
        remaining = max(0, (args.epochs + args.patience) - epoch)
        eta_str   = format_time(avg_ep * remaining) if remaining > 0 else "done"

        # ── Loss-AUC decoupling detector ───────────────────────────────────
        # WHAT IT DETECTS:
        #   "Ghost training" — the loss is still falling (which looks like
        #   progress) but val_AUC is no longer improving (meaning nothing
        #   useful is being learned). This happens because:
        #     1. Focal loss self-suppression reduces gradients as the model
        #        becomes confident, so the loss falls even with tiny updates.
        #     2. The model has genuinely converged and is only fine-tuning
        #        marginal confidence values without changing predictions.
        #
        # DETECTION LOGIC:
        #   If over the last 10 epochs:
        #     - val_loss moved by > 5e-4 (it is still changing)
        #     - val_AUC changed by < 0.001 (but AUC is frozen)
        #   → the loss signal has decoupled from generalisation.
        #
        # THRESHOLD RATIONALE:
        #   5e-4 loss movement: below this, loss is essentially flat too
        #   (numerical noise), so decoupling doesn't apply.
        #   0.001 AUC: changes smaller than this are within epoch-to-epoch
        #   noise and should not be counted as real improvements.
        _auc_history.append(auc)
        _loss_history.append(val_loss)
        if len(_auc_history) >= 10:
            auc_delta  = max(_auc_history[-10:]) - min(_auc_history[-10:])
            loss_delta = max(_loss_history[-10:]) - min(_loss_history[-10:])
            if loss_delta > 5e-4 and auc_delta < 0.001:
                logger.warning(
                    f"[DECOUPLE E{epoch}] val_loss moving ({loss_delta:.4f}) "
                    f"but val_AUC flat ({auc_delta:.4f}) over last 10 epochs. "
                    f"Loss is no longer a reliable training signal. "
                    f"Consider stopping if patience also indicates plateau."
                )

        # ── Update training state ──────────────────────────────────────────
        # current_state is declared nonlocal above — this assignment correctly
        # updates the outer scope variable. Without nonlocal, this line would
        # create a new local variable and the outer current_state would never
        # be updated (the original bug).
        current_state = {
            "epoch": epoch, "auc": auc,  "acc": acc,
            "sens":  sens,  "spec": spec, "f1":  f1,
            "train_loss": train_loss, "val_loss": val_loss,
        }

        is_best = auc > best_auc
        if is_best:
            best_auc   = auc
            best_state = current_state.copy()
            patience_counter = 0   # reset on any improvement, not just post-min_epochs

            # Save checkpoint: includes optimizer state for exact resume.
            # weights_only=False at load time is required (see resume block).
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc":                  auc,
                    "args":                 vars(args),
                },
                checkpoint_path,
            )
        else:
            # Only increment patience counter after min_epochs.
            # Reason: the model may still be in an active learning phase
            # (Phase 1/1.5 → Phase 2 transition) and temporarily plateau
            # before the full backbone kicks in. Premature patience would
            # stop the run before the most productive phase begins.
            if epoch >= args.min_epochs:
                patience_counter += 1

        # ── Compose log line ───────────────────────────────────────────────
        bLR = optimizer.param_groups[0]["lr"]
        hLR = optimizer.param_groups[1]["lr"]
        ms  = monitor_state   # hardware snapshot from monitor thread

        epoch_line = (
            f"E {epoch:>3} | "
            # [50/50] tag: reminds the reader that train_acc is on a synthetic
            # balanced distribution, not the real data distribution.
            f"TrL={train_loss:.4f} TrA={train_acc:.4f}[50/50] | "
            f"VlA={acc:.4f} F1={f1:.4f} "
            f"Se={sens:.4f} Sp={spec:.4f} AUC={auc:.4f} | "
            f"bLR={bLR:.6f} hLR={hLR:.6f} | "
            f"{epoch_time}s ETA={eta_str} | "
            f"pat={patience_counter}/{args.patience} | "
            f"CPU={ms['cpu_pct']:.0f}% "
            f"RAM={ms['ram_used']:.1f}GB "
            f"GPU={ms['gpu_util']:.0f}% "
            f"VRAM={ms['vram_used']/1024:.1f}GB "
            f"T={ms['gpu_temp']:.0f}C"
            + (" * BEST" if is_best else "")
        )

        # Append epoch to CSV metrics file.
        # train_acc_50_50 column name makes the synthetic distribution
        # explicit in any downstream data analysis.
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},"
                f"{acc:.4f},{auc:.4f},{sens:.4f},{spec:.4f},"
                f"{f1:.4f},{epoch_time}\n"
            )

        return epoch_line

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING | {run_id}")
    print(
        f"  min_epochs={args.min_epochs}  "
        f"patience={args.patience}  "
        f"hard_ceiling={args.epochs}"
    )
    print(f"{'='*70}\n")

    epoch = start_epoch
    while True:
        print(f"\n--- Epoch {epoch} ---", flush=True)
        epoch_line = run_epoch(epoch)
        logger.info(epoch_line)

        # STOPPING LOGIC:
        # ─────────────────
        # Early stopping: fires only after min_epochs, preventing premature
        # stops during Phase 2 ramp-up. patience_counter counts consecutive
        # epochs without val_AUC improvement (resetting on any improvement).
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            logger.info(
                f"Early stopping at epoch {epoch}: val_AUC has not improved "
                f"for {patience_counter} consecutive epochs "
                f"(min_epochs={args.min_epochs} satisfied). "
                f"Best AUC: {best_auc:.4f} (saved to {checkpoint_path})."
            )
            break

        # Hard ceiling: prevents runaway training if patience never fires
        # (e.g. AUC oscillates by tiny amounts keeping patience=0 forever).
        if epoch >= args.epochs + args.patience:
            logger.info(
                f"Hard ceiling reached at epoch {epoch} "
                f"({args.epochs} guaranteed + {args.patience} max patience). "
                f"Stopping."
            )
            break

        epoch += 1

    # ── Post-training: threshold optimisation + ONNX export ───────────────────
    stop_event.set()
    monitor_thread.join()
    logger.info(f"System hardware log: {sys_log_path}")

    # Load the best checkpoint (not the final epoch's weights).
    logger.info("Loading best checkpoint for threshold optimisation...")
    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect TTA-averaged P(ALL) scores over the full validation set.
    # These are used to find the optimal decision threshold via Youden's J.
    all_targets_t, all_scores_t = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            variants = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]
            avg_probs = torch.zeros(images.size(0), 2, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                for v in variants:
                    avg_probs += torch.softmax(model(v).float(), dim=1)
            avg_probs /= 4
            all_targets_t.extend(labels.cpu().numpy())
            all_scores_t.extend(avg_probs[:, 0].cpu().numpy())

    all_targets_t = np.array(all_targets_t)
    all_scores_t  = np.array(all_scores_t)

    # Youden's J = Sensitivity + Specificity - 1.
    # The threshold that maximises J is the optimal operating point for
    # clinical use: it balances sensitivity (catching real ALL) against
    # specificity (not over-calling HEM as ALL).
    logger.info("ALL-positive ROC threshold optimisation (Youden's J):")
    logger.info(f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'J':>8}")

    optimal_metrics = sweep_all_positive_thresholds(all_targets_t, all_scores_t)

    for thresh in np.arange(0.35, 0.76, 0.05):
        tm  = compute_all_positive_metrics(
            all_targets_t, all_scores_t, threshold=thresh
        )
        j = tm["sens"] + tm["spec"] - 1.0
        logger.info(
            f"{thresh:>10.2f} {tm['sens']:>8.4f} "
            f"{tm['spec']:>8.4f} {j:>8.4f}"
        )

    optimal_threshold = optimal_metrics["threshold"]
    logger.info(
        f"Optimal ALL threshold: {optimal_threshold:.2f} "
        f"(J={optimal_metrics['j']:.4f})"
    )

    # Save optimal threshold into the params JSON so inference pipelines can
    # load a single file to get both the hyperparameters and decision threshold.
    with open(params_path, "r") as f:
        params = json.load(f)
    params["optimal_threshold"] = float(optimal_threshold)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    # ONNX export for deployment / inference without PyTorch runtime.
    # opset_version=13: stable, widely supported by ONNX Runtime >= 1.10.
    # dynamic_axes: allows variable batch sizes at inference time without
    # re-exporting.
    logger.info("Exporting best model to ONNX...")
    onnx_path = os.path.join(output_dir, f"{run_id}_best.onnx")
    dummy = torch.randn(1, 3, args.res, args.res).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=13,
    )
    logger.info(f"ONNX saved: {onnx_path}")

    logger.info(
        f"\n[COMPLETE] "
        f"Best AUC: {best_auc:.4f} | "
        f"Threshold: {optimal_threshold:.2f} | "
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
