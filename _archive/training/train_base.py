"""
train_base.py  [v5 — Throughput-Optimised]
==========================================
Unified Training Script for ALL Leukemia Edge Classifier.
Publication-grade implementation following strict Research Engineering Plan v1.0.

Pipeline architecture (v5):
  CPU workers:  disk read -> BGR->RGB -> resize -> flips/rot90/brightness -> CHW tensor
  GPU pipeline: RandomCrop -> affine -> elastic -> blur -> noise -> dropout -> normalise
  CUDAPrefetcher overlaps H->D transfer of batch N+1 with GPU compute of batch N.

  Lightweight augmentations (flips, rot90, brightness/contrast) run on CPU workers
  to overlap with GPU compute. Heavy spatial transforms (affine, elastic) stay on GPU.

Quality features:
  - EMA (Exponential Moving Average) via timm.utils.ModelEmaV2 (decay=0.9998)
  - Staged TTA: 1-way on normal epochs, 8-way every 5th + final epoch
  - Channels-last memory format for faster NHWC convolutions
  - Gaussian noise augmentation (K.RandomGaussianNoise) in GPU pipeline
  - Fine threshold sweep: 0.0 to 1.0 in 0.001 steps (milestone epochs only)

Features:
  - Model training (MNV3L, EffB0/B4, ResNet50) with AMP + gradient clipping
  - CV fold execution using cv_splits_3fold.json
  - Staged TTA validation with ROC threshold optimization (Youden's J)
  - Rich live display with hardware monitoring
  - System resource logging (CPU/RAM/GPU/VRAM/Temp)
  - Checkpoint saving + ONNX export
  - Early stopping with patience

Usage:
  python training_scripts/train_base.py --model mnv3l --fold 1 --run_name mnv3l_v1
"""

import os
import sys
import json
import random
import time
import math
import argparse
import logging
import threading
import subprocess
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import psutil
from timm.utils import ModelEmaV2
import kornia.augmentation as K
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score as sk_f1
from PIL import Image
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console
from rich import box


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALL Leukemia Edge Classifier Training"
    )
    parser.add_argument("--model", type=str, default="mnv3l",
                        choices=["mnv3l", "effb0", "effb4", "rn50"])
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint.pth to resume from")
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--res", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--splits_json", type=str,
                        default="cv_splits/cv_splits_3fold.json")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_live", action="store_true",
                        help="Disable Rich Live UI (use simple print lines)")

    args = parser.parse_args()
    if args.model != "mnv3l" and args.lr_head == 2e-4:
        args.lr_head = 1e-4
    return args


def setup_logging(output_dir, run_id):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_id}.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset  --  CPU workers do: read -> resize -> cheap augments -> CHW tensor
# ═══════════════════════════════════════════════════════════════════════════════

class CNMCDataset(Dataset):
    """
    CPU dataset with lightweight augmentations that overlap with GPU compute.
    __getitem__ does:
        disk read -> BGR->RGB -> resize -> albumentations -> CHW tensor
    """
    def __init__(self, image_label_pairs, target_size: int, is_train: bool = True):
        self.samples     = image_label_pairs
        self.target_size = target_size
        self.is_train    = is_train
        
        # No albumentations. CPU handles lightweight flip/pixel augs.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]

        image = cv2.imread(fpath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(fpath).convert("RGB"))

        # Resize: INTER_AREA for downscaling quality
        image = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        # -- CPU augmentations (train only) --------------------------------
        if self.is_train:
            # Random horizontal flip
            if random.random() < 0.5:
                image = image[:, ::-1, :]  # flip cols

            # Random vertical flip
            if random.random() < 0.5:
                image = image[::-1, :, :]  # flip rows

            # Random 90/180/270 rotation
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k)

            # Random brightness/contrast jitter
            if random.random() < 0.3:
                alpha = 1.0 + random.uniform(-0.15, 0.15)  # contrast
                beta  = random.uniform(-10, 10)             # brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Gaussian blur (simulates slight defocus in microscopy)
            if random.random() < 0.2:
                ksize = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (ksize, ksize), 0)

            # Gaussian noise (simulates staining/sensor noise)
            if random.random() < 0.2:
                noise = np.random.normal(0, 3.8, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Ensure contiguous after all numpy transforms
        image = np.ascontiguousarray(image)

        # HWC -> CHW, contiguous uint8 tensor
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor, label


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Augmentation Pipelines  (Kornia)
# ═══════════════════════════════════════════════════════════════════════════════

class GPUCoarseDropout(nn.Module):
    """
    GPU-native equivalent of albumentations CoarseDropout.
    Zeroes out `num_holes` random rectangles per image in the batch.
    Operates on float tensors [B, C, H, W].
    """
    def __init__(self, num_holes: int = 8, hole_size_ratio: float = 0.1,
                 p: float = 0.2):
        super().__init__()
        self.num_holes      = num_holes
        self.hole_size_ratio = hole_size_ratio
        self.p              = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B, C, H, W = x.shape
        hole_h = max(1, int(H * self.hole_size_ratio))
        hole_w = max(1, int(W * self.hole_size_ratio))
        mask   = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)

        # Vectorized: generate all hole coords at once, no Python loops over batch
        tops  = torch.randint(0, max(1, H - hole_h), (B, self.num_holes), device=x.device)
        lefts = torch.randint(0, max(1, W - hole_w), (B, self.num_holes), device=x.device)

        # Build row/col index grids for each hole and zero them out
        rows = torch.arange(H, device=x.device).view(1, 1, H, 1)
        cols = torch.arange(W, device=x.device).view(1, 1, 1, W)
        for i in range(self.num_holes):
            t = tops[:, i].view(B, 1, 1, 1)
            l = lefts[:, i].view(B, 1, 1, 1)
            in_hole = (rows >= t) & (rows < t + hole_h) & \
                      (cols >= l) & (cols < l + hole_w)
            mask[in_hole.expand_as(mask)] = 0.0

        return x * mask


class SafeNormalize(nn.Module):
    """
    Drop-in replacement for K.Normalize that uses .reshape() instead of .view().
    Kornia's spatial transforms (RandomCrop, RandomAffine, etc.) produce
    non-contiguous tensors, causing K.Normalize's internal .view() to crash.
    This module is functionally identical but layout-safe.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std",  std.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def get_gpu_transforms(res: int, device: torch.device):
    """
    Returns (train_tf, val_tf) -- nn.Sequential modules on `device`.
    Operates on float [0, 1] CHW tensors.

    GPU-only (heavy spatial transforms):
        RandomElasticTransform, SafeNormalize.

    All other transforms run on CPU in CNMCDataset.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)

    train_tf = nn.Sequential(
        # -- Spatial (heavy) ------------------------------------------------
        K.RandomCrop((res, res)),
        K.RandomAffine(
            degrees=0,
            shear=[-10.0, 10.0, -10.0, 10.0],
            scale=[0.9, 1.1],
            p=0.5,
        ),
        K.RandomElasticTransform(
            kernel_size=(63, 63),
            sigma=(10.0, 10.0),
            alpha=(1.0, 1.0),
            p=0.3,
        ),
        # CoarseDropout applied separately after this Sequential
        # -- Normalise (reshape-safe) --------------------------------------
        SafeNormalize(mean=mean, std=std),
    ).to(device)

    # Validation: just normalise.  TTA flips are handled manually in validate().
    val_tf = nn.Sequential(
        SafeNormalize(mean=mean, std=std),
    ).to(device)

    return train_tf, val_tf


# ═══════════════════════════════════════════════════════════════════════════════
#  CUDA Prefetcher
# ═══════════════════════════════════════════════════════════════════════════════

class CUDAPrefetcher:
    """
    Overlaps H→D transfer of batch N+1 with GPU compute of batch N.

    Uses a dedicated CUDA stream so the copy runs in parallel with the
    default stream's forward/backward pass.  Requires pin_memory=True
    and non_blocking=True in the DataLoader — both already set.

    Also performs the uint8→float32 / 255 conversion in the copy stream,
    so train_one_epoch and validate receive ready-to-use float tensors.
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_images: torch.Tensor | None = None
        self.next_labels: torch.Tensor | None = None
        self._iter = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        # Wait for the copy stream to finish before handing tensors to caller
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        images = self.next_images
        labels = self.next_labels

        if images is None:
            raise StopIteration

        # Record so PyTorch knows these tensors were produced in copy stream
        if images is not None:
            images.record_stream(torch.cuda.current_stream(self.device))
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream(self.device))

        self._preload()
        return images, labels

    def _preload(self):
        try:
            images, labels = next(self._iter)
        except StopIteration:
            self.next_images = None
            self.next_labels = None
            return

        with torch.cuda.stream(self.stream):
            # Transfer + type conversion happen on the copy stream
            self.next_images = images.to(
                self.device, non_blocking=True
            ).float().div_(255.0).contiguous()
            self.next_labels = labels.to(self.device, non_blocking=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricMonitor:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val  = val
        self.sum  += val * n
        self.count += n
        self.avg  = self.sum / self.count


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, use_amp, train_tf, coarse_dropout,
                    model_ema, progress, task_id):
    model.train()
    coarse_dropout.train()
    monitor = MetricMonitor()
    correct = 0
    total   = 0
    current_batch = 0

    for images, labels in loader:
        # ── GPU augmentation ─────────────────────────────────────────────────
        # Kornia requires contiguous NCHW tensors; .contiguous() guarantees this
        images = train_tf(images.contiguous())
        images = coarse_dropout(images)

        # Apply channels_last AFTER Kornia so convolutions are fast
        images = images.to(memory_format=torch.channels_last)

        # ── Forward / backward ───────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model_ema.update(model)

        monitor.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
        current_batch += 1
        progress.update(
            task_id, advance=1,
            description=f"  [yellow]Batch {current_batch}/{len(loader)}[/yellow]",
        )

    return monitor.avg, correct / total if total > 0 else 0


def validate(model, loader, criterion, device, use_amp, val_tf, tta_ways=8):
    """
    Validate with configurable TTA.
    tta_ways=1: no TTA (fast, for intermediate epochs)
    tta_ways=8: full 8-way TTA (accurate, for milestone epochs)
    """
    model.eval()
    monitor     = MetricMonitor()
    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for images, labels in loader:
            # Normalise. Kornia needs contiguous tensors.
            images = val_tf(images.contiguous())

            # Apply channels_last AFTER Kornia so convolutions are fast
            images = images.to(memory_format=torch.channels_last)

            if tta_ways >= 8:
                # 8-way TTA: original + 3 flips + rot90/180/270 + rot90+hflip
                variants = [
                    images,                                          # original
                    torch.flip(images, [3]),                         # h-flip
                    torch.flip(images, [2]),                         # v-flip
                    torch.flip(images, [2, 3]),                      # hv-flip
                    torch.rot90(images, 1, dims=[2, 3]),             # rot90
                    torch.rot90(images, 2, dims=[2, 3]),             # rot180
                    torch.rot90(images, 3, dims=[2, 3]),             # rot270
                    torch.flip(torch.rot90(images, 1, dims=[2, 3]),
                               [3]),                                  # rot90+hflip
                ]
            elif tta_ways >= 4:
                # 4-way TTA: original + h-flip + v-flip + hv-flip
                variants = [
                    images,
                    torch.flip(images, [3]),
                    torch.flip(images, [2]),
                    torch.flip(images, [2, 3]),
                ]
            else:
                # 1-way: no TTA, just the original
                variants = [images]

            avg_probs   = torch.zeros(images.size(0), 2, device=device)
            logits_orig = None

            with torch.amp.autocast("cuda", enabled=use_amp):
                for i, v in enumerate(variants):
                    logits = model(v)
                    if i == 0:
                        logits_orig = logits
                    avg_probs += torch.softmax(logits.float(), dim=1)

            avg_probs /= len(variants)
            loss = criterion(logits_orig.float(), labels)

            monitor.update(loss.item(), images.size(0))
            all_targets.extend(labels.cpu().numpy())
            all_scores.extend(avg_probs[:, 1].cpu().numpy())

    all_targets = np.array(all_targets)
    all_scores  = np.array(all_scores)

    preds       = (all_scores > 0.5).astype(int)
    acc         = (preds == all_targets).mean()
    auc         = roc_auc_score(all_targets, all_scores)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1          = sk_f1(all_targets, preds, zero_division=0)

    return monitor.avg, acc, auc, sensitivity, specificity, f1


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

TIMM_NAME_MAP = {
    "mnv3l": "mobilenetv3_large_100",
    "effb0": "efficientnet_b0",
    "effb4": "efficientnet_b4",
    "rn50":  "resnet50",
}


class ConstrainedHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=2, dropout=0.6):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def get_model(args):
    timm_name = TIMM_NAME_MAP[args.model]
    backbone  = timm.create_model(timm_name, pretrained=True, num_classes=0)
    backbone.eval()
    with torch.no_grad():
        in_features = backbone(torch.zeros(2, 3, 224, 224)).shape[-1]
    backbone.train()

    model          = nn.Sequential(backbone, ConstrainedHead(in_features, dropout=0.6))
    backbone_params = list(model[0].parameters())
    head_params     = list(model[1].parameters())
    return model, backbone_params, head_params, timm_name, in_features


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def get_loaders(args):
    with open(args.splits_json, "r") as f:
        splits = json.load(f)
    fold_key = f"fold_{args.fold}"
    if fold_key not in splits["folds"]:
        raise ValueError(f"Fold {args.fold} not found in {args.splits_json}")

    fold_data  = splits["folds"][fold_key]
    train_pairs = fold_data["train_images"]
    val_pairs   = fold_data["val_images"]

    # CPU datasets with lightweight augmentations
    # Train: resize to res+64 for better RandomCrop diversity on GPU
    # Val:   resize to res exactly (no augmentation)
    train_ds = CNMCDataset(train_pairs, target_size=args.res + 64, is_train=True)
    val_ds   = CNMCDataset(val_pairs,   target_size=args.res,      is_train=False)

    labels         = [p[1] for p in train_pairs]
    class_counts   = np.bincount(labels)
    sample_weights = np.array([1.0 / class_counts[l] for l in labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    prefetch   = 4 if args.num_workers > 0 else None   # increased from 2
    persistent = True if args.num_workers > 0 else False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )

    normed_weights = len(class_counts) / (class_counts * 2.0)

    train_all = sum(1 for _, l in train_pairs if l == 0)
    train_hem = sum(1 for _, l in train_pairs if l == 1)
    val_all   = sum(1 for _, l in val_pairs   if l == 0)
    val_hem   = sum(1 for _, l in val_pairs   if l == 1)

    info = {
        "train_total": len(train_pairs), "val_total": len(val_pairs),
        "train_all": train_all, "train_hem": train_hem,
        "val_all": val_all,     "val_hem": val_hem,
    }

    return (train_loader, val_loader,
            torch.tensor(normed_weights, dtype=torch.float32),
            class_counts, info)


# ═══════════════════════════════════════════════════════════════════════════════
#  System Monitor Thread
# ═══════════════════════════════════════════════════════════════════════════════

def system_monitor(log_path, stop_event, monitor_state, interval=5):
    with open(log_path, "w") as f:
        f.write("timestamp,cpu_pct,ram_used_gb,ram_total_gb,"
                "gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c\n")

    while not stop_event.is_set():
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram     = psutil.virtual_memory()
            ram_used  = ram.used  / 1024 ** 3
            ram_total = ram.total / 1024 ** 3

            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=timestamp,utilization.gpu,memory.used,"
                 "memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            parts     = [p.strip() for p in result.stdout.strip().split(",")]
            timestamp = parts[0]
            gpu_util  = parts[1]
            vram_used = parts[2]
            vram_total = parts[3]
            gpu_temp  = parts[4]

            with open(log_path, "a") as f:
                f.write(
                    f"{timestamp},{cpu_pct},{ram_used:.2f},{ram_total:.2f},"
                    f"{gpu_util},{vram_used},{vram_total},{gpu_temp}\n"
                )

            monitor_state["cpu_pct"]   = float(cpu_pct)
            monitor_state["ram_used"]  = ram_used
            monitor_state["ram_total"] = ram_total
            monitor_state["gpu_util"]  = float(gpu_util)
            monitor_state["vram_used"] = float(vram_used)
            monitor_state["vram_total"] = float(vram_total)
            monitor_state["gpu_temp"]  = float(gpu_temp)
        except Exception:
            pass

        stop_event.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rich Display Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_bar(pct, width=20):
    filled = int(width * pct / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def hardware_rows(ms):
    cpu   = ms["cpu_pct"]
    ram_u = ms["ram_used"]
    ram_t = ms["ram_total"]
    gpu   = ms["gpu_util"]
    vr_u  = ms["vram_used"]
    vr_t  = ms["vram_total"]
    temp  = ms["gpu_temp"]

    cpu_c = "red" if cpu  > 90 else "yellow" if cpu  > 70 else "green"
    gpu_c = "red" if gpu  > 95 else "yellow" if gpu  > 80 else "green"
    tmp_c = "red" if temp > 85 else "yellow" if temp > 75 else "green"

    r1 = (f"  CPU:  {cpu:>2.0f}%  [bold {cpu_c}]{make_bar(cpu)}[/bold {cpu_c}]   "
          f"RAM:  {ram_u:>5.1f} / {ram_t:>4.1f} GB")
    r2 = (f"  GPU:  {gpu:>2.0f}%  [bold {gpu_c}]{make_bar(gpu)}[/bold {gpu_c}]   "
          f"VRAM: {vr_u / 1024:>5.1f} / {vr_t / 1024:>4.1f} GB")
    r3 = f"  GPU Temp: [{tmp_c}]{temp:.0f}°C[/{tmp_c}]"
    return r1, r2, r3


def format_time(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def build_display(run_name, fold_str, epoch, total_epochs, batch_progress,
                  best, current, ms, elapsed_str, eta_str,
                  bLR, hLR, pat_str, time_ep_str):
    table = Table(box=box.DOUBLE, expand=False, show_header=False,
                  padding=(0, 1), min_width=76)
    table.add_column(min_width=72)

    table.add_row("  [bold cyan]ALL Leukemia Edge Classifier[/bold cyan]")
    run_text = f"  Run: {run_name}"
    pad      = 72 - len(run_text) - len(fold_str)
    table.add_row(f"{run_text}{' ' * max(2, pad)}{fold_str}")
    table.add_row(f"  [bold]PROGRESS[/bold]   Epoch {epoch} / {total_epochs}")
    table.add_row(batch_progress)
    elapsed_text = f"  Elapsed: {elapsed_str}  |  Time/Ep: {time_ep_str}"
    eta_text     = f"ETA: {eta_str}"
    pad = 72 - len(elapsed_text) - len(eta_text)
    table.add_row(f"{elapsed_text}{' ' * max(2, pad)}{eta_text}")
    table.add_section()

    if best["epoch"] > 0:
        header = f"  [bold green]BEST  (Epoch {best['epoch']})[/bold green]"
        pad    = 72 - len(f"  BEST  (Epoch {best['epoch']})") - 4
        table.add_row(f"{header}{' ' * max(2, pad)}[yellow]★[/yellow]")
        table.add_row(
            f"  AUC:  [green]{best['auc']:.4f}[/green]    Acc:  {best['acc']:.4f}    "
            f"Sens: {best['sens']:.4f}    Spec: {best['spec']:.4f}"
        )
        table.add_row(
            f"  F1:   {best['f1']:.4f}    TrLoss: {best['train_loss']:.4f}  "
            f"VlLoss: {best['val_loss']:.4f}"
        )
    else:
        table.add_row("  [bold green]BEST[/bold green]")
        table.add_row("  Waiting for first epoch...")
        table.add_row("")
    table.add_section()

    if current["epoch"] > 0:
        table.add_row(f"  [bold yellow]CURRENT  (Epoch {current['epoch']})[/bold yellow]")
        table.add_row(
            f"  AUC:  [yellow]{current['auc']:.4f}[/yellow]    Acc:  {current['acc']:.4f}    "
            f"F1: {current['f1']:.4f}"
        )
        table.add_row(
            f"  TrLoss: {current['train_loss']:.4f}    "
            f"bLR: {bLR:.6f}   hLR: {hLR:.6f}   Pat: {pat_str}"
        )
    else:
        table.add_row("  [bold yellow]CURRENT[/bold yellow]")
        table.add_row("  Training...")
        table.add_row(f"  bLR: {bLR:.6f}   hLR: {hLR:.6f}   Pat: {pat_str}")
    table.add_section()

    table.add_row("  [bold magenta]HARDWARE[/bold magenta]")
    r1, r2, r3 = hardware_rows(ms)
    table.add_row(r1)
    table.add_row(r2)
    table.add_row(r3)

    return table


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

WARMUP_EPOCHS = 5


def main():
    args    = parse_args()
    console = Console(force_terminal=True)

    run_id     = (f"{args.run_name}_fold{args.fold}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = os.path.join(args.output_root, args.run_name)
    logger     = setup_logging(output_dir, run_id)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "N/A")

    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")  # better tensor core utilisation
    use_amp = device.type == "cuda"

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader, loss_weights,
     class_counts, data_info) = get_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, backbone_params, head_params, timm_name, in_features = get_model(args)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head     = sum(p.numel() for p in head_params)

    # ── GPU Augmentation Pipelines ────────────────────────────────────────────
    train_tf, val_tf = get_gpu_transforms(args.res, device)
    coarse_dropout   = GPUCoarseDropout(
        num_holes=8, hole_size_ratio=0.1, p=0.2
    ).to(device)

    # Compile the MODEL (not augmentation transforms) for kernel fusion.
    # Kornia transforms have dynamic shapes that cause graph breaks.
    if device.type == "cuda" and sys.platform != "win32":
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile applied to model (mode=reduce-overhead)")
    elif device.type == "cuda":
        logger.info("torch.compile skipped on Windows (no cl.exe)")

    # ── EMA ───────────────────────────────────────────────────────────────────
    # Maintains a smoothed copy of model weights — evaluated instead of the
    # raw model. decay=0.9998 is standard for 150-epoch medical imaging runs.
    model_ema = ModelEmaV2(model, decay=0.9998, device=device)

    # ── Startup Block ─────────────────────────────────────────────────────────
    other_folds  = [f for f in [1, 2, 3] if f != args.fold]
    folds_train  = "+".join(str(f) for f in other_folds)

    startup_lines = [
        "=" * 70,
        "  ALL Leukemia Edge Classifier -- train_base.py  [GPU-Optimised v4]",
        f"  Run ID:          {run_id}",
        f"  Model:           {args.model}",
        f"  Fold:            {args.fold} (val) | Folds {folds_train} (train)",
        f"  Resolution:      {args.res}x{args.res}",
        f"  Batch Size:      {args.batch_size}",
        f"  Epochs:          {args.epochs}",
        f"  LR Backbone:     {args.lr_backbone}",
        f"  LR Head:         {args.lr_head:.0e}",
        f"  Weight Decay:    {args.weight_decay:.0e}",
        f"  Label Smoothing: {args.label_smoothing}",
        f"  Warmup Epochs:   {WARMUP_EPOCHS}",
        f"  Patience:        {args.patience}",
        f"  AMP:             {use_amp}",
        f"  Device:          {device} ({gpu_name})",
        f"  Workers:         {args.num_workers}",
        f"  Splits JSON:     {args.splits_json}",
        f"  Augmentation:    GPU (Kornia) — CPU workers do resize only",
        f"  Prefetch factor: 4",
        "=" * 70,
        "  Dataset",
        (f"  Train images:  {data_info['train_total']}  "
         f"(ALL: {data_info['train_all']}, HEM: {data_info['train_hem']})"),
        (f"  Val images:    {data_info['val_total']}  "
         f"(ALL: {data_info['val_all']}, HEM: {data_info['val_hem']})"),
        (f"  Loss weights:  ALL={loss_weights[0]:.6f}, "
         f"HEM={loss_weights[1]:.6f}"),
        "=" * 70,
        "  Model",
        f"  Backbone:        {timm_name}",
        f"  Backbone params: {n_backbone:>10,}",
        f"  Head params:     {n_head:>10,}",
        f"  Total params:    {n_backbone + n_head:>10,}",
        f"  Head dropout:    0.6",
        f"  Head hidden dim: 128",
        "=" * 70,
    ]
    for line in startup_lines:
        logger.info(line)

    params_path = os.path.join(output_dir, f"{run_id}_params.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
        weight=loss_weights.to(device),
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - WARMUP_EPOCHS)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── System Monitor ────────────────────────────────────────────────────────
    monitor_state = {
        "cpu_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0,
        "gpu_util": 0.0, "vram_used": 0.0, "vram_total": 0.0,
        "gpu_temp": 0.0,
    }
    sys_log_path  = os.path.join(output_dir, f"{run_id}_system.log")
    stop_event    = threading.Event()
    monitor_thread = threading.Thread(
        target=system_monitor,
        args=(sys_log_path, stop_event, monitor_state),
        daemon=True,
    )
    monitor_thread.start()

    # ── Training loop setup ───────────────────────────────────────────────────
    best_auc        = 0.0
    patience_counter = 0
    checkpoint_path  = os.path.join(output_dir, f"{run_id}_best.pth")
    metrics_path     = os.path.join(output_dir, f"{run_id}_metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                "auc,sensitivity,specificity,f1,epoch_time\n")

    batch_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36, complete_style="blue", finished_style="blue"),
        TextColumn("{task.percentage:>3.0f}%"),
    )
    epoch_task = batch_progress.add_task(
        f"  [blue]Epochs ({args.epochs})[/blue]", total=args.epochs
    )
    batch_task = batch_progress.add_task(
        f"  [yellow]Batches ({len(train_loader)})[/yellow]",
        total=len(train_loader),
    )

    _empty = {"epoch": 0, "auc": 0, "acc": 0, "sens": 0, "spec": 0,
              "f1": 0, "train_loss": 0, "val_loss": 0}
    best_state    = _empty.copy()
    current_state = _empty.copy()
    epoch_times   = []

    global_start_time = time.time()
    fold_str = f"Fold {args.fold} of 3"
    bLR = args.lr_backbone
    hLR = args.lr_head

    # Remove stdout handler during Live display
    stdout_handler = None
    if not args.no_live:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
                stdout_handler = h
                break
        if stdout_handler:
            logger.removeHandler(stdout_handler)
        console.clear()

    class Dashboard:
        def __rich__(self):
            elapsed_secs = int(time.time() - global_start_time)
            elapsed_str  = format_time(elapsed_secs)
            eta_str      = ui_state["eta"]
            if epoch_times:
                rolling_window = epoch_times[-min(5, len(epoch_times)):]
                avg_time = sum(rolling_window) / len(rolling_window)
                eta_secs = avg_time * (args.epochs - ui_state["epoch"] + 1)
                eta_str  = format_time(eta_secs)
            return build_display(
                run_id, fold_str, ui_state["epoch"], args.epochs,
                batch_progress, best_state, current_state, monitor_state,
                elapsed_str, eta_str, ui_state["bLR"], ui_state["hLR"],
                f"{ui_state['pat']} / {args.patience}", ui_state["time_ep"],
            )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "auc" in checkpoint:
            best_auc = checkpoint["auc"]

    ui_state = {
        "epoch": 0, "bLR": bLR, "hLR": hLR, "pat": 0,
        "eta": "Calculating...", "time_ep": "...",
    }

    # ── Wrap loaders with CUDA prefetcher ─────────────────────────────────────
    # Prefetcher only works on CUDA; fall back to plain loaders on CPU
    if device.type == "cuda":
        train_prefetch = CUDAPrefetcher(train_loader, device)
        val_prefetch   = CUDAPrefetcher(val_loader,   device)
        logger.info("CUDAPrefetcher enabled - async H->D transfer active")
    else:
        train_prefetch = train_loader
        val_prefetch   = val_loader
        logger.info("CUDAPrefetcher disabled - CPU device detected")

    def run_epoch(epoch):
        nonlocal best_auc, patience_counter, best_state
        t0             = time.time()
        ui_state["epoch"] = epoch

        if not args.no_live:
            batch_progress.update(
                epoch_task,
                description=f"  [blue]Epoch {epoch}/{args.epochs}[/blue]",
            )
            batch_progress.reset(
                batch_task, total=len(train_loader),
                description="  [yellow]Batch 0/...[/yellow]",
            )

        train_loss, train_acc = train_one_epoch(
            model, train_prefetch, criterion, optimizer,
            scaler, device, use_amp,
            train_tf, coarse_dropout, model_ema,
            batch_progress, batch_task,
        )
        if not args.no_live:
            batch_progress.advance(epoch_task)

        # Staged validation: 3-tier TTA policy
        #   Final epoch: 8-way TTA (full accuracy)
        #   Every 5th:   4-way TTA (good accuracy, moderate cost)
        #   Otherwise:   1-way     (fast, for monitoring trends)
        if epoch == args.epochs:
            tta_ways = 8
        elif epoch % 5 == 0:
            tta_ways = 4
        else:
            tta_ways = 1

        # Rebuild val prefetcher each epoch -- CUDAPrefetcher is a one-shot iterator
        val_loss, acc, auc, sens, spec, f1 = validate(
            model_ema.module,
            CUDAPrefetcher(val_loader, device),
            criterion, device, use_amp, val_tf,
            tta_ways=tta_ways,
        )

        scheduler.step()
        epoch_time     = int(time.time() - t0)
        epoch_times.append(epoch_time)
        ui_state["time_ep"] = f"{epoch_time}s"

        rolling_window = epoch_times[-min(5, len(epoch_times)):]
        avg_time = sum(rolling_window) / len(rolling_window)
        eta_secs = avg_time * (args.epochs - epoch)
        ui_state["eta"] = format_time(eta_secs) if epoch < args.epochs else "0s"

        current_state_local = {
            "epoch": epoch, "auc": auc, "acc": acc,
            "sens": sens,   "spec": spec, "f1": f1,
            "train_loss": train_loss, "val_loss": val_loss,
        }

        is_best = auc > best_auc
        if is_best:
            best_auc   = auc
            best_state = current_state_local.copy()
            patience_counter = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "ema_state_dict":      model_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc":                 auc,
                "args":                vars(args),
            }, checkpoint_path)
        else:
            patience_counter += 1

        ui_state["bLR"] = optimizer.param_groups[0]["lr"]
        ui_state["hLR"] = optimizer.param_groups[1]["lr"]
        ui_state["pat"] = patience_counter

        bLR_now = optimizer.param_groups[0]["lr"]
        hLR_now = optimizer.param_groups[1]["lr"]
        best_marker = " * BEST" if is_best else ""
        epoch_line  = (
            f"E {epoch:>3}/{args.epochs} | "
            f"TrL={train_loss:.4f} TrA={train_acc:.4f} | "
            f"VlA={acc:.4f} F1={f1:.4f} "
            f"Se={sens:.4f} Sp={spec:.4f} AUC={auc:.4f} | "
            f"bLR={bLR_now:.6f} hLR={hLR_now:.6f} | "
            f"{epoch_time}s | ETA {ui_state['eta']} | "
            f"pat={patience_counter}/{args.patience}"
            f"{best_marker}"
        )

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},"
                f"{acc:.4f},{auc:.4f},{sens:.4f},{spec:.4f},"
                f"{f1:.4f},{epoch_time}\n"
            )

        return epoch_line, is_best, epoch_time

    # ── Training Loop ─────────────────────────────────────────────────────────
    if args.no_live:
        print(f"\n{'='*70}")
        print(f"  TRAINING (no-live mode) | {run_id}")
        print(f"{'='*70}")
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_line, is_best, epoch_time = run_epoch(epoch)
            print(epoch_line)
            logger.info(epoch_line)
            if patience_counter >= args.patience and epoch >= 80:
                msg = f"Early stopping at epoch {epoch}."
                print(msg)
                logger.info(msg)
                break
    else:
        with Live(Dashboard(), refresh_per_second=4, console=console) as live:
            for epoch in range(start_epoch, args.epochs + 1):
                epoch_line, is_best, epoch_time = run_epoch(epoch)
                live.refresh()
                live.console.print(epoch_line)
                logger.info(epoch_line)
                if patience_counter >= args.patience and epoch >= 80:
                    msg = f"Early stopping at epoch {epoch}."
                    live.console.print(msg)
                    logger.info(msg)
                    break

    if stdout_handler:
        logger.addHandler(stdout_handler)

    # ══════════════════════════════════════════════════════════════════════════
    #  Post-Training: Threshold Optimisation + ONNX Export
    # ══════════════════════════════════════════════════════════════════════════

    stop_event.set()
    monitor_thread.join()
    logger.info(f"System log saved to {sys_log_path}")

    logger.info("Loading best EMA model for threshold optimisation...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_ema.load_state_dict(ckpt["ema_state_dict"])
    eval_model = model_ema.module.eval()

    all_targets_t = []
    all_scores_t  = []
    with torch.no_grad():
        for images, labels in CUDAPrefetcher(val_loader, device):
            images = images.to(memory_format=torch.channels_last)
            images = val_tf(images)
            variants = [
                images,                                          # original
                torch.flip(images, [3]),                        # h-flip
                torch.flip(images, [2]),                        # v-flip
                torch.flip(images, [2, 3]),                     # hv-flip
                torch.rot90(images, 1, dims=[2, 3]),            # rot90
                torch.rot90(images, 2, dims=[2, 3]),            # rot180
                torch.rot90(images, 3, dims=[2, 3]),            # rot270
                torch.flip(torch.rot90(images, 1, dims=[2, 3]),
                           [3]),                                 # rot90+hflip
            ]
            avg_probs = torch.zeros(images.size(0), 2, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                for v in variants:
                    avg_probs += torch.softmax(eval_model(v).float(), dim=1)
            avg_probs /= len(variants)
            all_targets_t.extend(labels.cpu().numpy())
            all_scores_t.extend(avg_probs[:, 1].cpu().numpy())

    all_targets_t = np.array(all_targets_t)
    all_scores_t  = np.array(all_scores_t)

    logger.info("ROC threshold optimisation (Youden's J) — fine sweep 0.001 steps:")
    logger.info(f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'J':>8}")
    best_j             = -1
    best_acc           = 0.0
    optimal_threshold  = 0.5
    best_acc_threshold = 0.5
    for thresh in np.arange(0.0, 1.001, 0.001):
        preds_ = (all_scores_t > thresh).astype(int)
        cm_    = confusion_matrix(all_targets_t, preds_, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm_.ravel()
        se_  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        sp_  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
        acc_ = (tp_ + tn_) / len(all_targets_t)
        j_   = se_ + sp_ - 1
        if j_ > best_j:
            best_j            = j_
            optimal_threshold = thresh
        if acc_ > best_acc:
            best_acc           = acc_
            best_acc_threshold = thresh

    # Log only coarse steps to keep log readable; full data in params JSON
    for thresh in np.arange(0.30, 0.81, 0.05):
        preds_ = (all_scores_t > thresh).astype(int)
        cm_    = confusion_matrix(all_targets_t, preds_, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm_.ravel()
        se_  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        sp_  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
        acc_ = (tp_ + tn_) / len(all_targets_t)
        j_   = se_ + sp_ - 1
        logger.info(f"{thresh:>10.2f} {se_:>8.4f} {sp_:>8.4f} {acc_:>8.4f} {j_:>8.4f}")

    logger.info(f"Optimal threshold (Youden J):  {optimal_threshold:.3f} "
                f"(J={best_j:.4f})")
    logger.info(f"Optimal threshold (Accuracy):  {best_acc_threshold:.3f} "
                f"(Acc={best_acc:.4f})")

    with open(params_path, "r") as f:
        params = json.load(f)
    params["optimal_threshold_youden"]   = round(float(optimal_threshold), 3)
    params["optimal_threshold_accuracy"] = round(float(best_acc_threshold), 3)
    params["best_val_acc_at_threshold"]  = round(float(best_acc), 4)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    logger.info("Exporting best EMA model to ONNX...")
    onnx_path = os.path.join(output_dir, f"{run_id}_best.onnx")
    dummy     = torch.randn(1, 3, args.res, args.res).to(device)
    torch.onnx.export(
        eval_model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    logger.info(f"Exported to ONNX: {onnx_path}")
    logger.info(
        f"\n[COMPLETE] Best AUC: {best_auc:.4f} | "
        f"Youden threshold: {optimal_threshold:.3f} | "
        f"Accuracy threshold: {best_acc_threshold:.3f} (Acc={best_acc:.4f}) | "
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()