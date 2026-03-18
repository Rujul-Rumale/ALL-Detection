"""
train.py
========
Phase 1: Unified Training Script for ALL Leukemia Edge Classifier.
Publication-grade implementation following strict Research Engineering Plan v1.0.

Features:
  - Model training (MNV3L, EffB0/B4, ResNet50) with AMP + gradient clipping
  - CV fold execution using cv_splits_3fold.json
  - 4-way TTA validation with ROC threshold optimization (Youden's J)
  - Rich live display with hardware monitoring
  - System resource logging (CPU/RAM/GPU/VRAM/Temp)
  - Checkpoint saving + ONNX export
  - Early stopping with patience

Usage:
  python training_scripts/train.py --model mnv3l --fold 1 --run_name mnv3l_v1
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
import subprocess
from datetime import datetime

import cv2
cv2.setNumThreads(0)  # Prevent OpenCV from eating memory in DataLoader workers
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
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint.pth to resume from")
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--res", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--phase1_start", type=int, default=11)
    parser.add_argument("--phase1_5_start", type=int, default=21)
    parser.add_argument("--phase2_start", type=int, default=41)
    parser.add_argument("--cosine_t_max", type=int, default=150)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--splits_json", type=str,
                        default="cv_splits/cv_splits_3fold.json")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_live", action="store_true",
                        help="Disable Rich Live UI (use simple print lines)")

    args = parser.parse_args()
    if args.model != "mnv3l" and args.lr_head == 3e-4:
        args.lr_head = 1.5e-4
    return args


def setup_logging(output_dir, run_id):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_id}.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset & Augmentations
# ═══════════════════════════════════════════════════════════════════════════════

class CNMCDataset(Dataset):
    def __init__(self, image_label_pairs, transform=None):
        self.samples = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = cv2.imread(fpath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(fpath).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def get_transforms(res):
    train_transform = A.Compose([
        A.Resize(res + 32, res + 32),
        A.RandomCrop(res, res),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(shear=(-10, 10), scale=(0.9, 1.1), p=0.5),
        A.ElasticTransform(alpha=1, sigma=10, p=0.3, approximate=True),
        # Stain/color jitter — simulates H&E staining variation across devices
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CoarseDropout(num_holes_range=(8, 8),
                        hole_height_range=(res // 10, res // 10),
                        hole_width_range=(res // 10, res // 10), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(res, res),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricMonitor:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, use_amp, progress, task_id):
    model.train()
    monitor = MetricMonitor()
    correct = 0
    total = 0
    current_batch = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

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

        monitor.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        current_batch += 1
        progress.update(task_id, advance=1, description=f"  [yellow]Batch {current_batch}/{len(loader)}[/yellow]")

    return monitor.avg, correct / total if total > 0 else 0


def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    monitor = MetricMonitor()
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            variants = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]

            avg_probs = torch.zeros(images.size(0), 2, device=device)
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
    all_scores = np.array(all_scores)

    preds = (all_scores > 0.5).astype(int)
    acc = (preds == all_targets).mean()
    auc = roc_auc_score(all_targets, all_scores)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = sk_f1(all_targets, preds, zero_division=0)

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


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with per-class weighting.
    Penalizes hard examples exponentially, and HEM misclassification more than ALL.
    
    alpha: [alpha_ALL, alpha_HEM] - higher value penalizes that class more
    gamma: focal weight exponent (2.0 is standard)
    eps: numerical stability for clamping p_t
    """
    def __init__(self, alpha=None, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [N, 2], targets: [N] with values 0 or 1
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability of true class for each sample
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        p_t = (probs * targets_one_hot).sum(dim=1).clamp(self.eps, 1.0 - self.eps)
        
        # Focal weight: (1 - p_t)^gamma — exponentially weights hard examples
        focal_term = (1.0 - p_t) ** self.gamma
        
        # Cross-entropy part
        loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = focal_term * loss
        
        # Apply asymmetric alpha weights per class
        # alpha[0] for ALL errors, alpha[1] for HEM errors
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
            alpha_weights = alpha_t[targets]
            loss = alpha_weights * loss
        
        return loss.mean()


class ConstrainedHead(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def get_model(args):
    timm_name = TIMM_NAME_MAP[args.model]
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
    backbone.eval()
    with torch.no_grad():
        in_features = backbone(torch.zeros(2, 3, 224, 224)).shape[-1]
    backbone.train()

    model = nn.Sequential(backbone, ConstrainedHead(in_features))
    backbone_params = list(model[0].parameters())
    head_params = list(model[1].parameters())
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

    fold_data = splits["folds"][fold_key]
    train_pairs = fold_data["train_images"]
    val_pairs = fold_data["val_images"]

    train_transform, val_transform = get_transforms(args.res)
    train_ds = CNMCDataset(train_pairs, transform=train_transform)
    val_ds = CNMCDataset(val_pairs, transform=val_transform)

    labels = [p[1] for p in train_pairs]
    class_counts = np.bincount(labels)
    sample_weights = np.array([1.0 / class_counts[l] for l in labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    prefetch = 2 if args.num_workers > 0 else None
    persistent = True if args.num_workers > 0 else False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )

    normed_weights = len(class_counts) / (class_counts * 2.0)

    # Compute per-split class breakdown
    train_all = sum(1 for _, l in train_pairs if l == 0)
    train_hem = sum(1 for _, l in train_pairs if l == 1)
    val_all = sum(1 for _, l in val_pairs if l == 0)
    val_hem = sum(1 for _, l in val_pairs if l == 1)

    info = {
        "train_total": len(train_pairs), "val_total": len(val_pairs),
        "train_all": train_all, "train_hem": train_hem,
        "val_all": val_all, "val_hem": val_hem,
    }

    return (train_loader, val_loader,
            torch.tensor(normed_weights, dtype=torch.float32), class_counts, info)


# ═══════════════════════════════════════════════════════════════════════════════
#  System Monitor Thread (Part 3)
# ═══════════════════════════════════════════════════════════════════════════════

def system_monitor(log_path, stop_event, monitor_state, interval=5):
    with open(log_path, "w") as f:
        f.write("timestamp,cpu_pct,ram_used_gb,ram_total_gb,"
                "gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c\n")

    while not stop_event.is_set():
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024 ** 3
            ram_total = ram.total / 1024 ** 3

            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=timestamp,utilization.gpu,memory.used,"
                 "memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            timestamp = parts[0]
            gpu_util = parts[1]
            vram_used = parts[2]
            vram_total = parts[3]
            gpu_temp = parts[4]

            with open(log_path, "a") as f:
                f.write(f"{timestamp},{cpu_pct},{ram_used:.2f},{ram_total:.2f},"
                        f"{gpu_util},{vram_used},{vram_total},{gpu_temp}\n")

            monitor_state["cpu_pct"] = float(cpu_pct)
            monitor_state["ram_used"] = ram_used
            monitor_state["ram_total"] = ram_total
            monitor_state["gpu_util"] = float(gpu_util)
            monitor_state["vram_used"] = float(vram_used)
            monitor_state["vram_total"] = float(vram_total)
            monitor_state["gpu_temp"] = float(gpu_temp)
        except Exception:
            pass

        stop_event.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rich Display Helpers (Part 4)
# ═══════════════════════════════════════════════════════════════════════════════

def make_bar(pct, width=20):
    filled = int(width * pct / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def hardware_rows(ms):
    cpu = ms["cpu_pct"]
    ram_u = ms["ram_used"]
    ram_t = ms["ram_total"]
    gpu = ms["gpu_util"]
    vr_u = ms["vram_used"]
    vr_t = ms["vram_total"]
    temp = ms["gpu_temp"]

    cpu_c = "red" if cpu > 90 else "yellow" if cpu > 70 else "green"
    gpu_c = "red" if gpu > 95 else "yellow" if gpu > 80 else "green"
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
                  best, current, ms, elapsed_str, eta_str, bLR, hLR, pat_str, time_ep_str):
    table = Table(box=box.DOUBLE, expand=False, show_header=False,
                  padding=(0, 1), min_width=76)
    table.add_column(min_width=72)

    # Top Block
    table.add_row("  [bold cyan]ALL Leukemia Edge Classifier[/bold cyan]")
    
    run_text = f"  Run: {run_name}"
    pad = 72 - len(run_text) - len(fold_str)
    padding = " " * max(2, pad)
    table.add_row(f"{run_text}{padding}{fold_str}")
    
    table.add_row(f"  [bold]PROGRESS[/bold]   Epoch {epoch} / {total_epochs}")
    table.add_row(batch_progress)
    
    elapsed_text = f"  Elapsed: {elapsed_str}  |  Time/Ep: {time_ep_str}"
    eta_text = f"ETA: {eta_str}"
    pad = 72 - len(elapsed_text) - len(eta_text)
    padding = " " * max(2, pad)
    table.add_row(f"{elapsed_text}{padding}{eta_text}")
    
    table.add_section()

    # BEST
    if best["epoch"] > 0:
        header = f"  [bold green]BEST  (Epoch {best['epoch']})[/bold green]"
        pad = 72 - len(f"  BEST  (Epoch {best['epoch']})") - 4
        padding = " " * max(2, pad)
        table.add_row(f"{header}{padding}[yellow]★[/yellow]")
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

    # CURRENT
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

    # HARDWARE
    table.add_row("  [bold magenta]HARDWARE[/bold magenta]")
    r1, r2, r3 = hardware_rows(ms)
    table.add_row(r1)
    table.add_row(r2)
    table.add_row(r3)

    return table


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

# Warmup epochs now controlled via args.warmup_epochs (default: 10)


def set_backbone_unfreeze_phase(model, phase, logger):
    """
    Phase 0: backbone fully frozen — head only
    Phase 1: unfreeze last 2 blocks of backbone
    Phase 2: unfreeze full backbone
    """
    backbone = model[0]

    if phase == 0:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen (Phase 0 — head only)")

    elif phase == 1:
        for param in backbone.parameters():
            param.requires_grad = False
        blocks = list(backbone.children())
        for module in blocks[-2:]:
            for param in module.parameters():
                param.requires_grad = True
        logger.info("Backbone partially unfrozen — last 2 blocks (Phase 1)")

    elif phase == 1.5:
        for param in backbone.parameters():
            param.requires_grad = False
        blocks = list(backbone.children())
        for module in blocks[-4:]:
            for param in module.parameters():
                param.requires_grad = True
        logger.info("Backbone partially unfrozen — last 4 blocks (Phase 1.5)")

    elif phase == 2:
        for param in backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone fully unfrozen (Phase 2)")


def main():
    args = parse_args()
    console = Console(force_terminal=True)

    # ── Run identity ──────────────────────────────────────────────────────────
    run_id = (f"{args.run_name}_fold{args.fold}_"
              f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = os.path.join(args.output_root, args.run_name)
    logger = setup_logging(output_dir, run_id)

    # ── Device & GPU setup ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else "N/A")

    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader, loss_weights,
     class_counts, data_info) = get_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, backbone_params, head_params, timm_name, in_features = get_model(args)
    model = model.to(device)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)

    # ══════════════════════════════════════════════════════════════════════════
    #  Part 1 — Startup Block
    # ══════════════════════════════════════════════════════════════════════════
    other_folds = [f for f in [1, 2, 3] if f != args.fold]
    folds_train = "+".join(str(f) for f in other_folds)

    startup_lines = [
        "=" * 70,
        "  ALL Leukemia Edge Classifier -- train.py",
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
        f"  Warmup Epochs:   {args.warmup_epochs}",
        f"  Phase 1 start:   {args.phase1_start}",
        f"  Phase 1.5 start: {args.phase1_5_start}",
        f"  Phase 2 start:   {args.phase2_start}",
        f"  Cosine T_max:    {args.cosine_t_max}",
        f"  Cosine eta_min:  1e-7",
        f"  Patience:        {args.patience}",
        f"  AMP:             {use_amp}",
        f"  Device:          {device} ({gpu_name})",
        f"  Workers:         {args.num_workers}",
        f"  Splits JSON:     {args.splits_json}",
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
        f"  Head architecture: BN->Drop(0.4)->Linear(512)->ReLU->Drop(0.3)->Linear(2)",
        "=" * 70,
    ]
    for line in startup_lines:
        logger.info(line)

    # Save params
    params_path = os.path.join(output_dir, f"{run_id}_params.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    criterion = AsymmetricFocalLoss(
        alpha=[0.25, 0.75],  # 0.25 for ALL, 0.75 for HEM (3x penalty for HEM errors)
        gamma=2.0,           # Focal loss exponent
        eps=1e-7,
    )
    # Phase 0: freeze backbone, train head only
    set_backbone_unfreeze_phase(model, phase=0, logger=logger)

    optimizer = optim.AdamW([
        {"params": filter(lambda p: p.requires_grad,
                          model[0].parameters()), "lr": args.lr_backbone},
        {"params": model[1].parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # Warmup applies to head LR during frozen phase — head needs to stabilize
    # before backbone unfreezes at epoch 6
    warmup_sched = LinearLR(
        optimizer, start_factor=0.5, total_iters=args.warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, args.cosine_t_max), eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── System Monitor ────────────────────────────────────────────────────────
    monitor_state = {
        "cpu_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0,
        "gpu_util": 0.0, "vram_used": 0.0, "vram_total": 0.0, "gpu_temp": 0.0,
    }
    sys_log_path = os.path.join(output_dir, f"{run_id}_system.log")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=system_monitor,
        args=(sys_log_path, stop_event, monitor_state),
        daemon=True,
    )
    monitor_thread.start()

    # ── Training loop setup ───────────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, f"{run_id}_best.pth")
    metrics_path = os.path.join(output_dir, f"{run_id}_metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                "auc,sensitivity,specificity,f1,epoch_time\n")

    batch_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36, complete_style="blue", finished_style="blue"),
        TextColumn("{task.percentage:>3.0f}%"),
    )
    epoch_task = batch_progress.add_task(f"  [blue]Epochs ({args.epochs})[/blue]", total=args.epochs)
    batch_task = batch_progress.add_task(f"  [yellow]Batches ({len(train_loader)})[/yellow]", total=len(train_loader))

    _empty = {"epoch": 0, "auc": 0, "acc": 0, "sens": 0, "spec": 0,
              "f1": 0, "train_loss": 0, "val_loss": 0}
    best_state = _empty.copy()
    current_state = _empty.copy()
    epoch_times = []
    
    global_start_time = time.time()
    fold_str = f"Fold {args.fold} of 3"
    bLR = args.lr_backbone
    hLR = args.lr_head


    # ══════════════════════════════════════════════════════════════════════════
    #  Training Loop (Part 5)
    # ══════════════════════════════════════════════════════════════════════════

    # Remove stdout from logger during Live display to prevent interleaving
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
            elapsed_str = format_time(elapsed_secs)
            eta_str = ui_state["eta"]
            
            if epoch_times:
                rolling_window = epoch_times[-min(5, len(epoch_times)):]
                avg_time = sum(rolling_window) / len(rolling_window)
                eta_secs = avg_time * (args.epochs - ui_state["epoch"] + 1)
                eta_str = format_time(eta_secs)

            return build_display(
                run_id, fold_str, ui_state["epoch"], args.epochs, batch_progress,
                best_state, current_state, monitor_state,
                elapsed_str, eta_str, ui_state["bLR"], ui_state["hLR"], 
                f"{ui_state['pat']} / {args.patience}", ui_state["time_ep"]
            )

    # ── Try Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            resumed_epoch = checkpoint["epoch"]  # epoch that was saved
            start_epoch = resumed_epoch + 1

            logger.info(f"Checkpoint was saved at epoch {resumed_epoch}, "
                        f"resuming from epoch {start_epoch}")

            # Align backbone freeze state to match where we are resuming from.
            # Phase boundaries: epoch 6 = partial unfreeze, epoch 31 = full unfreeze.
            # resumed_epoch is the last completed epoch, so:
            #   if resumed_epoch >= 31: we are in or past Phase 2
            #   if resumed_epoch >= 6:  we are in Phase 1
            #   else:                   still in Phase 0
            if resumed_epoch >= args.phase2_start:
                set_backbone_unfreeze_phase(model, phase=2, logger=logger)
                optimizer.param_groups[0]["params"] = list(model[0].parameters())
                logger.info(f"Resume: aligned to Phase 2 (full unfreeze)")
            elif resumed_epoch >= args.phase1_5_start:
                set_backbone_unfreeze_phase(model, phase=1.5, logger=logger)
                optimizer.param_groups[0]["params"] = list(
                    filter(lambda p: p.requires_grad, model[0].parameters())
                )
                logger.info(f"Resume: aligned to Phase 1.5 (last 4 blocks)")
            elif resumed_epoch >= args.phase1_start:
                set_backbone_unfreeze_phase(model, phase=1, logger=logger)
                optimizer.param_groups[0]["params"] = list(
                    filter(lambda p: p.requires_grad, model[0].parameters())
                )
                logger.info(f"Resume: aligned to Phase 1 (last 2 blocks)")
            else:
                logger.info(f"Resume: staying in Phase 0 (backbone frozen)")

        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state restored from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}. "
                               f"Optimizer will start fresh.")

        if "auc" in checkpoint:
            best_auc = checkpoint["auc"]
            logger.info(f"Resume: best AUC set to {best_auc:.4f}")

    ui_state = {
        "epoch": 0, "bLR": bLR, "hLR": hLR, "pat": 0, "eta": "Calculating...",
        "time_ep": "..."
    }

    def run_epoch(epoch):
        """Run one train+val epoch. Returns (train_loss, train_acc, val_loss, acc, auc, sens, spec, f1, epoch_time, is_best)."""
        nonlocal best_auc, patience_counter, best_state
        
        # Progressive backbone unfreezing phase transitions
        if epoch == args.phase1_start:
            set_backbone_unfreeze_phase(model, phase=1, logger=logger)
            optimizer.param_groups[0]["params"] = list(
                filter(lambda p: p.requires_grad, model[0].parameters())
            )
            optimizer.param_groups[0]["lr"] = args.lr_backbone * 0.5
            logger.info(f"Phase 1: last 2 blocks unfrozen — backbone LR set to "
                        f"{args.lr_backbone * 0.5:.2e}")

        elif epoch == args.phase1_5_start:
            set_backbone_unfreeze_phase(model, phase=1.5, logger=logger)
            optimizer.param_groups[0]["params"] = list(
                filter(lambda p: p.requires_grad, model[0].parameters())
            )
            optimizer.param_groups[0]["lr"] = args.lr_backbone * 0.75
            logger.info(f"Phase 1.5: last 4 blocks unfrozen — backbone LR set to "
                        f"{args.lr_backbone * 0.75:.2e}")

        elif epoch == args.phase2_start:
            set_backbone_unfreeze_phase(model, phase=2, logger=logger)
            optimizer.param_groups[0]["params"] = list(model[0].parameters())
            optimizer.param_groups[0]["lr"] = args.lr_backbone
            logger.info(f"Phase 2: full backbone unfrozen — backbone LR set to "
                        f"{args.lr_backbone:.2e}")

        t0 = time.time()
        ui_state["epoch"] = epoch

        if not args.no_live:
            batch_progress.update(epoch_task, description=f"  [blue]Epoch {epoch}/{args.epochs}[/blue]")
            batch_progress.reset(batch_task, total=len(train_loader), description="  [yellow]Batch 0/...[/yellow]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, use_amp, batch_progress, batch_task,
        )
        if not args.no_live:
            batch_progress.advance(epoch_task)
        val_loss, acc, auc, sens, spec, f1 = validate(
            model, val_loader, criterion, device, use_amp,
        )

        scheduler.step()
        
        # Garbage collection to prevent memory fragmentation over 150 epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        epoch_time = int(time.time() - t0)
        epoch_times.append(epoch_time)
        ui_state["time_ep"] = f"{epoch_time}s"

        rolling_window = epoch_times[-min(5, len(epoch_times)):]
        avg_time = sum(rolling_window) / len(rolling_window)
        remaining = max(0, args.epochs - ui_state["epoch"])
        eta_secs = avg_time * remaining if remaining > 0 else 0
        ui_state["eta"] = format_time(eta_secs) if remaining > 0 else "patience phase"

        current_state = {
            "epoch": epoch, "auc": auc, "acc": acc,
            "sens": sens, "spec": spec, "f1": f1,
            "train_loss": train_loss, "val_loss": val_loss,
        }

        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_state = current_state.copy()
            # Reset patience counter only during patience phase
            if epoch > args.epochs:
                patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc": auc,
                "args": vars(args),
            }, checkpoint_path)
        else:
            # Only increment patience counter after guaranteed epochs
            if epoch > args.epochs:
                patience_counter += 1

        ui_state["bLR"] = optimizer.param_groups[0]["lr"]
        ui_state["hLR"] = optimizer.param_groups[1]["lr"]
        ui_state["pat"] = patience_counter

        bLR_now = optimizer.param_groups[0]["lr"]
        hLR_now = optimizer.param_groups[1]["lr"]
        best_marker = " * BEST" if is_best else ""
        epoch_line = (
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
    # Phase 1: Run all guaranteed epochs (1..args.epochs)
    # Phase 2: Continue past args.epochs; patience counter activates and
    #          training stops when patience is exhausted
    if args.no_live:
        # Simple print-based loop (no Rich overhead, ideal for benchmarking)
        print(f"\n{'='*70}")
        print(f"  TRAINING (no-live mode) | {run_id}")
        print(f"  Guaranteed epochs: {args.epochs} | Then patience: {args.patience}")
        print(f"{'='*70}")
        epoch = start_epoch
        while True:
            epoch_line, is_best, epoch_time = run_epoch(epoch)
            logger.info(epoch_line)
            # Patience only kicks in after guaranteed epochs
            if epoch > args.epochs and patience_counter >= args.patience:
                msg = (f"Early stopping at epoch {epoch} "
                       f"({epoch - args.epochs} extra epochs after {args.epochs} guaranteed).")
                logger.info(msg)
                break
            epoch += 1
    else:
        # Rich Live dashboard
        with Live(Dashboard(), refresh_per_second=4, console=console) as live:
            epoch = start_epoch
            while True:
                epoch_line, is_best, epoch_time = run_epoch(epoch)
                live.refresh()
                live.console.print(epoch_line)
                logger.info(epoch_line)
                # Patience only kicks in after guaranteed epochs
                if epoch > args.epochs and patience_counter >= args.patience:
                    msg = (f"Early stopping at epoch {epoch} "
                           f"({epoch - args.epochs} extra epochs after {args.epochs} guaranteed).")
                    live.console.print(msg)
                    logger.info(msg)
                    break
                epoch += 1

    # Restore stdout handler after training loop
    if stdout_handler:
        logger.addHandler(stdout_handler)

    # ══════════════════════════════════════════════════════════════════════════
    #  Post-Training: Threshold Optimization + ONNX Export
    # ══════════════════════════════════════════════════════════════════════════

    # Stop system monitor
    stop_event.set()
    monitor_thread.join()
    logger.info(f"System log saved to {sys_log_path}")

    # Load best model
    logger.info("Loading best model for threshold optimization...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect TTA scores
    all_targets_t = []
    all_scores_t = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            variants = [
                images, torch.flip(images, [3]),
                torch.flip(images, [2]), torch.flip(images, [2, 3]),
            ]
            avg_probs = torch.zeros(images.size(0), 2, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                for v in variants:
                    avg_probs += torch.softmax(model(v).float(), dim=1)
            avg_probs /= 4
            all_targets_t.extend(labels.cpu().numpy())
            all_scores_t.extend(avg_probs[:, 1].cpu().numpy())

    all_targets_t = np.array(all_targets_t)
    all_scores_t = np.array(all_scores_t)

    # Youden's J sweep
    logger.info("ROC threshold optimization (Youden's J):")
    logger.info(f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'J':>8}")
    best_j = -1
    optimal_threshold = 0.5
    for thresh in np.arange(0.35, 0.76, 0.05):
        preds_ = (all_scores_t > thresh).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(all_targets_t, preds_).ravel()
        se_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        sp_ = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
        j_ = se_ + sp_ - 1
        logger.info(f"{thresh:>10.2f} {se_:>8.4f} {sp_:>8.4f} {j_:>8.4f}")
        if j_ > best_j:
            best_j = j_
            optimal_threshold = thresh

    logger.info(f"Optimal threshold: {optimal_threshold:.2f} (J={best_j:.4f})")

    with open(params_path, "r") as f:
        params = json.load(f)
    params["optimal_threshold"] = float(optimal_threshold)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    # ONNX export
    logger.info("Exporting best model to ONNX...")
    onnx_path = os.path.join(output_dir, f"{run_id}_best.onnx")
    dummy = torch.randn(1, 3, args.res, args.res).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    logger.info(f"Exported to ONNX: {onnx_path}")

    logger.info(
        f"\n[COMPLETE] Best AUC: {best_auc:.4f} | "
        f"Threshold: {optimal_threshold:.2f} | "
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
