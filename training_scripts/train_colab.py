"""
train_colab.py  [GPU-Optimised]
==============================
High-throughput training script for Google Colab.
Uses Kornia for GPU-accelerated augmentations and CUDAPrefetcher for async I/O.
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
        description="ALL Leukemia Edge Classifier Training (Colab Optimised)"
    )
    parser.add_argument("--model", type=str, default="mnv3l",
                        choices=["mnv3l", "effb0", "effb4", "rn50"])
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint.pth to resume from")
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--res", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=128)
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
    def __init__(self, image_label_pairs, target_size: int, is_train: bool = True):
        self.samples     = image_label_pairs
        self.target_size = target_size
        self.is_train    = is_train

    def __len__(self):
        return len(self.samples)

    _ROOT_CACHE = None

    def _normalize_path(self, path: str):
        # Convert any style path to Linux/Forward slash
        orig_path = path.replace("\\", "/")
        
        # 1. Try absolute path directly (if matches perfectly)
        if os.path.exists(orig_path):
            return orig_path
            
        # 2. Lazy discovery of Kaggle/Colab root
        if CNMCDataset._ROOT_CACHE is None:
            print("🔍 Discovering dataset root...")
            candidates = [
                "/kaggle/input",
                "/content/drive/MyDrive",
                "/content",
                os.getcwd(),
                os.path.dirname(os.getcwd())
            ]
            for root_cand in candidates:
                if not os.path.exists(root_cand): continue
                # Search for the anchor folder
                for root, dirs, _ in os.walk(root_cand):
                    if "C-NMC_training_data" in dirs:
                        CNMCDataset._ROOT_CACHE = root
                        print(f"✅ Discovered Root: {root}")
                        break
                if CNMCDataset._ROOT_CACHE: break
            
            if CNMCDataset._ROOT_CACHE is None:
                CNMCDataset._ROOT_CACHE = "" # Avoid re-scanning

        # 3. Extract relative part and join with discovered root
        anchors = ["C-NMC_training_data", "C-NMC_test_prelim_phase_data"]
        for anchor in anchors:
            if anchor in orig_path:
                rel_part = anchor + orig_path.split(anchor)[-1]
                if CNMCDataset._ROOT_CACHE:
                    # In some environments, the anchor is ALREADY inside the cache
                    # e.g. Cache = .../C-NMC 2019 (PKG), Rel = C-NMC_training_data/...
                    cand = os.path.join(CNMCDataset._ROOT_CACHE, rel_part)
                    if os.path.exists(cand): return cand
                    # Try parent if cache was a subfolder
                    cand = os.path.join(os.path.dirname(CNMCDataset._ROOT_CACHE), rel_part)
                    if os.path.exists(cand): return cand
                break
                    
        return orig_path

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        fpath = self._normalize_path(fpath)

        # Attempt to read with OpenCV
        image = cv2.imread(fpath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(fpath).convert("RGB"))

        image = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.is_train:
            if random.random() < 0.5: image = image[:, ::-1, :]
            if random.random() < 0.5: image = image[::-1, :, :]
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k)
            if random.random() < 0.3:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta  = random.uniform(-10, 10)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor, label


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Augmentation Pipelines  (Kornia)
# ═══════════════════════════════════════════════════════════════════════════════

class GPUCoarseDropout(nn.Module):
    def __init__(self, num_holes: int = 8, hole_size_ratio: float = 0.1, p: float = 0.2):
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
        tops  = torch.randint(0, max(1, H - hole_h), (B, self.num_holes), device=x.device)
        lefts = torch.randint(0, max(1, W - hole_w), (B, self.num_holes), device=x.device)
        rows = torch.arange(H, device=x.device).view(1, 1, H, 1)
        cols = torch.arange(W, device=x.device).view(1, 1, 1, W)
        for i in range(self.num_holes):
            t = tops[:, i].view(B, 1, 1, 1)
            l = lefts[:, i].view(B, 1, 1, 1)
            in_hole = (rows >= t) & (rows < t + hole_h) & (cols >= l) & (cols < l + hole_w)
            mask[in_hole.expand_as(mask)] = 0.0
        return x * mask

class SafeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std",  std.view(1, -1, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

def get_gpu_transforms(res: int, device: torch.device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)
    train_tf = nn.Sequential(
        K.RandomCrop((res, res)),
        K.RandomAffine(degrees=0, shear=[-10.0, 10.0, -10.0, 10.0], scale=[0.9, 1.1], p=0.5),
        K.RandomElasticTransform(kernel_size=(63, 63), sigma=(10.0, 10.0), alpha=(1.0, 1.0), p=0.3),
        SafeNormalize(mean=mean, std=std),
    ).to(device)
    val_tf = nn.Sequential(SafeNormalize(mean=mean, std=std)).to(device)
    return train_tf, val_tf


# ═══════════════════════════════════════════════════════════════════════════════
#  CUDA Prefetcher
# ═══════════════════════════════════════════════════════════════════════════════

class CUDAPrefetcher:
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_images: torch.Tensor | None = None
        self.next_labels: torch.Tensor | None = None
        self._iter = None
    def __len__(self): return len(self.loader)
    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self
    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        images, labels = self.next_images, self.next_labels
        if images is None: raise StopIteration
        images.record_stream(torch.cuda.current_stream(self.device))
        labels.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return images, labels
    def _preload(self):
        try: images, labels = next(self._iter)
        except StopIteration:
            self.next_images, self.next_labels = None, None
            return
        with torch.cuda.stream(self.stream):
            self.next_images = images.to(self.device, non_blocking=True).float().div_(255.0).contiguous()
            self.next_labels = labels.to(self.device, non_blocking=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricMonitor:
    def __init__(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, train_tf, coarse_dropout, model_ema, progress, task_id, no_live=False):
    model.train(); coarse_dropout.train()
    monitor = MetricMonitor(); correct = total = current_batch = 0
    num_batches = len(loader)
    for images, labels in loader:
        images = train_tf(images.contiguous())
        images = coarse_dropout(images)
        images = images.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model_ema.update(model)
        monitor.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item(); total += labels.size(0); current_batch += 1
        
        if no_live:
            if current_batch % 10 == 0 or current_batch == num_batches:
                print(f"    Batch {current_batch}/{num_batches} | Loss: {monitor.avg:.4f} | Acc: {correct/total:.4f}")
        else:
            progress.update(task_id, advance=1, description=f"  [yellow]Batch {current_batch}/{num_batches}[/yellow]")
            
    return monitor.avg, correct / total if total > 0 else 0

def validate(model, loader, criterion, device, use_amp, val_tf, tta_ways=8):
    model.eval()
    monitor = MetricMonitor(); all_targets = []; all_scores = []
    with torch.no_grad():
        for images, labels in loader:
            images = val_tf(images.contiguous()).to(memory_format=torch.channels_last)
            if tta_ways >= 8:
                variants = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3]),
                            torch.rot90(images, 1, dims=[2, 3]), torch.rot90(images, 2, dims=[2, 3]),
                            torch.rot90(images, 3, dims=[2, 3]), torch.flip(torch.rot90(images, 1, dims=[2, 3]), [3])]
            elif tta_ways >= 4:
                variants = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])]
            else: variants = [images]
            avg_probs = torch.zeros(images.size(0), 2, device=device); logits_orig = None
            with torch.amp.autocast("cuda", enabled=use_amp):
                for i, v in enumerate(variants):
                    logits = model(v)
                    if i == 0: logits_orig = logits
                    avg_probs += torch.softmax(logits.float(), dim=1)
            avg_probs /= len(variants); loss = criterion(logits_orig.float(), labels)
            monitor.update(loss.item(), images.size(0)); all_targets.extend(labels.cpu().numpy()); all_scores.extend(avg_probs[:, 1].cpu().numpy())
    all_targets = np.array(all_targets); all_scores = np.array(all_scores)
    preds = (all_scores > 0.5).astype(int); acc = (preds == all_targets).mean(); auc = roc_auc_score(all_targets, all_scores)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0; specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = sk_f1(all_targets, preds, zero_division=0)
    return monitor.avg, acc, auc, sensitivity, specificity, f1


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

TIMM_NAME_MAP = {"mnv3l": "mobilenetv3_large_100", "effb0": "efficientnet_b0", "effb4": "efficientnet_b4", "rn50": "resnet50"}

class ConstrainedHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=2, dropout=0.6):
        super().__init__()
        self.head = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(dropout), nn.Linear(in_features, hidden_dim),
                                   nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.head(x)

def get_model(args):
    timm_name = TIMM_NAME_MAP[args.model]
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
    backbone.eval()
    with torch.no_grad(): in_features = backbone(torch.zeros(2, 3, 224, 224)).shape[-1]
    backbone.train()
    model = nn.Sequential(backbone, ConstrainedHead(in_features, dropout=0.6))
    return model, list(model[0].parameters()), list(model[1].parameters()), timm_name, in_features


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def get_loaders(args):
    with open(args.splits_json, "r") as f: splits = json.load(f)
    fold_key = f"fold_{args.fold}"; fold_data = splits["folds"][fold_key]
    train_ds = CNMCDataset(fold_data["train_images"], target_size=args.res + 64, is_train=True)
    val_ds   = CNMCDataset(fold_data["val_images"],   target_size=args.res,      is_train=False)
    labels = [p[1] for p in fold_data["train_images"]]; class_counts = np.bincount(labels)
    sampler = WeightedRandomSampler(weights=np.array([1.0 / class_counts[l] for l in labels]), num_samples=len(labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    normed_weights = len(class_counts) / (class_counts * 2.0)
    info = {"train_total": len(fold_data["train_images"]), "val_total": len(fold_data["val_images"]),
            "train_all": sum(1 for _, l in fold_data["train_images"] if l == 0), "train_hem": sum(1 for _, l in fold_data["train_images"] if l == 1),
            "val_all": sum(1 for _, l in fold_data["val_images"] if l == 0), "val_hem": sum(1 for _, l in fold_data["val_images"] if l == 1)}
    return train_loader, val_loader, torch.tensor(normed_weights, dtype=torch.float32), class_counts, info


# ═══════════════════════════════════════════════════════════════════════════════
#  System Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def system_monitor(log_path, stop_event, ms, interval=5):
    with open(log_path, "w") as f: f.write("timestamp,cpu_pct,ram_used_gb,ram_total_gb,gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c\n")
    while not stop_event.is_set():
        try:
            cpu = psutil.cpu_percent(); ram = psutil.virtual_memory()
            res = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            p = [x.strip() for x in res.stdout.strip().split(",")]
            with open(log_path, "a") as f: f.write(f"{p[0]},{cpu},{ram.used/1024**3:.2f},{ram.total/1024**3:.2f},{p[1]},{p[2]},{p[3]},{p[4]}\n")
            ms["cpu_pct"]=float(cpu); ms["ram_used"]=ram.used/1024**3; ms["ram_total"]=ram.total/1024**3; ms["gpu_util"]=float(p[1]); ms["vram_used"]=float(p[2]); ms["vram_total"]=float(p[3]); ms["gpu_temp"]=float(p[4])
        except: pass
        stop_event.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════════

def format_time(s):
    s = int(s); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"

def build_display(run_id, fold_str, epoch, total_epochs, batch_progress, best, current, ms, elapsed, eta, bLR, hLR, pat, time_ep):
    table = Table(box=box.DOUBLE, expand=False, show_header=False, padding=(0,1), min_width=76)
    table.add_column(min_width=72)
    table.add_row("  [bold cyan]ALL Leukemia Edge Classifier (Colab)[/bold cyan]")
    table.add_row(f"  Run: {run_id}{' '*(68-len(run_id)-len(fold_str))}{fold_str}")
    table.add_row(f"  [bold]PROGRESS[/bold]   Epoch {epoch} / {total_epochs}")
    table.add_row(batch_progress)
    table.add_row(f"  Elapsed: {elapsed}  |  Time/Ep: {time_ep}{' '*(40-len(elapsed)-len(time_ep)-len(eta))}ETA: {eta}")
    table.add_section()
    if best["epoch"] > 0:
        table.add_row(f"  [bold green]BEST  (Epoch {best['epoch']})[/bold green]"); table.add_row(f"  AUC: [green]{best['auc']:.4f}[/green]  Acc: {best['acc']:.4f}  Sens: {best['sens']:.4f}  Spec: {best['spec']:.4f}")
    else: table.add_row("  [bold green]BEST[/bold green]\n  Waiting..."); table.add_section()
    if current["epoch"] > 0:
        table.add_row(f"  [bold yellow]CURRENT  (Epoch {current['epoch']})[/bold yellow]"); table.add_row(f"  AUC: [yellow]{current['auc']:.4f}[/yellow]  Acc: {current['acc']:.4f}  F1: {current['f1']:.4f}\n  bLR: {bLR:.6f}  hLR: {hLR:.6f}  Pat: {pat}")
    else: table.add_row("  [bold yellow]CURRENT[/bold yellow]\n  Training...")
    table.add_section(); table.add_row("  [bold magenta]HARDWARE[/bold magenta]")
    table.add_row(f"  CPU: {ms['cpu_pct']:>2.0f}%  RAM: {ms['ram_used']:>4.1f} / {ms['ram_total']:>4.1f} GB")
    table.add_row(f"  GPU: {ms['gpu_util']:>2.0f}%  VRAM: {ms['vram_used']/1024:>4.1f} / {ms['vram_total']/1024:>4.1f} GB  Temp: {ms['gpu_temp']:.0f}°C")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

WARMUP_EPOCHS = 5

def main():
    args = parse_args(); console = Console(force_terminal=True)
    run_id = f"{args.run_name}_fold{args.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.output_root, args.run_name); logger = setup_logging(out_dir, run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    train_loader, val_loader, loss_weights, class_counts, data_info = get_loaders(args)
    model, backbone_params, head_params, timm_name, in_feat = get_model(args)
    model = model.to(device).to(memory_format=torch.channels_last)
    train_tf, val_tf = get_gpu_transforms(args.res, device); coarse_dropout = GPUCoarseDropout().to(device)
    model_ema = ModelEmaV2(model, decay=0.9998, device=device)
    
    # --- IDENTITY BLOCK ---
    logger.info("=" * 70)
    logger.info(f"  Run ID:          {run_id}")
    logger.info(f"  Model:           {args.model} ({timm_name})")
    logger.info(f"  Fold:            {args.fold}")
    logger.info(f"  Resolution:      {args.res}x{args.res}")
    logger.info(f"  Batch Size:      {args.batch_size}")
    logger.info(f"  Device:          {device}")
    logger.info(f"  Workers:         {args.num_workers}")
    logger.info("=" * 70)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=loss_weights.to(device))
    optimizer = optim.AdamW([{"params": backbone_params, "lr": args.lr_backbone}, {"params": head_params, "lr": args.lr_head}], weight_decay=args.weight_decay)
    scheduler = SequentialLR(optimizer, schedulers=[LinearLR(optimizer, 0.1, total_iters=WARMUP_EPOCHS), CosineAnnealingLR(optimizer, args.epochs-WARMUP_EPOCHS)], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    
    ms = {"cpu_pct":0, "ram_used":0, "ram_total":0, "gpu_util":0, "vram_used":0, "vram_total":0, "gpu_temp":0}
    stop_event = threading.Event(); threading.Thread(target=system_monitor, args=(os.path.join(out_dir, f"{run_id}_system.log"), stop_event, ms), daemon=True).start()
    
    best_auc = 0.0; patience_counter = 0; best_state = {"epoch":0, "auc":0, "acc":0, "sens":0, "spec":0, "f1":0, "train_loss":0, "val_loss":0}; current_state = best_state.copy(); epoch_times = []
    ckpt_path = os.path.join(out_dir, f"{run_id}_best.pth"); metrics_path = os.path.join(out_dir, f"{run_id}_metrics.csv")
    with open(metrics_path, "w") as f: f.write("epoch,train_loss,train_acc,val_loss,val_acc,auc,sensitivity,specificity,f1,epoch_time\n")
    
    batch_progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=36), TextColumn("{task.percentage:>3.0f}%"))
    epoch_task = batch_progress.add_task(f"  [blue]Epochs ({args.epochs})[/blue]", total=args.epochs)
    batch_task = batch_progress.add_task(f"  [yellow]Batches ({len(train_loader)})[/yellow]", total=len(train_loader))
    
    global_start = time.time(); ui_state = {"epoch":0, "bLR":args.lr_backbone, "hLR":args.lr_head, "pat":0, "eta":"...", "time_ep":"..."}

    def run_epoch(epoch):
        nonlocal best_auc, patience_counter, best_state
        t0 = time.time(); ui_state["epoch"] = epoch
        if not args.no_live:
            batch_progress.reset(batch_task, description=f"  [yellow]Batch 0/{len(train_loader)}[/yellow]")
        else:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            
        train_l, train_a = train_one_epoch(model, CUDAPrefetcher(train_loader, device), criterion, optimizer, scaler, device, True, train_tf, coarse_dropout, model_ema, batch_progress, batch_task, no_live=args.no_live)
        tta = 8 if epoch == args.epochs else 4 if epoch % 5 == 0 else 1
        val_l, acc, auc, sens, spec, f1 = validate(model_ema.module, CUDAPrefetcher(val_loader, device), criterion, device, True, val_tf, tta)
        scheduler.step(); ep_t = int(time.time()-t0); epoch_times.append(ep_t); ui_state["time_ep"] = f"{ep_t}s"
        avg_t = sum(epoch_times[-5:]) / len(epoch_times[-5:]); ui_state["eta"] = format_time(avg_t*(args.epochs-epoch))
        cur = {"epoch":epoch, "auc":auc, "acc":acc, "sens":sens, "spec":spec, "f1":f1, "train_loss":train_l, "val_loss":val_l}
        if auc > best_auc:
            best_auc = auc; best_state = cur.copy(); patience_counter = 0
            torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "ema_state_dict":model_ema.state_dict(), "auc":auc}, ckpt_path)
        else: patience_counter += 1
        ui_state["bLR"]=optimizer.param_groups[0]["lr"]; ui_state["hLR"]=optimizer.param_groups[1]["lr"]; ui_state["pat"]=patience_counter
        line = f"E {epoch:>3}/{args.epochs} | TrL={train_l:.4f} TrA={train_a:.4f} | VlA={acc:.4f} AUC={auc:.4f} | {ep_t}s | pat={patience_counter}/{args.patience}{' *BEST' if cur['auc']==best_auc else ''}"
        with open(metrics_path, "a") as f: f.write(f"{epoch},{train_l:.4f},{train_a:.4f},{val_l:.4f},{acc:.4f},{auc:.4f},{sens:.4f},{spec:.4f},{f1:.4f},{ep_t}\n")
        return line

    if args.no_live:
        for e in range(1, args.epochs + 1):
            line = run_epoch(e); print(line); logger.info(line)
            if patience_counter >= args.patience and e >= 80: break
    else:
        with Live(Dashboard(), refresh_per_second=4, console=console) as live:
            for e in range(1, args.epochs + 1):
                line = run_epoch(e); live.console.print(line); logger.info(line)
                if patience_counter >= args.patience and e >= 80: break
    
    stop_event.set(); logger.info("Training Complete")

if __name__ == "__main__":
    main()
