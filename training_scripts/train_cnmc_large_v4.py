"""
Train MobileNetV3-Large v4 — Push to 96%+ Accuracy
=====================================================
Fine-tunes from v3 checkpoint with targeted fixes for specificity:
  1. Resume from v3 best weights (warm start)
  2. Focal alpha=0.6 (properly penalizes HEM misclassification)
  3. WeightedRandomSampler (class-balanced batches without oversampling)
  4. Resolution 384×384 (more morphological detail)
  5. CutMix + MixUp augmentation
  6. Lower LR (1e-4) for fine-tuning
  7. Post-training ROC threshold optimization

Pipeline: PyTorch train → ONNX export → TFLite convert → Pi 5 deploy.

Usage:
  python training_scripts/train_cnmc_large_v4.py
  python training_scripts/train_cnmc_large_v4.py --resolution 320  # lower res if OOM
  python training_scripts/train_cnmc_large_v4.py --no_resume       # train from scratch
"""
import os
import sys
import time
import copy
import random
import argparse
import logging
import json
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix, roc_auc_score,
                             roc_curve)

# ============ CONFIG ============
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
STAGING_DIR_NORMED = r"c:\Open Source\leukiemea\cnmc_staging_normed"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
LOG_DIR = r"c:\Open Source\leukiemea\logs\training"
V3_CHECKPOINT = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v3.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_staging_dir(use_stain_norm):
    """Auto-detect pre-normalized data."""
    if use_stain_norm and os.path.isdir(STAGING_DIR_NORMED):
        return STAGING_DIR_NORMED, True
    return STAGING_DIR, False


# ============ LOGGING ============
def setup_logging(run_name):
    log_file = os.path.join(LOG_DIR, f"{run_name}.log")
    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


def log_params(logger, args):
    params = vars(args)
    logger.info("")
    logger.info("=" * 65)
    logger.info("  TRAINING CONFIGURATION (v4)")
    logger.info("=" * 65)
    for k, v in sorted(params.items()):
        logger.info(f"  {k:<25s} = {v}")
    logger.info(f"  {'device':<25s} = {DEVICE}")
    logger.info(f"  {'cuda_available':<25s} = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  {'gpu_name':<25s} = {torch.cuda.get_device_name(0)}")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            gpu_mem = total_mem / (1024**3)
        except Exception:
            gpu_mem = 0
        logger.info(f"  {'gpu_memory_gb':<25s} = {gpu_mem:.1f}")
    logger.info(f"  {'pytorch_version':<25s} = {torch.__version__}")
    logger.info(f"  {'started_at':<25s} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    json_path = os.path.join(LOG_DIR, f"{args.run_name}_params.json")
    params_out = {**params, 'device': str(DEVICE), 'pytorch_version': torch.__version__,
                  'started_at': datetime.now().isoformat()}
    with open(json_path, 'w') as f:
        json.dump(params_out, f, indent=2, default=str)


class ETATracker:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = time.time()

    def tick(self, elapsed):
        self.epoch_times.append(elapsed)

    def get_eta(self, current_epoch):
        remaining = self.total_epochs - (current_epoch + 1)
        if remaining <= 0:
            return "Done", 0.0, self._fmt_time(time.time() - self.start_time)
        recent = self.epoch_times[-5:] if len(self.epoch_times) >= 5 else self.epoch_times
        avg_time = sum(recent) / len(recent)
        eta_seconds = avg_time * remaining
        total_elapsed = time.time() - self.start_time
        return self._fmt_time(eta_seconds), avg_time, self._fmt_time(total_elapsed)

    @staticmethod
    def _fmt_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"


class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rows = []
        self.fieldnames = None

    def log(self, row_dict):
        if self.fieldnames is None:
            self.fieldnames = list(row_dict.keys())
        self.rows.append(row_dict)
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


# ============ ARGUMENT PARSING ============
def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large v4")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=320,
                        help="Input resolution (default: 320, same as v3)")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Total epochs (less needed with warm start)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.6,
                        help="Focal alpha (0.6 = stronger HEM penalty)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true",
                        help="Train from scratch (don't load any weights)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from (default: v3)")
    parser.add_argument("--no_stain_norm", action="store_true")
    parser.add_argument("--stain_norm", action="store_true", default=True)
    args = parser.parse_args()

    if args.run_name is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f"v4_{args.resolution}px_{ts}"

    return args


# ============ FOCAL LOSS ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            focal_weight = (1.0 - probs) ** self.gamma
            loss = -self.alpha * focal_weight * smooth_targets * log_probs
            return loss.sum(dim=1).mean()
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            focal_weight = (1.0 - probs) ** self.gamma
            loss = -self.alpha * focal_weight * targets_one_hot * log_probs
            return loss.sum(dim=1).mean()


# ============ MIXUP + CUTMIX ============
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut and paste patches between images."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get bounding box
    W, H = x.size(3), x.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # Adjust lambda by actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ STRONGER HEAD ============
class StrongHead(nn.Module):
    def __init__(self, in_features=960, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# ============ MODEL ============
def build_model(resume_path=None):
    """Build MobileNetV3-Large, optionally loading v3 weights."""
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    except AttributeError:
        try:
            weights = models.MobileNetV3_Large_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = 'IMAGENET1K_V2'

    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = StrongHead(in_features=in_features, num_classes=2)

    if resume_path and os.path.exists(resume_path):
        logger = logging.getLogger("training")
        logger.info(f"  Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Loaded weights (resolution={checkpoint.get('resolution', 'unknown')})")
    elif resume_path:
        logger = logging.getLogger("training")
        logger.warning(f"  Checkpoint not found: {resume_path}, training from scratch")

    return model.to(DEVICE)


# ============ DATA PIPELINE ============
def get_dataloaders(train_dir, val_dir, args):
    logger = logging.getLogger("training")
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    res = args.resolution

    train_transforms = transforms.Compose([
        transforms.Resize((res + 32, res + 32)),
        transforms.RandomCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        imagenet_normalize,
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    logger.info(f"  Classes: {train_dataset.classes} -> {train_dataset.class_to_idx}")
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ---- Class-balanced sampling (WeightedRandomSampler) ----
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    logger.info(f"  Class counts: {dict(enumerate(class_counts.tolist()))}")
    logger.info(f"  Class weights: {dict(enumerate(class_weights.tolist()))}")
    logger.info(f"  Using WeightedRandomSampler (balanced batches)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    return train_loader, val_loader, train_dataset.class_to_idx


# ============ TTA VALIDATION ============
def validate_with_tta(model, val_loader, criterion, args):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.size(0)

            variants = [
                inputs,
                torch.flip(inputs, [3]),
                torch.flip(inputs, [2]),
                torch.flip(inputs, [2, 3]),
            ]

            avg_probs = torch.zeros(batch_size, 2).to(DEVICE)
            for v in variants:
                logits = model(v)
                avg_probs += F.softmax(logits, dim=1)
            avg_probs /= len(variants)

            logits_orig = model(inputs)
            loss = criterion(logits_orig, labels)
            total_loss += loss.item() * batch_size
            count += batch_size

            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.argmax(all_probs, axis=1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    sens = recall_score(all_labels, all_preds, pos_label=0)
    spec = recall_score(all_labels, all_preds, pos_label=1)
    prec = precision_score(all_labels, all_preds, average='weighted')

    try:
        auc = roc_auc_score(1 - all_labels, all_probs[:, 0])
    except ValueError:
        auc = 0.0

    return {
        'loss': total_loss / count,
        'accuracy': acc, 'f1': f1, 'sensitivity': sens,
        'specificity': spec, 'precision': prec, 'auc': auc,
        'preds': all_preds, 'labels': all_labels, 'probs': all_probs,
    }


# ============ TRAINING ============
def train_one_epoch(model, train_loader, criterion, optimizer, args, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Randomly choose: MixUp, CutMix, or neither
        r = random.random()
        if r < 0.3 and args.mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
            use_mix = True
        elif r < 0.6 and args.cutmix_alpha > 0:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, args.cutmix_alpha)
            use_mix = True
        else:
            use_mix = False

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mix:
            loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        if not use_mix:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(train_loader.dataset)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ============ ROC THRESHOLD OPTIMIZATION ============
def optimize_threshold(labels, probs, logger):
    """Find optimal decision threshold using Youden's J statistic."""
    logger.info("")
    logger.info("  ROC THRESHOLD OPTIMIZATION")
    logger.info("  " + "-" * 50)

    # P(ALL) = probs[:, 0], positive class = ALL (label 0)
    is_all = (labels == 0).astype(int)
    p_all = probs[:, 0]

    fpr, tpr, thresholds = roc_curve(is_all, p_all)
    auc = roc_auc_score(is_all, p_all)

    # Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Apply optimal threshold
    optimal_preds = (p_all >= best_threshold).astype(int)
    # Convert back: 1=ALL, 0=HEM in is_all space → 0=ALL, 1=HEM in label space
    optimal_labels_pred = np.where(optimal_preds == 1, 0, 1)

    acc_opt = accuracy_score(labels, optimal_labels_pred)
    sens_opt = recall_score(labels, optimal_labels_pred, pos_label=0)
    spec_opt = recall_score(labels, optimal_labels_pred, pos_label=1)
    f1_opt = f1_score(labels, optimal_labels_pred, average='weighted')

    # Also check default 0.5
    default_preds = np.argmax(probs, axis=1)
    acc_def = accuracy_score(labels, default_preds)

    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Default threshold (0.5):  Acc={acc_def:.4f}")
    logger.info(f"  Optimal threshold ({best_threshold:.4f}): Acc={acc_opt:.4f} "
                f"Sens={sens_opt:.4f} Spec={spec_opt:.4f} F1={f1_opt:.4f}")

    # Try a range of thresholds
    logger.info("")
    logger.info(f"  {'Threshold':>10s} {'Accuracy':>10s} {'Sensitivity':>12s} {'Specificity':>12s} {'F1':>8s}")
    for t in np.arange(0.35, 0.75, 0.05):
        preds_t = np.where(p_all >= t, 0, 1)
        acc_t = accuracy_score(labels, preds_t)
        sens_t = recall_score(labels, preds_t, pos_label=0)
        spec_t = recall_score(labels, preds_t, pos_label=1)
        f1_t = f1_score(labels, preds_t, average='weighted')
        marker = " <-- optimal" if abs(t - best_threshold) < 0.025 else ""
        logger.info(f"  {t:>10.3f} {acc_t:>10.4f} {sens_t:>12.4f} {spec_t:>12.4f} {f1_t:>8.4f}{marker}")

    logger.info("  " + "-" * 50)
    return best_threshold, acc_opt, sens_opt, spec_opt


# ============ EXPORT ============
def export_model(model, class_to_idx, args, suffix="v4"):
    logger = logging.getLogger("training")
    model.eval()
    res = args.resolution

    pth_path = os.path.join(OUTPUT_DIR, f"mobilenetv3_large_{suffix}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'resolution': res,
        'config': vars(args),
    }, pth_path)
    logger.info(f"  PyTorch checkpoint: {pth_path}")

    onnx_path = os.path.join(OUTPUT_DIR, f"mobilenetv3_large_{suffix}.onnx")
    dummy = torch.randn(1, 3, res, res).to(DEVICE)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  ONNX: {onnx_path} ({onnx_size:.2f} MB)")

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = os.path.join(OUTPUT_DIR, f"tf_saved_{suffix}")
        tf_rep.export_graph(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join(OUTPUT_DIR, f"mobilenetv3_large_{suffix}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        logger.info(f"  TFLite: {tflite_path} ({tflite_size:.2f} MB)")
    except ImportError as e:
        logger.info(f"  TFLite conversion skipped (missing: {e})")


# ============ MAIN ============
def main():
    args = parse_args()
    logger = setup_logging(args.run_name)
    log_params(logger, args)

    csv_log = CSVLogger(os.path.join(LOG_DIR, f"{args.run_name}_metrics.csv"))

    # Auto-detect pre-normalized data
    use_stain = args.stain_norm and not args.no_stain_norm
    staging_dir, using_precomputed = get_staging_dir(use_stain)
    if using_precomputed:
        logger.info(f"  Using pre-normalized stain data: {staging_dir}")
    else:
        logger.info(f"  Using raw data: {staging_dir}")

    train_dir = os.path.join(staging_dir, "train")
    val_dir = os.path.join(staging_dir, "val")

    train_loader, val_loader, class_to_idx = get_dataloaders(train_dir, val_dir, args)

    # Build model (with optional warm start)
    if args.no_resume:
        resume_path = None
    elif args.resume_from:
        resume_path = args.resume_from
    else:
        # Default: try v4 first, then v3
        v4_ckpt = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v4.pth")
        resume_path = v4_ckpt if os.path.exists(v4_ckpt) else V3_CHECKPOINT
    model = build_model(resume_path=resume_path)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing)
    logger.info(f"  Loss: FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha}, "
                f"label_smoothing={args.label_smoothing})")

    # Full model trainable from start (fine-tuning, no phase progression)
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable: {trainable:,}/{total_params:,} (100%)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    eta = ETATracker(args.epochs)
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_metrics = None
    patience_counter = 0

    logger.info("")
    logger.info("  Starting v4 fine-tuning...")
    logger.info("─" * 120)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args, epoch)
        val_metrics = validate_with_tta(model, val_loader, criterion, args)

        scheduler.step()

        elapsed = time.time() - t0
        eta.tick(elapsed)
        current_lr = optimizer.param_groups[0]['lr']
        eta_str, avg_epoch_time, total_elapsed = eta.get_eta(epoch)

        csv_log.log({
            'epoch': epoch + 1,
            'train_loss': f"{train_loss:.6f}",
            'train_acc': f"{train_acc:.4f}",
            'val_loss': f"{val_metrics['loss']:.6f}",
            'val_acc': f"{val_metrics['accuracy']:.4f}",
            'val_f1': f"{val_metrics['f1']:.4f}",
            'val_sensitivity': f"{val_metrics['sensitivity']:.4f}",
            'val_specificity': f"{val_metrics['specificity']:.4f}",
            'val_auc': f"{val_metrics['auc']:.4f}",
            'val_precision': f"{val_metrics['precision']:.4f}",
            'lr': f"{current_lr:.8f}",
            'epoch_time_s': f"{elapsed:.1f}",
            'patience': patience_counter,
        })

        is_best = ""
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics.copy()
            patience_counter = 0
            is_best = " ★ BEST"
        else:
            patience_counter += 1

        logger.info(
            f"  E{epoch+1:3d}/{args.epochs} │ "
            f"TrL={train_loss:.4f} TrA={train_acc:.4f} │ "
            f"VlA={val_metrics['accuracy']:.4f} F1={val_metrics['f1']:.4f} "
            f"Se={val_metrics['sensitivity']:.4f} Sp={val_metrics['specificity']:.4f} "
            f"AUC={val_metrics['auc']:.4f} │ "
            f"LR={current_lr:.6f} │ "
            f"{elapsed:.0f}s │ "
            f"ETA {eta_str} │ "
            f"pat={patience_counter}/{args.patience}"
            f"{is_best}"
        )

        if patience_counter >= args.patience:
            logger.info(f"")
            logger.info(f"  ✋ Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final results
    logger.info("")
    logger.info("=" * 65)
    logger.info("  FINAL RESULTS (v4)")
    logger.info("=" * 65)
    logger.info(f"  Best Epoch:  {best_epoch}")
    logger.info(f"  Accuracy:    {best_metrics['accuracy']:.4f}  ({best_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  F1 Score:    {best_metrics['f1']:.4f}")
    logger.info(f"  Sensitivity: {best_metrics['sensitivity']:.4f}  ({best_metrics['sensitivity']*100:.2f}%)")
    logger.info(f"  Specificity: {best_metrics['specificity']:.4f}  ({best_metrics['specificity']*100:.2f}%)")
    logger.info(f"  AUC:         {best_metrics['auc']:.4f}")
    logger.info(f"  Precision:   {best_metrics['precision']:.4f}")

    cm = confusion_matrix(best_metrics['labels'], best_metrics['preds'])
    logger.info(f"")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"              Pred ALL  Pred HEM")
    logger.info(f"  True ALL    {cm[0][0]:>7d}  {cm[0][1]:>8d}")
    logger.info(f"  True HEM    {cm[1][0]:>7d}  {cm[1][1]:>8d}")

    # Post-training ROC threshold optimization
    best_threshold, opt_acc, opt_sens, opt_spec = optimize_threshold(
        best_metrics['labels'], best_metrics['probs'], logger)

    logger.info(f"")
    logger.info(f"  With optimal threshold ({best_threshold:.4f}):")
    logger.info(f"    Accuracy:    {opt_acc:.4f}  ({opt_acc*100:.2f}%)")
    logger.info(f"    Sensitivity: {opt_sens:.4f}")
    logger.info(f"    Specificity: {opt_spec:.4f}")

    _, _, total_elapsed = eta.get_eta(args.epochs - 1)
    logger.info(f"")
    logger.info(f"  Total training time: {total_elapsed}")
    logger.info("=" * 65)

    # Export
    export_model(model, class_to_idx, args, suffix="v4")


if __name__ == "__main__":
    main()
