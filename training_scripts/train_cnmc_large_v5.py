"""
Train MobileNetV3-Large v5 — Generalization & Stabilization Blueprint
======================================================================
Key changes from v4 (based on Gemini collaboration):
  1. Constrained head: 128 dims (down from 512) with 0.6 dropout
  2. Standard CrossEntropy loss (no Focal — avoids double-compensating with sampler)
  3. Differential learning rates: 1e-5 backbone, 1e-3 head
  4. Albumentations: biologically-plausible augmentations (Affine, mild Elastic)
     - NO CutMix/MixUp (destroys N/C ratio critical for ALL diagnosis)
     - NO ColorJitter (conflicts with Macenko stain normalization)
  5. Tightened early stopping patience (10 epochs)
  6. Resume from v4 best checkpoint

Pipeline: PyTorch train → ONNX export → TFLite convert → Pi 5 deploy.

Usage:
  python training_scripts/train_cnmc_large_v5.py
  python training_scripts/train_cnmc_large_v5.py --run_name v5_test
  python training_scripts/train_cnmc_large_v5.py --no_resume  # train from scratch
"""
import os
import sys
import time
import copy
import argparse
import logging
import json
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import models
import numpy as np
import cv2  # OpenCV — faster than PIL for image loading
from PIL import Image  # Fallback only
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix, roc_auc_score,
                             roc_curve)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    from torchvision import transforms

# ============ FORCE GPU ============
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Ensure async CUDA calls
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

# ============ CONFIG ============
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
STAGING_DIR_NORMED = r"c:\Open Source\leukiemea\cnmc_staging_normed"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
LOG_DIR = r"c:\Open Source\leukiemea\logs\training"
V4_CHECKPOINT = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v4.pth")
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
    logger.info("  TRAINING CONFIGURATION (v5)")
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
    logger.info(f"  {'albumentations':<25s} = {HAS_ALBUMENTATIONS}")
    logger.info(f"  {'started_at':<25s} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    json_path = os.path.join(LOG_DIR, f"{args.run_name}_params.json")
    params_out = {**params, 'device': str(DEVICE), 'pytorch_version': torch.__version__,
                  'albumentations': HAS_ALBUMENTATIONS,
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
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large v5")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=320,
                        help="Input resolution (default: 320)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (32 with AMP fits 6GB VRAM)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr_head", type=float, default=1e-3,
                        help="Learning rate for classification head")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Learning rate for backbone (pretrained layers)")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Hidden dimension of classification head (128 or 64)")
    parser.add_argument("--head_dropout", type=float, default=0.6,
                        help="Dropout rate in classification head")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (tightened for v5)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (2 for Windows spawn safety)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Train from scratch (don't load any weights)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_stain_norm", action="store_true")
    parser.add_argument("--stain_norm", action="store_true", default=True)
    args = parser.parse_args()

    if args.run_name is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f"v5_{args.resolution}px_{ts}"

    return args


# ============ CONSTRAINED HEAD (v5: 128 dims, 0.6 dropout) ============
class ConstrainedHead(nn.Module):
    """Smaller head to prevent memorization / overfitting.
    v4 used 512 dims + 0.3-0.4 dropout = overfitting (98.7% train, 88% val).
    v5 uses 128 dims + 0.6 dropout = forces generalization."""
    def __init__(self, in_features=960, hidden_dim=128, num_classes=2, dropout=0.6):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# ============ MODEL ============
def build_model(resume_path=None, head_dim=128, head_dropout=0.6):
    """Build MobileNetV3-Large with constrained head."""
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    except AttributeError:
        try:
            weights = models.MobileNetV3_Large_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = 'IMAGENET1K_V2'

    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = ConstrainedHead(
        in_features=in_features,
        hidden_dim=head_dim,
        num_classes=2,
        dropout=head_dropout
    )

    if resume_path and os.path.exists(resume_path):
        logger = logging.getLogger("training")
        logger.info(f"  Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        # Filter out classifier weights (head size changed from 512 to 128)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # Only load weights that match in both name and shape
        filtered_dict = {}
        skipped = []
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                skipped.append(k)
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info(f"  Loaded {len(filtered_dict)}/{len(pretrained_dict)} weight tensors")
        if skipped:
            logger.info(f"  Skipped {len(skipped)} tensors (head re-initialized):")
            for s in skipped:
                logger.info(f"    - {s}")
        else:
            logger.info(f"  Loaded all weights (resolution={checkpoint.get('resolution', 'unknown')})")
    elif resume_path:
        logger = logging.getLogger("training")
        logger.warning(f"  Checkpoint not found: {resume_path}, training from scratch")

    return model.to(DEVICE)


# ============ ALBUMENTATIONS DATA PIPELINE ============
class AlbumentationsDataset(Dataset):
    """ImageFolder-compatible dataset using Albumentations transforms."""
    def __init__(self, root_dir, transform=None):
        from torchvision.datasets import ImageFolder
        self._dataset = ImageFolder(root_dir)
        self.samples = self._dataset.samples
        self.targets = self._dataset.targets
        self.classes = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # OpenCV is significantly faster than PIL and is what Albumentations expects
        image = cv2.imread(path)
        if image is None:
            # Fallback to PIL if OpenCV fails
            image = np.array(Image.open(path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_dataloaders(train_dir, val_dir, args):
    logger = logging.getLogger("training")
    res = args.resolution

    if HAS_ALBUMENTATIONS:
        logger.info("  Using Albumentations (biological augmentations)")
        train_transforms = A.Compose([
            A.Resize(res + 32, res + 32),
            A.RandomCrop(res, res),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Safe biological deformations (simulates slide preparation)
            A.Affine(shear=(-10, 10), scale=(0.9, 1.1), p=0.5),
            A.ElasticTransform(alpha=1, sigma=10, p=0.3, approximate=True),  # GPU-friendly
            # NO ColorJitter — conflicts with Macenko stain normalization
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=int(res*0.08),
                           max_width=int(res*0.08), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        val_transforms = A.Compose([
            A.Resize(res, res),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        train_dataset = AlbumentationsDataset(train_dir, transform=train_transforms)
        val_dataset = AlbumentationsDataset(val_dir, transform=val_transforms)
    else:
        logger.warning("  Albumentations not found, falling back to torchvision transforms")
        from torchvision import transforms as T
        imagenet_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = T.Compose([
            T.Resize((res + 32, res + 32)),
            T.RandomCrop(res),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            imagenet_normalize,
        ])
        val_transforms = T.Compose([T.Resize((res, res)), T.ToTensor(), imagenet_normalize])
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder(train_dir, transform=train_transforms)
        val_dataset = ImageFolder(val_dir, transform=val_transforms)

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
def validate_with_tta(model, val_loader, criterion):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    count = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            batch_size = inputs.size(0)

            # 4 TTA variants — capture first pass logits for loss
            variants = [
                inputs,
                torch.flip(inputs, [3]),
                torch.flip(inputs, [2]),
                torch.flip(inputs, [2, 3]),
            ]

            logits_orig = None
            avg_probs = torch.zeros(batch_size, 2).to(DEVICE)
            for i, v in enumerate(variants):
                logits = model(v)
                if i == 0:
                    logits_orig = logits  # capture first pass for loss
                avg_probs += F.softmax(logits, dim=1)
            avg_probs /= len(variants)

            loss = criterion(logits_orig, labels)  # no extra forward pass
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


# ============ TRAINING (simplified — no CutMix/MixUp) ============
def train_one_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Free gradient memory

        # AMP: mixed precision forward + backward
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
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

    is_all = (labels == 0).astype(int)
    p_all = probs[:, 0]

    fpr, tpr, thresholds = roc_curve(is_all, p_all)
    auc = roc_auc_score(is_all, p_all)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    optimal_preds = (p_all >= best_threshold).astype(int)
    optimal_labels_pred = np.where(optimal_preds == 1, 0, 1)

    acc_opt = accuracy_score(labels, optimal_labels_pred)
    sens_opt = recall_score(labels, optimal_labels_pred, pos_label=0)
    spec_opt = recall_score(labels, optimal_labels_pred, pos_label=1)
    f1_opt = f1_score(labels, optimal_labels_pred, average='weighted')

    default_preds = np.argmax(probs, axis=1)
    acc_def = accuracy_score(labels, default_preds)

    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Default threshold (0.5):  Acc={acc_def:.4f}")
    logger.info(f"  Optimal threshold ({best_threshold:.4f}): Acc={acc_opt:.4f} "
                f"Sens={sens_opt:.4f} Spec={spec_opt:.4f} F1={f1_opt:.4f}")

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
def export_model(model, class_to_idx, args, suffix="v5"):
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

    # Build model (with optional warm start from v4)
    if args.no_resume:
        resume_path = None
    elif args.resume_from:
        resume_path = args.resume_from
    else:
        # Default: try v4 first, then v3
        resume_path = V4_CHECKPOINT if os.path.exists(V4_CHECKPOINT) else V3_CHECKPOINT
    model = build_model(resume_path=resume_path, head_dim=args.head_dim,
                        head_dropout=args.head_dropout)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    # v5: Standard CrossEntropy (no Focal — sampler handles class balance)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    logger.info(f"  Loss: CrossEntropyLoss(label_smoothing={args.label_smoothing})")
    logger.info(f"  Head: ConstrainedHead(dim={args.head_dim}, dropout={args.head_dropout})")

    # Differential learning rates: backbone gets 1e-5, head gets 1e-3
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=1e-4)

    logger.info(f"  Optimizer: AdamW (backbone_lr={args.lr_backbone}, head_lr={args.lr_head})")
    logger.info(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
    logger.info(f"  Head params: {sum(p.numel() for p in head_params):,}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable: {trainable:,}/{total_params:,} (100%)")

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # AMP: Mixed precision training for faster GPU utilization
    scaler = torch.amp.GradScaler('cuda')
    logger.info(f"  AMP: Mixed precision enabled (GradScaler)")

    eta = ETATracker(args.epochs)
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_metrics = None
    patience_counter = 0

    logger.info("")
    logger.info("  Starting v5 training (constrained head, differential LR, no CutMix/MixUp)...")
    logger.info("─" * 120)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = validate_with_tta(model, val_loader, criterion)

        scheduler.step()

        elapsed = time.time() - t0
        eta.tick(elapsed)
        # Show both LRs
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
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
            'lr_backbone': f"{backbone_lr:.8f}",
            'lr_head': f"{head_lr:.8f}",
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
            f"bLR={backbone_lr:.6f} hLR={head_lr:.4f} │ "
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
    logger.info("  FINAL RESULTS (v5)")
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
    export_model(model, class_to_idx, args, suffix="v5")


if __name__ == "__main__":
    main()
