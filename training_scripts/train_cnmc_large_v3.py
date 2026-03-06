"""
Train MobileNetV3-Large v3 — Push to 96%+ Accuracy
=====================================================
Key improvements over v2:
  1. Focal Loss (gamma=2, alpha=0.25) — replaces aggressive class weighting
  2. Higher resolution: 320×320 (captures fine morphology)
  3. Stronger classification head: BN → Dense(512) → ReLU → Dropout → Dense(2)
  4. 3-phase progressive fine-tuning (100 epochs total)
  5. Macenko stain normalization preprocessing
  6. MixUp augmentation (alpha=0.2)
  7. Cosine decay with 5-epoch linear warmup
  8. Label smoothing = 0.05
  9. TTA at validation (4 orientations)
  10. Optional 5-fold cross-validation

Pipeline: PyTorch train → ONNX export → TFLite convert → Pi 5 deploy.

Usage:
  python training_scripts/train_cnmc_large_v3.py
  python training_scripts/train_cnmc_large_v3.py --folds 5   # 5-fold CV
  python training_scripts/train_cnmc_large_v3.py --resolution 384  # Even higher res
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
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix, roc_auc_score)

# ============ CONFIG ============
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
STAGING_DIR_NORMED = r"c:\Open Source\leukiemea\cnmc_staging_normed"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
LOG_DIR = r"c:\Open Source\leukiemea\logs\training"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_staging_dir(use_stain_norm):
    """Auto-detect pre-normalized data. Use it if available and stain norm is wanted."""
    if use_stain_norm and os.path.isdir(STAGING_DIR_NORMED):
        return STAGING_DIR_NORMED, True
    return STAGING_DIR, False


# ============ LOGGING SETUP ============
def setup_logging(run_name):
    """Set up dual logging: file + console with timestamps."""
    log_file = os.path.join(LOG_DIR, f"{run_name}.log")

    # Create logger
    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — full debug output
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


def log_params(logger, args):
    """Log all training parameters at startup."""
    params = vars(args)
    logger.info("")
    logger.info("=" * 65)
    logger.info("  TRAINING CONFIGURATION")
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
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else 0
        logger.info(f"  {'gpu_memory_gb':<25s} = {gpu_mem:.1f}")
    logger.info(f"  {'pytorch_version':<25s} = {torch.__version__}")
    logger.info(f"  {'started_at':<25s} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # Also save as JSON
    json_path = os.path.join(LOG_DIR, f"{args.run_name}_params.json")
    params_out = {**params, 'device': str(DEVICE), 'pytorch_version': torch.__version__,
                  'started_at': datetime.now().isoformat()}
    with open(json_path, 'w') as f:
        json.dump(params_out, f, indent=2, default=str)


# ============ ETA TRACKER ============
class ETATracker:
    """Track elapsed time per epoch and estimate remaining time."""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = time.time()

    def tick(self, elapsed):
        """Record one epoch's elapsed time."""
        self.epoch_times.append(elapsed)

    def get_eta(self, current_epoch):
        """Return ETA string, avg epoch time, and total elapsed."""
        remaining = self.total_epochs - (current_epoch + 1)
        if remaining <= 0:
            return "Done", 0.0, self._fmt_time(time.time() - self.start_time)

        # Use rolling average of last 5 epochs for better estimate
        recent = self.epoch_times[-5:] if len(self.epoch_times) >= 5 else self.epoch_times
        avg_time = sum(recent) / len(recent)
        eta_seconds = avg_time * remaining

        total_elapsed = time.time() - self.start_time
        return self._fmt_time(eta_seconds), avg_time, self._fmt_time(total_elapsed)

    @staticmethod
    def _fmt_time(seconds):
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"


# ============ CSV LOGGER ============
class CSVLogger:
    """Log metrics per epoch to a CSV file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.rows = []
        self.fieldnames = None

    def log(self, row_dict):
        """Append one row of metrics."""
        if self.fieldnames is None:
            self.fieldnames = list(row_dict.keys())
        self.rows.append(row_dict)
        # Write incrementally
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

# ============ ARGUMENT PARSING ============
def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large v3")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for logs (auto-generated if empty)")
    parser.add_argument("--resolution", type=int, default=320,
                        help="Input resolution (default: 320, try 384)")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size (reduce if OOM at higher resolution)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="MixUp alpha (0 to disable)")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing factor")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--folds", type=int, default=0,
                        help="Number of CV folds (0 = single train/val split)")
    parser.add_argument("--stain_norm", action="store_true", default=True,
                        help="Apply Macenko stain normalization")
    parser.add_argument("--no_stain_norm", action="store_true",
                        help="Disable stain normalization")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Auto-generate run name
    if args.run_name is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f"v3_{args.resolution}px_{ts}"

    return args


# ============ FOCAL LOSS ============
class FocalLoss(nn.Module):
    """
    Focal Loss: downweights easy samples, focuses on hard examples.
    Much better than inverse-frequency weighting for imbalanced data.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        if self.label_smoothing > 0:
            # Focal with smooth targets
            focal_weight = (1.0 - probs) ** self.gamma
            loss = -self.alpha * focal_weight * smooth_targets * log_probs
            return loss.sum(dim=1).mean()
        else:
            # Standard focal loss
            targets_one_hot = F.one_hot(targets, num_classes).float()
            focal_weight = (1.0 - probs) ** self.gamma
            loss = -self.alpha * focal_weight * targets_one_hot * log_probs
            return loss.sum(dim=1).mean()


# ============ MIXUP ============
def mixup_data(x, y, alpha=0.2):
    """MixUp: blend pairs of images and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp-blended samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ MACENKO STAIN NORMALIZATION ============
class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.
    Normalizes color distribution to match a reference image.
    """
    def __init__(self):
        # Standard H&E reference stain vectors + concentrations
        self.HERef = np.array([[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
    
    def _get_stain_matrix(self, I, beta=0.15, alpha=1):
        """Extract stain matrix from an image using SVD."""
        I = I.reshape(-1, 3).astype(np.float64)
        
        # Convert to optical density
        OD = -np.log10(np.clip(I / 255.0, 1e-6, 1.0))
        
        # Remove background (low OD)
        mask = np.all(OD > beta, axis=1)
        if mask.sum() < 10:
            return self.HERef, self.maxCRef
        
        OD_hat = OD[mask]
        
        # SVD
        try:
            _, _, V = np.linalg.svd(OD_hat, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.HERef, self.maxCRef
        
        # Plane from first two principal components
        V = V[:2, :]
        
        # Project and find angles
        proj = OD_hat @ V.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        v1 = V.T @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V.T @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        
        # Ensure H&E ordering (H is more blue-absorbing)
        if v1[0] > v2[0]:
            HE = np.array([v1, v2]).T
        else:
            HE = np.array([v2, v1]).T
        
        # Get concentrations
        Y = OD @ np.linalg.pinv(HE)
        maxC = np.percentile(Y, 99, axis=0)
        
        return HE, maxC
    
    def normalize(self, img):
        """Normalize a single image (numpy array, uint8, RGB)."""
        h, w, _ = img.shape
        
        try:
            HE, maxC = self._get_stain_matrix(img)
        except Exception:
            return img
        
        # Convert to OD
        OD = -np.log10(np.clip(img.reshape(-1, 3).astype(np.float64) / 255.0, 1e-6, 1.0))
        
        # Get concentrations
        C = OD @ np.linalg.pinv(HE)
        
        # Normalize concentrations
        C = C / maxC * self.maxCRef
        
        # Reconstruct
        OD_norm = C @ self.HERef.T
        I_norm = np.clip(255.0 * np.power(10, -OD_norm), 0, 255).astype(np.uint8)
        
        return I_norm.reshape(h, w, 3)


class StainNormTransform:
    """Torchvision-compatible transform wrapper for Macenko normalization."""
    def __init__(self):
        self.normalizer = MacenkoNormalizer()
    
    def __call__(self, img):
        # img is PIL Image
        arr = np.array(img)
        try:
            normalized = self.normalizer.normalize(arr)
            return Image.fromarray(normalized)
        except Exception:
            return img


# ============ STRONGER CLASSIFICATION HEAD ============
class StrongHead(nn.Module):
    """
    Stronger classification head for MobileNetV3.
    GlobalAvgPool → BN → Dropout(0.4) → Dense(512) → ReLU → Dropout(0.3) → Dense(2)
    """
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


# ============ MODEL BUILDER ============
def build_model():
    """Build MobileNetV3-Large with stronger head."""
    # Compatible with different torchvision versions
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    except AttributeError:
        try:
            weights = models.MobileNetV3_Large_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = 'IMAGENET1K_V2'
    
    model = models.mobilenet_v3_large(weights=weights)
    
    # Extract the feature dimension from the existing classifier
    in_features = model.classifier[0].in_features  # 960
    
    # Replace classifier with stronger head
    model.classifier = StrongHead(in_features=in_features, num_classes=2)
    
    return model.to(DEVICE)


# ============ DATA PIPELINE ============
def get_dataloaders(train_dir, val_dir, args):
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    res = args.resolution
    
    # Build transform list
    train_transform_list = []
    
    # Stain normalization first (on PIL image)
    if args.stain_norm and not args.no_stain_norm:
        train_transform_list.append(StainNormTransform())
    
    train_transform_list.extend([
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
    
    val_transform_list = []
    if args.stain_norm and not args.no_stain_norm:
        val_transform_list.append(StainNormTransform())
    val_transform_list.extend([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        imagenet_normalize,
    ])
    
    train_transforms = transforms.Compose(train_transform_list)
    val_transforms = transforms.Compose(val_transform_list)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    logger = logging.getLogger("training")
    logger.info(f"  Classes: {train_dataset.classes} → {train_dataset.class_to_idx}")
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    
    return train_loader, val_loader, train_dataset.class_to_idx


# ============ TTA VALIDATION ============
def validate_with_tta(model, val_loader, criterion, args):
    """Validate with 4-orientation TTA: original, H-flip, V-flip, both."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.size(0)
            
            # 4 TTA variants
            variants = [
                inputs,                           # original
                torch.flip(inputs, [3]),           # horizontal flip
                torch.flip(inputs, [2]),           # vertical flip
                torch.flip(inputs, [2, 3]),        # both flips
            ]
            
            # Average softmax across variants
            avg_probs = torch.zeros(batch_size, 2).to(DEVICE)
            for v in variants:
                logits = model(v)
                avg_probs += F.softmax(logits, dim=1)
            avg_probs /= len(variants)
            
            # Loss on original
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
    sens = recall_score(all_labels, all_preds, pos_label=0)  # ALL = class 0
    spec = recall_score(all_labels, all_preds, pos_label=1)  # HEM = class 1
    prec = precision_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 0], 
                           labels=[0, 1])
        # ALL is class 0, so we use P(ALL) for AUC
        # But roc_auc_score expects P(positive class)
        # If ALL=0 is our positive class, we use 1 - all_probs[:, 0]
        # Actually, let's compute correctly
        auc = roc_auc_score(1 - all_labels, all_probs[:, 0])  # P(ALL) vs is_ALL
    except ValueError:
        auc = 0.0
    
    avg_loss = total_loss / count
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1': f1,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs,
    }


# ============ TRAINING LOOP ============
def train_one_epoch(model, train_loader, criterion, optimizer, args, epoch):
    """Train one epoch with optional MixUp."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # MixUp
        use_mixup = args.mixup_alpha > 0 and random.random() < 0.5
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(train_loader.dataset)
    acc = correct / max(total, 1)
    return avg_loss, acc


def get_phase_config(epoch, total_epochs, warmup_epochs):
    """Determine training phase based on epoch."""
    if epoch < 10:
        return 1, "Head only"
    elif epoch < 30:
        return 2, "Last 25% unfrozen"
    else:
        return 3, "Full model"


def set_trainable_layers(model, phase):
    """Freeze/unfreeze layers based on training phase."""
    if phase == 1:
        # Freeze everything except classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif phase == 2:
        # Unfreeze last 25% of feature layers
        all_children = list(model.features.children())
        cutoff = int(len(all_children) * 0.75)
        for i, child in enumerate(all_children):
            for param in child.parameters():
                param.requires_grad = (i >= cutoff)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ============ EXPORT ============
def export_model(model, class_to_idx, args, suffix="v3"):
    """Export to ONNX → TFLite."""
    model.eval()
    res = args.resolution
    
    # Save PyTorch checkpoint
    pth_path = os.path.join(OUTPUT_DIR, f"mobilenetv3_large_{suffix}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'resolution': res,
        'config': vars(args),
    }, pth_path)
    print(f"  PyTorch checkpoint: {pth_path}")
    
    # ONNX export
    onnx_path = os.path.join(OUTPUT_DIR, f"mobilenetv3_large_{suffix}.onnx")
    dummy = torch.randn(1, 3, res, res).to(DEVICE)
    
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX: {onnx_path} ({onnx_size:.2f} MB)")
    
    # TFLite conversion
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
        print(f"  TFLite: {tflite_path} ({tflite_size:.2f} MB)")
    except ImportError as e:
        print(f"  TFLite conversion skipped (missing dependency: {e})")
        print(f"  Run manually: python export_scripts/onnx_to_tflite.py {onnx_path}")


# ============ MAIN ============
def train_single_split(args):
    """Train on the standard train/val split."""
    logger = logging.getLogger("training")
    log_params(logger, args)

    # CSV logger for per-epoch metrics
    csv_log = CSVLogger(os.path.join(LOG_DIR, f"{args.run_name}_metrics.csv"))

    # Auto-detect pre-normalized stain data
    use_stain = args.stain_norm and not args.no_stain_norm
    staging_dir, using_precomputed = get_staging_dir(use_stain)
    if using_precomputed:
        logger.info(f"  Using pre-normalized stain data: {staging_dir}")
        args.no_stain_norm = True  # Disable on-the-fly since data is already normalized
    else:
        logger.info(f"  Using raw data: {staging_dir}")

    train_dir = os.path.join(staging_dir, "train")
    val_dir = os.path.join(staging_dir, "val")

    train_loader, val_loader, class_to_idx = get_dataloaders(train_dir, val_dir, args)

    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing)
    logger.info(f"  Loss: FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha}, "
                f"label_smoothing={args.label_smoothing})")

    # ETA tracker
    eta = ETATracker(args.epochs)

    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    current_phase = 0

    logger.info("")
    logger.info("  Starting training...")
    logger.info("─" * 120)

    for epoch in range(args.epochs):
        # Determine training phase
        phase, phase_name = get_phase_config(epoch, args.epochs, args.warmup_epochs)

        # Phase transition
        if phase != current_phase:
            current_phase = phase
            trainable, total = set_trainable_layers(model, phase)
            logger.info(f"")
            logger.info(f"  ★ PHASE {phase}: {phase_name} "
                        f"({trainable:,}/{total:,} params trainable, "
                        f"{trainable/total*100:.1f}%)")

            # Reset optimizer for new phase
            lr = args.lr if phase == 1 else args.lr / (3 ** (phase - 1))
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=1e-4
            )
            logger.debug(f"  Optimizer reset: AdamW lr={lr:.6f}, wd=1e-4")

            # Cosine decay for remaining epochs in this phase
            remaining = args.epochs - epoch
            warmup_iters = min(args.warmup_epochs, remaining)
            cosine_iters = max(remaining - warmup_iters, 1)
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_iters)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_iters])
            patience_counter = 0  # Reset patience on phase change

        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args, epoch)

        # Validate with TTA
        val_metrics = validate_with_tta(model, val_loader, criterion, args)

        scheduler.step()

        elapsed = time.time() - t0
        eta.tick(elapsed)
        current_lr = optimizer.param_groups[0]['lr']
        eta_str, avg_epoch_time, total_elapsed = eta.get_eta(epoch)

        # Log to CSV
        csv_log.log({
            'epoch': epoch + 1,
            'phase': phase,
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

        # Console + file log line
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
            f"P{phase} │ "
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

        # Early stopping (only after phase 3 starts)
        if phase == 3 and patience_counter >= args.patience:
            logger.info(f"")
            logger.info(f"  ✋ Early stopping at epoch {epoch+1} "
                        f"(no improvement for {args.patience} epochs)")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final report
    logger.info("")
    logger.info("=" * 65)
    logger.info("  FINAL RESULTS")
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

    _, _, total_elapsed = eta.get_eta(args.epochs - 1)
    logger.info(f"")
    logger.info(f"  Total training time: {total_elapsed}")
    logger.info(f"  Logs saved to: {LOG_DIR}/{args.run_name}*")
    logger.info("=" * 65)

    # Export
    export_model(model, class_to_idx, args, suffix="v3")

    return best_metrics


def train_kfold(args):
    """K-fold cross-validation training."""
    print(f"\n{'='*60}")
    print(f"  {args.folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    # Combine train + val for k-fold
    all_dir = os.path.join(STAGING_DIR, "train")  # Use train as the full dataset
    
    # Get all image paths and labels
    dataset = datasets.ImageFolder(all_dir)
    targets = np.array([s[1] for s in dataset.samples])
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(range(len(dataset)), targets)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1}/{args.folds}")
        print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")
        print(f"{'='*60}")
        
        # Create fold-specific data loaders
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        res = args.resolution
        
        train_transform_list = []
        if args.stain_norm and not args.no_stain_norm:
            train_transform_list.append(StainNormTransform())
        train_transform_list.extend([
            transforms.Resize((res + 32, res + 32)),
            transforms.RandomCrop(res),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(), imagenet_normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
        
        val_transform_list = []
        if args.stain_norm and not args.no_stain_norm:
            val_transform_list.append(StainNormTransform())
        val_transform_list.extend([
            transforms.Resize((res, res)),
            transforms.ToTensor(), imagenet_normalize,
        ])
        
        train_dataset = datasets.ImageFolder(all_dir, transform=transforms.Compose(train_transform_list))
        val_dataset = datasets.ImageFolder(all_dir, transform=transforms.Compose(val_transform_list))
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True)
        
        # Train this fold (simplified — just full training, no phase transitions for speed)
        model = build_model()
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)
        
        # Full unfreeze from start for k-fold (faster per fold)
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_f1 = 0.0
        best_state = None
        
        for epoch in range(min(args.epochs, 50)):  # Cap at 50 per fold
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args, epoch)
            val_metrics = validate_with_tta(model, val_loader, criterion, args)
            scheduler.step()
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics.copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"    E{epoch+1}: Acc={val_metrics['accuracy']:.4f} "
                      f"F1={val_metrics['f1']:.4f} "
                      f"Sens={val_metrics['sensitivity']:.4f}")
        
        print(f"  Fold {fold_idx+1} Best: Acc={best_metrics['accuracy']:.4f} "
              f"F1={best_metrics['f1']:.4f} "
              f"Sens={best_metrics['sensitivity']:.4f} "
              f"Spec={best_metrics['specificity']:.4f}")
        fold_results.append(best_metrics)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.folds}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for metric in ['accuracy', 'f1', 'sensitivity', 'specificity', 'auc']:
        values = [r[metric] for r in fold_results]
        print(f"  {metric:>12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.run_name)

    if args.folds > 0:
        train_kfold(args)
    else:
        train_single_split(args)


if __name__ == "__main__":
    main()
