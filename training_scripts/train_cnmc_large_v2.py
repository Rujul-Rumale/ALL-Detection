"""
Train MobileNetV3-Large v2 on C-NMC Dataset — Improved Version.
Fixes from v1:
  - Optimizes for F1 score (balance precision & recall) instead of recall-only
  - Stronger augmentation: affine, Gaussian blur, random erasing
  - AdamW with cosine annealing instead of fixed LR
  - Label smoothing to prevent overconfident predictions
Pipeline: PyTorch train → ONNX export → TFLite convert → Pi 5 deploy.
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

# ============ CONFIG ============
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
LR = 3e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ DATA LOADERS ============
def get_dataloaders(train_dir, val_dir):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Aggressive augmentation — fixes orientation sensitivity
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize larger then crop
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),  # Full rotation range
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random cutout
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    print(f"  Classes: {train_dataset.class_to_idx}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Balanced class weights (full inverse frequency)
    class_counts = np.bincount([label for _, label in train_dataset.samples])
    total = sum(class_counts)
    class_weights = torch.tensor([
        total / (2.0 * c) for c in class_counts
    ], dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {dict(zip(train_dataset.classes, class_weights.cpu().tolist()))}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights, train_dataset.class_to_idx


# ============ MODEL ============
def build_model():
    """MobileNetV3-Large with custom 2-class head + dropout."""
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.4),      # Increased dropout
        nn.Linear(256, 2),
    )
    return model


# ============ METRICS ============
def compute_metrics(all_labels, all_preds, class_to_idx):
    all_idx = class_to_idx.get('all', 0)
    labels = np.array(all_labels)
    preds = np.array(all_preds)
    
    acc = np.mean(labels == preds)
    
    tp = np.sum((preds == all_idx) & (labels == all_idx))
    fp = np.sum((preds == all_idx) & (labels != all_idx))
    fn = np.sum((preds != all_idx) & (labels == all_idx))
    tn = np.sum((preds != all_idx) & (labels != all_idx))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return acc, recall, precision, specificity, f1


# ============ TRAINING ============
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), all_labels, all_preds


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), all_labels, all_preds


def train():
    print("=" * 60)
    print("  MobileNetV3-Large v2 — Improved C-NMC Training")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    train_dir = os.path.join(STAGING_DIR, "train")
    val_dir = os.path.join(STAGING_DIR, "val")
    
    if not os.path.exists(train_dir):
        print("ERROR: cnmc_staging/train not found. Run train_cnmc_large.py first to stage data.")
        return

    train_loader, val_loader, class_weights, class_to_idx = get_dataloaders(train_dir, val_dir)

    # Build Model
    print("\n  Building MobileNetV3-Large...")
    model = build_model().to(DEVICE)
    
    # Label smoothing + class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW + Cosine Annealing
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    MAX_PATIENCE = 8

    # Phase 1: Freeze base for 3 epochs
    print("\n  Phase 1: Head only (3 epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False

    for epoch in range(1, 4):
        start = time.time()
        t_loss, t_labels, t_preds = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_labels, v_preds = validate(model, val_loader, criterion)
        scheduler.step(epoch)
        
        t_acc, t_rec, t_prec, _, t_f1 = compute_metrics(t_labels, t_preds, class_to_idx)
        v_acc, v_rec, v_prec, v_spec, v_f1 = compute_metrics(v_labels, v_preds, class_to_idx)
        
        print(f"  [{epoch:02d}] {time.time()-start:.0f}s | "
              f"train: acc={t_acc:.3f} f1={t_f1:.3f} | "
              f"val: acc={v_acc:.3f} rec={v_rec:.3f} prec={v_prec:.3f} f1={v_f1:.3f}")

    # Phase 2: Unfreeze all, train with low LR
    print(f"\n  Phase 2: Full fine-tuning ({EPOCHS - 3} epochs)...")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=LR / 3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 3)

    for epoch in range(4, EPOCHS + 1):
        start = time.time()
        t_loss, t_labels, t_preds = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_labels, v_preds = validate(model, val_loader, criterion)
        scheduler.step()
        
        t_acc, t_rec, t_prec, _, t_f1 = compute_metrics(t_labels, t_preds, class_to_idx)
        v_acc, v_rec, v_prec, v_spec, v_f1 = compute_metrics(v_labels, v_preds, class_to_idx)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"  [{epoch:02d}] {time.time()-start:.0f}s | "
              f"train: acc={t_acc:.3f} f1={t_f1:.3f} | "
              f"val: acc={v_acc:.3f} rec={v_rec:.3f} prec={v_prec:.3f} spec={v_spec:.3f} f1={v_f1:.3f} | lr={lr:.1e}")

        # Save best model based on F1 (balanced metric)
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    ★ New best F1: {best_f1:.4f} (rec={v_rec:.3f}, prec={v_prec:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"    Early stop — no improvement for {MAX_PATIENCE} epochs")
                break

    # Restore best
    print(f"\n  Restoring best model (F1={best_f1:.4f})...")
    model.load_state_dict(best_model_state)

    # Save PyTorch
    save_path = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v2_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_f1': best_f1,
    }, save_path)
    print(f"  Saved: {save_path}")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v2.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Convert to TFLite
    print("  Converting ONNX → TFLite...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_saved = os.path.join(OUTPUT_DIR, "tf_saved_model_v2")
        tf_rep.export_graph(tf_saved)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join(OUTPUT_DIR, "mobilenetv3_large_v2.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"  TFLite saved: {tflite_path} ({os.path.getsize(tflite_path)/1024/1024:.1f} MB)")
    except ImportError:
        print("  ⚠  onnx-tf not installed. ONNX is ready for manual conversion.")

    print(f"\n{'='*60}")
    print(f"  DONE — Best F1: {best_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
