"""
Train MobileNetV3-Small on C-NMC Dataset — PyTorch + GPU Edition.
Pipeline: PyTorch train → ONNX export → TFLite convert → Pi 5 deploy.

3-Phase Training:
  Phase 1: Frozen base, train head only (5 epochs)
  Phase 2: Unfreeze top layers (10 epochs)
  Phase 3: Unfreeze ALL layers with very low LR (15 epochs)
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
CNMC_DIR = r"c:\Open Source\leukiemea\C-NMC"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 8    # cumulative
PHASE3_EPOCHS = 20   # cumulative
LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ STAGE DATA ============
def stage_data():
    """Create train/val split from C-NMC folds (if not already staged)."""
    import shutil
    train_dir = os.path.join(STAGING_DIR, "train")
    val_dir = os.path.join(STAGING_DIR, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_count = sum(len(os.listdir(os.path.join(train_dir, c))) for c in ["all", "hem"] if os.path.exists(os.path.join(train_dir, c)))
        if train_count > 7000:
            print("  Staging directory already exists, skipping copy.")
            return train_dir, val_dir

    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    for split_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(split_dir, "all"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "hem"), exist_ok=True)

    train_folds = ["fold_0", "fold_1"]
    val_folds = ["fold_2"]

    for fold in train_folds:
        for cls in ["all", "hem"]:
            src = os.path.join(CNMC_DIR, fold, fold, cls)
            dst = os.path.join(train_dir, cls)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    for fold in val_folds:
        for cls in ["all", "hem"]:
            src = os.path.join(CNMC_DIR, fold, fold, cls)
            dst = os.path.join(val_dir, cls)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    return train_dir, val_dir


# ============ DATA LOADERS ============
def get_dataloaders(train_dir, val_dir):
    # ImageNet normalization for pretrained MobileNetV3
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.15, contrast=0.1),
        transforms.ToTensor(),
        normalize,
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

    # Class weights for imbalanced data (DISABLED for true unweighted baseline)
    class_counts = np.bincount([label for _, label in train_dataset.samples])
    total = sum(class_counts)
    print(f"  Class counts: {dict(zip(train_dataset.classes, class_counts))}")
    class_weights = None # Disabled

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights, train_dataset.class_to_idx


# ============ MODEL ============
def build_model():
    """MobileNetV3-Small with custom 2-class head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace classifier head for binary classification
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),  # 2 classes: ALL, HEM
    )
    return model


# ============ TRAINING ============
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tp_fp = 0
    tp_fn = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # For recall/precision of class 0 (ALL)
        tp += ((predicted == 0) & (labels == 0)).sum().item()
        tp_fp += (predicted == 0).sum().item()
        tp_fn += (labels == 0).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}, acc: {correct/total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    recall = tp / tp_fn if tp_fn > 0 else 0
    precision = tp / tp_fp if tp_fp > 0 else 0

    return epoch_loss, epoch_acc, recall, precision


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tp_fp = 0
    tp_fn = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        tp += ((predicted == 0) & (labels == 0)).sum().item()
        tp_fp += (predicted == 0).sum().item()
        tp_fn += (labels == 0).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    recall = tp / tp_fn if tp_fn > 0 else 0
    precision = tp / tp_fp if tp_fp > 0 else 0

    return epoch_loss, epoch_acc, recall, precision


def train():
    print("=" * 60)
    print("  MobileNetV3-Small — C-NMC PyTorch Training (GPU)")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # Stage data
    print("\n[1/5] Staging C-NMC dataset...")
    train_dir, val_dir = stage_data()

    # Data loaders
    print("\n[2/5] Setting up data loaders...")
    train_loader, val_loader, class_weights, class_to_idx = get_dataloaders(train_dir, val_dir)

    # Build model (Unweighted)
    print("\n[3/5] Building MobileNetV3-Small (UNWEIGHTED)...")
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Track best model
    best_val_recall = 0.0
    best_model_state = None

    # CSV log
    log_path = os.path.join(OUTPUT_DIR, "training_log_pytorch_unweighted.csv")
    with open(log_path, "w") as f:
        f.write("epoch,phase,train_loss,train_acc,train_recall,train_precision,val_loss,val_acc,val_recall,val_precision,lr\n")

    def log_epoch(epoch, phase, t_loss, t_acc, t_rec, t_prec, v_loss, v_acc, v_rec, v_prec, lr):
        with open(log_path, "a") as f:
            f.write(f"{epoch},{phase},{t_loss:.4f},{t_acc:.4f},{t_rec:.4f},{t_prec:.4f},{v_loss:.4f},{v_acc:.4f},{v_rec:.4f},{v_prec:.4f},{lr}\n")

    # ===== PHASE 1: Frozen base, train head only =====
    print("\n[4/5] Phase 1: Head only (base frozen, 5 epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        start = time.time()
        t_loss, t_acc, t_rec, t_prec = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc, v_rec, v_prec = validate(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"  Epoch {epoch}/{PHASE1_EPOCHS} ({elapsed:.0f}s) — "
              f"train: acc={t_acc:.4f} rec={t_rec:.4f} | "
              f"val: acc={v_acc:.4f} rec={v_rec:.4f} prec={v_prec:.4f}")

        log_epoch(epoch, "P1", t_loss, t_acc, t_rec, t_prec, v_loss, v_acc, v_rec, v_prec, LR)

        if v_rec > best_val_recall:
            best_val_recall = v_rec
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    ★ New best val_recall: {v_rec:.4f}")

    # ===== PHASE 2: Unfreeze top layers =====
    print(f"\n  Phase 2: Unfreeze top layers (epochs {PHASE1_EPOCHS+1}-{PHASE2_EPOCHS})...")
    for param in model.features.parameters():
        param.requires_grad = True
    # Freeze early layers (keep first 8 blocks frozen)
    for i, (name, param) in enumerate(model.features.named_parameters()):
        if i < len(list(model.features.parameters())) // 2:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR / 10)

    for epoch in range(PHASE1_EPOCHS + 1, PHASE2_EPOCHS + 1):
        start = time.time()
        t_loss, t_acc, t_rec, t_prec = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc, v_rec, v_prec = validate(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"  Epoch {epoch}/{PHASE2_EPOCHS} ({elapsed:.0f}s) — "
              f"train: acc={t_acc:.4f} rec={t_rec:.4f} | "
              f"val: acc={v_acc:.4f} rec={v_rec:.4f} prec={v_prec:.4f}")

        log_epoch(epoch, "P2", t_loss, t_acc, t_rec, t_prec, v_loss, v_acc, v_rec, v_prec, LR/10)

        if v_rec > best_val_recall:
            best_val_recall = v_rec
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    ★ New best val_recall: {v_rec:.4f}")

    # ===== PHASE 3: Full fine-tuning, ALL layers =====
    print(f"\n  Phase 3: Full fine-tuning — ALL layers (epochs {PHASE2_EPOCHS+1}-{PHASE3_EPOCHS})...")
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All {trainable:,} params are now trainable.")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(PHASE2_EPOCHS + 1, PHASE3_EPOCHS + 1):
        start = time.time()
        t_loss, t_acc, t_rec, t_prec = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc, v_rec, v_prec = validate(model, val_loader, criterion)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch}/{PHASE3_EPOCHS} ({elapsed:.0f}s) — "
              f"train: acc={t_acc:.4f} rec={t_rec:.4f} | "
              f"val: acc={v_acc:.4f} rec={v_rec:.4f} prec={v_prec:.4f} | lr={current_lr:.1e}")

        log_epoch(epoch, "P3", t_loss, t_acc, t_rec, t_prec, v_loss, v_acc, v_rec, v_prec, current_lr)
        scheduler.step(v_rec)

        if v_rec > best_val_recall:
            best_val_recall = v_rec
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    ★ New best val_recall: {v_rec:.4f}")

    # Restore best model
    print(f"\n  Restoring best model (val_recall={best_val_recall:.4f})...")
    model.load_state_dict(best_model_state)

    # ===== EXPORT =====
    print("\n[5/5] Exporting model...")

    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_val_recall': best_val_recall,
    }, os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc_unweighted.pth"))

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX model saved: {onnx_path} ({onnx_size:.2f} MB)")

    # Convert ONNX → TFLite
    print("  Converting ONNX → TFLite...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_saved_model_path = os.path.join(OUTPUT_DIR, "tf_saved_model")
        tf_rep.export_graph(tf_saved_model_path)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc_unweighted.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  TFLite model saved: {tflite_path} ({tflite_size:.2f} MB)")
    except ImportError:
        print("  ⚠️  onnx-tf not installed. Run: pip install onnx onnx-tf")
        print("  ONNX model is ready for manual conversion.")

    # Final eval
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val_recall: {best_val_recall:.4f}")
    print(f"  Device used: {DEVICE}")
    print(f"  Models saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
