import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
IMG_SIZE = 224 # EfficientNetV2-S accepts 224-384. 224 is a good balance for memory
BATCH_SIZE = 16 # Reduced batch size for larger model
NUM_WORKERS = 4
EPOCHS = 25
LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA LOADING
# ==========================================
def get_dataloaders(train_dir, val_dir):
    # Aggressive augmentation for high accuracy
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Calculate balanced weights to handle C-NMC 2:1 ratio
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

# ==========================================
# METRICS HELPER
# ==========================================
def calculate_metrics(labels, preds, class_to_idx):
    acc = np.mean(labels == preds)
    # ALL is positive class (class 0 likely)
    all_idx = class_to_idx['all']
    tp = np.sum((labels == all_idx) & (preds == all_idx))
    fn = np.sum((labels == all_idx) & (preds != all_idx))
    fp = np.sum((labels != all_idx) & (preds == all_idx))
    tn = np.sum((labels != all_idx) & (preds != all_idx))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, recall, specificity

# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    print(f"\n============================================================")
    print(f"  EfficientNetV2-S — High Accuracy PyTorch Training")
    print(f"  Device: {DEVICE}")
    print(f"============================================================\n")

    train_dir = os.path.join(STAGING_DIR, "train")
    val_dir = os.path.join(STAGING_DIR, "val")
    train_loader, val_loader, class_weights, class_to_idx = get_dataloaders(train_dir, val_dir)

    # Build Model
    print("\n[1/3] Building EfficientNetV2-S...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    # Replace head for 2 classes (ALL vs HEM)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    print(f"\n[2/3] Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        t_acc, t_rec, t_spec = calculate_metrics(np.array(train_labels), np.array(train_preds), class_to_idx)
        v_acc, v_rec, v_spec = calculate_metrics(np.array(val_labels), np.array(val_preds), class_to_idx)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {t_acc:.4f} Seq: {t_rec:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {v_acc:.4f} Sens: {v_rec:.4f} Spec: {v_spec:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")

        # Save Best Model Based on ACCURACY
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            print(f"  ★ New best val_acc: {best_val_acc:.4f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'best_val_acc': best_val_acc,
            }, os.path.join(OUTPUT_DIR, "efficientnet_v2_s_cnmc.pth"))

    print("\n[3/3] Training Complete")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
