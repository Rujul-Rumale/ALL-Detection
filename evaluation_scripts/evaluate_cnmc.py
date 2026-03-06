import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"c:\Open Source\leukiemea\models\mobilenetv3_large_cnmc_best.pth"
VAL_DIR = r"c:\Open Source\leukiemea\cnmc_staging\val"
IMG_SIZE = 224
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# MODEL SETUP
# ==========================================
def load_trained_model():
    print(f"Loading MobileNetV3-Large from {MODEL_PATH}...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),  # 2 classes: ALL, HEM
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint['class_to_idx']

# ==========================================
# EVALUATION LOGIC
# ==========================================
def evaluate_cnmc():
    model, class_to_idx = load_trained_model()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Evaluating on C-NMC Validation Set: {len(val_dataset)} images...")
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    # Class 0 is 'all' (Leukemia), Class 1 is 'hem' (Healthy)
    all_idx = class_to_idx['all']
    hem_idx = class_to_idx['hem']
    
    tn, fp, fn, tp = cm.ravel() if all_idx == 1 else cm[::-1, ::-1].ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*50)
    print("C-NMC VALIDATION ACCURACY REPORT")
    print("="*50)
    print(f"Total Images Evaluated: {len(val_dataset)}")
    print(f"Overall Accuracy:  {acc*100:.2f}%")
    print(f"Sensitivity (ALL): {sensitivity*100:.2f}% (True Positive Rate)")
    print(f"Specificity (HEM): {specificity*100:.2f}% (True Negative Rate)")
    
    print("\nConfusion Matrix:")
    print(f"                 Predicted ALL  Predicted HEM")
    print(f"Actual ALL ({tp+fn:<4})   {tp:<13}  {fn:<13}")
    print(f"Actual HEM ({tn+fp:<4})   {fp:<13}  {tn:<13}")
    print("="*50)

if __name__ == "__main__":
    evaluate_cnmc()
