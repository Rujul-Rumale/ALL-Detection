"""
Visual Evaluation of MobileNetV3-Large v2 on C-NMC Validation Set.

Creates a grid visualization showing:
  - Cell images from both ALL and HEM classes
  - Ground truth labels
  - Model predictions with confidence scores
  - Color-coded borders: green=correct, red=incorrect
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"c:\Open Source\leukiemea\models\mobilenetv3_large_v2_best.pth"
VAL_DIR = r"c:\Open Source\leukiemea\cnmc_staging\val"
OUTPUT_DIR = r"c:\Open Source\leukiemea\evaluation_outputs"
IMG_SIZE = 224

# Number of sample images to show per class
SAMPLES_PER_CLASS = 8
# Grid: 4 rows x 8 columns (top 2 rows = ALL, bottom 2 rows = HEM)
COLS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL SETUP
# ==========================================
def load_model():
    """Load the trained MobileNetV3-Large v2 model."""
    print(f"Loading model from {MODEL_PATH}...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  Classes: {class_to_idx}")
    return model, class_to_idx, idx_to_class


# ==========================================
# DATA SETUP
# ==========================================
def get_val_dataset():
    """Load the validation dataset (with and without transforms)."""
    # Transform for model inference
    inference_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Raw transform (just resize, no normalization) for display
    display_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    inference_dataset = datasets.ImageFolder(VAL_DIR, transform=inference_transform)
    display_dataset = datasets.ImageFolder(VAL_DIR, transform=display_transform)

    return inference_dataset, display_dataset


def sample_indices(dataset, class_to_idx, samples_per_class):
    """Sample balanced indices from each class."""
    all_idx_value = class_to_idx['all']
    hem_idx_value = class_to_idx['hem']

    all_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == all_idx_value]
    hem_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == hem_idx_value]

    random.seed(42)  # Reproducible samples
    sampled_all = random.sample(all_indices, min(samples_per_class, len(all_indices)))
    sampled_hem = random.sample(hem_indices, min(samples_per_class, len(hem_indices)))

    return sampled_all, sampled_hem


# ==========================================
# VISUALIZATION
# ==========================================
def create_evaluation_grid(model, inference_dataset, display_dataset,
                           class_to_idx, idx_to_class,
                           sampled_all, sampled_hem):
    """Create a beautiful grid visualization of model predictions."""

    all_indices = sampled_all
    hem_indices = sampled_hem

    total_samples = len(all_indices) + len(hem_indices)
    n_rows = 4  # 2 rows for ALL, 2 rows for HEM
    n_cols = COLS

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 3.5))
    fig.patch.set_facecolor('#0d1117')

    # Title
    fig.suptitle('MobileNetV3-Large v2 — C-NMC Validation Evaluation',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # Section labels
    fig.text(0.02, 0.77, 'ALL\n(Leukemia)', fontsize=13, fontweight='bold',
             color='#ff6b6b', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#ff6b6b', alpha=0.9))
    fig.text(0.02, 0.30, 'HEM\n(Healthy)', fontsize=13, fontweight='bold',
             color='#51cf66', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#51cf66', alpha=0.9))

    def plot_cell(ax, idx, model, inference_dataset, display_dataset, idx_to_class):
        """Plot a single cell with prediction info."""
        # Get display image (un-normalized)
        display_img, gt_label = display_dataset[idx]
        display_img_np = display_img.permute(1, 2, 0).numpy()  # CHW -> HWC

        # Get model prediction
        with torch.no_grad():
            input_tensor, _ = inference_dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            confidence = probs[pred_label] * 100

        gt_name = idx_to_class[gt_label].upper()
        pred_name = idx_to_class[pred_label].upper()
        correct = (pred_label == gt_label)

        # Show image
        ax.imshow(display_img_np)

        # Color-coded border
        border_color = '#51cf66' if correct else '#ff4757'
        border_width = 4
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

        ax.set_xticks([])
        ax.set_yticks([])

        # Text below image
        # Ground truth
        ax.set_xlabel(
            f"GT: {gt_name}  |  Pred: {pred_name}\n"
            f"Conf: {confidence:.1f}%"
            + ("  ✓" if correct else "  ✗"),
            fontsize=8.5, fontweight='bold',
            color='#c9d1d9',
            labelpad=6
        )

        # Result indicator on image
        indicator = "✓" if correct else "✗"
        indicator_color = '#51cf66' if correct else '#ff4757'
        ax.text(0.95, 0.05, indicator, transform=ax.transAxes,
                fontsize=18, fontweight='bold', color=indicator_color,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))

        # Confidence bar
        bar_y = 0.92
        bar_height = 0.05
        # Background bar
        ax.add_patch(plt.Rectangle((0.05, bar_y), 0.9, bar_height,
                                    transform=ax.transAxes, facecolor='#333',
                                    alpha=0.7, zorder=5))
        # Confidence fill
        bar_color = '#51cf66' if confidence > 80 else '#ffd93d' if confidence > 60 else '#ff6b6b'
        ax.add_patch(plt.Rectangle((0.05, bar_y), 0.9 * (confidence / 100), bar_height,
                                    transform=ax.transAxes, facecolor=bar_color,
                                    alpha=0.9, zorder=6))

        return correct, confidence

    # Track stats
    total_correct = 0
    all_confidences = []

    # Plot ALL images (rows 0-1)
    for i, idx in enumerate(all_indices):
        row = i // n_cols
        col = i % n_cols
        if row >= 2:
            break
        ax = axes[row, col]
        ax.set_facecolor('#0d1117')
        correct, conf = plot_cell(ax, idx, model, inference_dataset, display_dataset, idx_to_class)
        total_correct += int(correct)
        all_confidences.append(conf)

    # Plot HEM images (rows 2-3)
    for i, idx in enumerate(hem_indices):
        row = 2 + (i // n_cols)
        col = i % n_cols
        if row >= 4:
            break
        ax = axes[row, col]
        ax.set_facecolor('#0d1117')
        correct, conf = plot_cell(ax, idx, model, inference_dataset, display_dataset, idx_to_class)
        total_correct += int(correct)
        all_confidences.append(conf)

    # Hide unused axes
    used_all = min(len(all_indices), 2 * n_cols)
    used_hem = min(len(hem_indices), 2 * n_cols)
    for i in range(used_all, 2 * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')
    for i in range(used_hem, 2 * n_cols):
        axes[2 + i // n_cols, i % n_cols].axis('off')

    # Summary stats bar at bottom
    total = used_all + used_hem
    acc = total_correct / total * 100 if total > 0 else 0
    avg_conf = np.mean(all_confidences) if all_confidences else 0

    fig.text(0.5, 0.01,
             f"Sampled Accuracy: {acc:.1f}%  |  "
             f"Avg Confidence: {avg_conf:.1f}%  |  "
             f"Correct: {total_correct}/{total}  |  "
             f"Model: MobileNetV3-Large v2",
             fontsize=11, color='#8b949e', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#30363d'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='#51cf66', linewidth=3, label='Correct'),
        mpatches.Patch(facecolor='none', edgecolor='#ff4757', linewidth=3, label='Incorrect'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=10, facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#c9d1d9', framealpha=0.9)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.06,
                        hspace=0.55, wspace=0.25)

    return fig


# ==========================================
# FULL DATASET EVALUATION SUMMARY
# ==========================================
def run_full_evaluation(model, inference_dataset, class_to_idx, idx_to_class):
    """Run evaluation on the entire validation set and print metrics."""
    loader = DataLoader(inference_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"\nRunning full validation on {len(inference_dataset)} images...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    all_idx = class_to_idx['all']
    hem_idx = class_to_idx['hem']

    acc = np.mean(all_preds == all_labels) * 100
    tp = np.sum((all_preds == all_idx) & (all_labels == all_idx))
    fp = np.sum((all_preds == all_idx) & (all_labels == hem_idx))
    fn = np.sum((all_preds != all_idx) & (all_labels == all_idx))
    tn = np.sum((all_preds != all_idx) & (all_labels == hem_idx))

    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print("\n" + "=" * 55)
    print("  FULL VALIDATION SET RESULTS")
    print("=" * 55)
    print(f"  Total Images:   {len(inference_dataset)}")
    print(f"  Accuracy:       {acc:.2f}%")
    print(f"  Sensitivity:    {sensitivity:.2f}%  (ALL recall)")
    print(f"  Specificity:    {specificity:.2f}%  (HEM recall)")
    print(f"  Precision:      {precision:.2f}%")
    print(f"  F1 Score:       {f1:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred ALL   Pred HEM")
    print(f"  Actual ALL       {tp:<10} {fn:<10}")
    print(f"  Actual HEM       {fp:<10} {tn:<10}")
    print("=" * 55)

    return acc, sensitivity, specificity, f1


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model, class_to_idx, idx_to_class = load_model()

    # Load datasets
    inference_dataset, display_dataset = get_val_dataset()

    # Sample images
    sampled_all, sampled_hem = sample_indices(
        inference_dataset, class_to_idx, SAMPLES_PER_CLASS * 2  # 2 rows worth
    )

    # Create visualization grid
    print("\nGenerating visualization grid...")
    fig = create_evaluation_grid(
        model, inference_dataset, display_dataset,
        class_to_idx, idx_to_class,
        sampled_all, sampled_hem
    )

    # Save
    output_path = os.path.join(OUTPUT_DIR, "cnmc_visual_evaluation.png")
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    print(f"\nVisualization saved to: {output_path}")
    plt.close(fig)

    # Run full eval metrics
    run_full_evaluation(model, inference_dataset, class_to_idx, idx_to_class)

    print("\nDone!")


if __name__ == "__main__":
    main()
