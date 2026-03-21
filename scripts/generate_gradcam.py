"""
generate_gradcam.py
===================
Generates GradCAM visualizations for the ALL leukemia classifier.
Samples 4 TP, 4 TN, 4 FP, 4 FN from the validation fold and saves
individual overlays + a 4×4 summary grid at 300 DPI.

Usage (from project root):
    python scripts/generate_gradcam.py \\
        --checkpoint outputs/local_effb0_final1/local_effb0_final1_fold1_20260316_220343_best.pth \\
        --model effb0 \\
        --fold 1 \\
        --splits cv_splits/cv_splits_3fold.json \\
        --n_samples 16 \\
        --out outputs/gradcam/

Dependencies:
    pip install grad-cam timm albumentations torch pillow
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Albumentations val transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Constants matching training ────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_RES     = 320
CLASS_NAMES   = ["ALL", "HEM"]   # 0=ALL (blast), 1=HEM (healthy)

TIMM_NAME_MAP = {
    "mnv3l": "mobilenetv3_large_100",
    "effb0":  "efficientnet_b0",
    "effb4":  "efficientnet_b4",
    "rn50":   "resnet50",
}

HIDDEN_DIM_MAP = {"mnv3l": 128, "effb0": 128, "effb4": 256, "rn50": 256}
DROPOUT_MAP    = {"mnv3l": 0.6, "effb0": 0.5, "effb4": 0.4, "rn50": 0.4}


# ── Model construction (mirrors train_colab.py exactly) ───────────────────────
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


def build_model(model_name, num_classes=2):
    if model_name == "effb0":
        timm_name = "efficientnet_b0"
        in_features = 1280
    elif model_name == "mnv3l":
        timm_name = "mobilenetv3_large_100"
        in_features = 1280
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
    model = nn.Sequential(backbone, ConstrainedHead(in_features, num_classes))
    return model


# ── Albumentations val transform ───────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(INPUT_RES, INPUT_RES),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def load_image_tensor(path: str, device: torch.device):
    """Loads image, applies val transform. Returns (tensor[1,3,H,W], rgb_float32[H,W,3])."""
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (INPUT_RES, INPUT_RES))
    img_float   = img_resized.astype(np.float32) / 255.0  # for overlay

    transformed = val_transform(image=img_resized)
    tensor = transformed["image"].unsqueeze(0).to(device)
    return tensor, img_float


# ── Inference helper ───────────────────────────────────────────────────────────

def run_inference(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float]:
    """Returns (pred_class_idx, all_probability)."""
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits.float(), dim=1)
    all_prob  = float(probs[0, 0])   # index 0 = ALL
    pred_idx  = int(probs[0].argmax())
    return pred_idx, all_prob


# ── GradCAM target layer resolver ─────────────────────────────────────────────

def get_target_layer(model: nn.Sequential, arch: str) -> nn.Module:
    """
    Returns the last inverted residual block (model[0].blocks[-1]).
    Works for mobilenetv3_large_100 and efficientnet_b0.
    """
    backbone = model[0]
    # Target layer for GradCAM - last block of the backbone
    if arch == "effb0":
        return backbone.blocks[-1]
    elif arch == "mnv3l":
        return backbone.blocks[-1]
    # Fallback: last child module that has parameters
    blocks   = list(backbone.children())
    for child in reversed(blocks):
        if list(child.parameters()):
            return child
    return blocks[-1]


# ── Sample TP/TN/FP/FN ────────────────────────────────────────────────────────

def sample_cases(model, val_images, device, n_per_cat=4, seed=42):
    """
    Runs inference on a shuffled subset of val images to find TP/TN/FP/FN.
    val_images: list of [path, label] where label 0=ALL, 1=HEM.
    Returns dict with keys "TP","TN","FP","FN", each a list of dicts.
    """
    rng = random.Random(seed)
    pool = list(val_images)
    rng.shuffle(pool)

    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    needed  = n_per_cat * 4

    for path, true_label in pool:
        if all(len(v) >= n_per_cat for v in buckets.values()):
            break
        if not os.path.exists(path):
            continue

        try:
            tensor, img_float = load_image_tensor(path, device)
        except Exception:
            continue

        pred_idx, all_prob = run_inference(model, tensor)

        # true_label: 0=ALL(blast), 1=HEM(healthy)
        # TP: true ALL predicted as ALL
        # TN: true HEM predicted as HEM
        # FP: true HEM predicted as ALL
        # FN: true ALL predicted as HEM
        entry = {
            "path": path, "true": true_label, "pred": pred_idx,
            "all_prob": all_prob, "tensor": tensor, "img_float": img_float,
        }
        if true_label == 0 and pred_idx == 0 and len(buckets["TP"]) < n_per_cat:
            buckets["TP"].append(entry)
        elif true_label == 1 and pred_idx == 1 and len(buckets["TN"]) < n_per_cat:
            buckets["TN"].append(entry)
        elif true_label == 1 and pred_idx == 0 and len(buckets["FP"]) < n_per_cat:
            buckets["FP"].append(entry)
        elif true_label == 0 and pred_idx == 1 and len(buckets["FN"]) < n_per_cat:
            buckets["FN"].append(entry)

    return buckets


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GradCAM visualizer for ALL leukemia classifier")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pth checkpoint")
    p.add_argument("--model",      required=True, choices=["mnv3l", "effb0", "effb4", "rn50"],
                   help="Model architecture key")
    p.add_argument("--fold",       required=True, type=int, choices=[1, 2, 3],
                   help="CV fold number (selects val split)")
    p.add_argument("--splits",     default="cv_splits/cv_splits_3fold.json",
                   help="Path to cv_splits_3fold.json")
    p.add_argument("--n_samples",  type=int, default=16,
                   help="Total samples (n_per_cat = n//4)")
    p.add_argument("--out",        default="outputs/gradcam/",
                   help="Output directory")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per_cat = max(1, args.n_samples // 4)

    # IEEE Styling settings
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {args.fold}  |  n_per_cat: {n_per_cat}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(args.model).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", {}).get("module", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Model loaded.")

    # ── Load val images ───────────────────────────────────────────────────────
    with open(args.splits, "r") as f:
        splits = json.load(f)
    fold_key   = f"fold_{args.fold}"
    val_images = splits["folds"][fold_key]["val_images"]   # [[path, label], ...]
    print(f"Val images: {len(val_images)}")

    # ── Sample TP/TN/FP/FN ──────────────────────────────────────────────────
    print("Running inference to find TP/TN/FP/FN ...")
    buckets = sample_cases(model, val_images, device, n_per_cat=n_per_cat)
    for cat, items in buckets.items():
        print(f"  {cat}: {len(items)} samples")

    # ── GradCAM setup ────────────────────────────────────────────────────────
    target_layer = get_target_layer(model, args.model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # ── Generate overlays ────────────────────────────────────────────────────
    CAT_COLORS = {
        "TP": "#40C057",    # green
        "TN": "#4CC9F0",    # cyan
        "FP": "#FF4D4D",    # red
        "FN": "#FFAA00",    # amber
    }
    CAT_DESC = {
        "TP": "TP",
        "TN": "TN",
        "FP": "FP",
        "FN": "FN",
    }

    grid_images = []   # (img, cat, idx, true_label, pred_label)

    for cat, entries in buckets.items():
        for idx, entry in enumerate(entries):
            tensor    = entry["tensor"]       # [1,3,H,W]
            img_float = entry["img_float"]    # [H,W,3] float32
            true_label = entry["true"]
            pred_label = entry["pred"]

            # Run GradCAM (target=None → highest scoring class)
            grayscale_cam = cam(input_tensor=tensor)[0]  # [H,W]

            # Overlay
            overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            # Save individual PNG
            fname = f"{cat}_{idx+1}_label{CLASS_NAMES[true_label]}_pred{CLASS_NAMES[pred_label]}.png"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            grid_images.append((overlay, cat, idx + 1, true_label, pred_label))

    # ── 4×4 summary grid ────────────────────────────────────────────────────
    print("Generating summary grid ...")
    cats_ordered = ["TP", "FP", "TN", "FN"]
    # Double column width (7.16 inches)
    fig, axes = plt.subplots(4, n_per_cat, figsize=(7.16, 8), dpi=300)
    fig.patch.set_facecolor("white")
    
    # User constraint: NO TITLE in the image

    for row_idx, cat in enumerate(cats_ordered):
        entries = buckets[cat]
        for col_idx in range(n_per_cat):
            ax = axes[row_idx, col_idx] if n_per_cat > 1 else axes[row_idx]
            ax.set_facecolor("white")
            ax.axis("off")

            if col_idx < len(entries):
                entry = entries[col_idx]
                true_label = entry["true"]
                pred_label = entry["pred"]
                all_prob   = entry["all_prob"]

                # Retrieve pre-computed overlay
                overlay_match = [img for img, c, i, t, p in grid_images
                                 if c == cat and i == col_idx + 1]
                if overlay_match:
                    ax.imshow(overlay_match[0])

                # IEEE Styling: Labels as subtitles per cell if needed, but keeping it clean
                # Removed detailed titles from individual axes for paper-ready look

            if col_idx == 0:
                ax.set_ylabel(CAT_DESC[cat], color='black',
                              fontsize=10, fontweight="bold", rotation=90, labelpad=4)
                # Show label on leftmost column
                ax.text(-50, 160, CAT_DESC[cat], color='black', fontsize=12, 
                        fontweight='bold', va='center', ha='right', rotation=90)

    fig.tight_layout(pad=0.2)
    grid_path = out_dir / "GradCAM_Interpretability_Grid.png"
    fig.savefig(str(grid_path), dpi=600, bbox_inches="tight")
    
    # Save EPS for vector requirement
    eps_path = out_dir / "GradCAM_Interpretability_Grid.eps"
    fig.savefig(str(eps_path), format='eps', dpi=300, bbox_inches="tight")
    
    plt.close(fig)

    print(f"\n✓  Paper-ready grid saved to: {out_dir}")
    print(f"   PNG (600 DPI): {grid_path}")
    print(f"   EPS: {eps_path}")


if __name__ == "__main__":
    main()
