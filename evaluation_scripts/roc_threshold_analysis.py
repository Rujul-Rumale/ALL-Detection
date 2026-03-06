"""
ROC Curve & Threshold Analysis for Paper
=========================================
Runs TFLite models on the full C-NMC validation set,
collects raw probabilities, and generates:
  1. ROC curve with AUC
  2. Sensitivity/Specificity vs. Threshold plot
  3. Confusion matrices at key thresholds (0.50, Youden, 0.85)
  4. Summary table for paper inclusion
  5. Precision-Recall curve

Usage:
  python evaluation_scripts/roc_threshold_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

# ============ CONFIG ============
VAL_DIR = r"c:\Open Source\leukiemea\cnmc_staging\val"
OUTPUT_DIR = r"c:\Open Source\leukiemea\evaluation_outputs\roc_analysis"

MODELS = {
    "MobileNetV3-Large v2 (Balanced)": r"c:\Open Source\leukiemea\models\mobilenetv3_large_v2.tflite",
    "MobileNetV3-Large (Weighted)": r"c:\Open Source\leukiemea\models\mobilenetv3_large_cnmc.tflite",
    "MobileNetV3-Small (Weighted)": r"c:\Open Source\leukiemea\models\mobilenetv3_cnmc.tflite",
}

IMG_SIZE = 224
# ImageNet normalization (matches training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class mapping (from training scripts)
CLASS_TO_IDX = {'all': 0, 'hem': 1}

# Thresholds to evaluate in detail
DETAIL_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ TFLite INFERENCE ============
def load_tflite_model(model_path):
    """Load TFLite model (tries tflite_runtime first, falls back to TF)."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite_module
        tflite = tflite_module

    if hasattr(tflite, 'Interpreter'):
        interpreter = tflite.Interpreter(model_path=model_path)
    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_image(img_path):
    """Load and preprocess a single image for TFLite inference."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # Channel-first (NCHW) for PyTorch-exported models
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def run_inference(interpreter, input_details, output_details, img_tensor):
    """Run TFLite inference, return softmax probabilities."""
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])
    # Apply softmax
    exp_out = np.exp(raw_output - np.max(raw_output))
    probs = exp_out / np.sum(exp_out)
    return probs[0]  # [P(ALL), P(HEM)]


def run_inference_tta(interpreter, input_details, output_details, img_path):
    """Run inference with 4-orientation TTA (matches demo_pipeline.py)."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 4 TTA variants
    variants = [
        img,
        np.fliplr(img),
        np.flipud(img),
        np.flipud(np.fliplr(img)),
    ]

    all_probs = []
    for variant in variants:
        v_norm = variant.astype(np.float32) / 255.0
        v_norm = (v_norm - MEAN) / STD
        v_trans = np.transpose(v_norm, (2, 0, 1))
        v_tensor = np.expand_dims(v_trans, axis=0).astype(np.float32)
        probs = run_inference(interpreter, input_details, output_details, v_tensor)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


# ============ EVALUATION ============
def collect_predictions(model_path, val_dir, use_tta=False):
    """Run model on entire validation set, collect ALL probabilities and true labels."""
    interpreter, input_details, output_details = load_tflite_model(model_path)

    all_dir = os.path.join(val_dir, "all")
    hem_dir = os.path.join(val_dir, "hem")

    true_labels = []  # 1 = ALL, 0 = HEM
    all_probs = []    # P(ALL) for each sample

    # Process ALL images
    all_images = sorted(Path(all_dir).glob("*"))
    print(f"  Processing {len(all_images)} ALL images...")
    for img_path in tqdm(all_images, desc="  ALL", leave=False):
        if use_tta:
            probs = run_inference_tta(interpreter, input_details, output_details, str(img_path))
        else:
            img_tensor = preprocess_image(str(img_path))
            if img_tensor is None:
                continue
            probs = run_inference(interpreter, input_details, output_details, img_tensor)

        if probs is None:
            continue
        all_probs.append(float(probs[CLASS_TO_IDX['all']]))  # P(ALL)
        true_labels.append(1)

    # Process HEM images
    hem_images = sorted(Path(hem_dir).glob("*"))
    print(f"  Processing {len(hem_images)} HEM images...")
    for img_path in tqdm(hem_images, desc="  HEM", leave=False):
        if use_tta:
            probs = run_inference_tta(interpreter, input_details, output_details, str(img_path))
        else:
            img_tensor = preprocess_image(str(img_path))
            if img_tensor is None:
                continue
            probs = run_inference(interpreter, input_details, output_details, img_tensor)

        if probs is None:
            continue
        all_probs.append(float(probs[CLASS_TO_IDX['all']]))
        true_labels.append(0)

    return np.array(true_labels), np.array(all_probs)


def compute_metrics_at_threshold(true_labels, all_probs, threshold):
    """Compute classification metrics at a given ALL probability threshold."""
    preds = (all_probs >= threshold).astype(int)

    tp = np.sum((preds == 1) & (true_labels == 1))
    fp = np.sum((preds == 1) & (true_labels == 0))
    fn = np.sum((preds == 0) & (true_labels == 1))
    tn = np.sum((preds == 0) & (true_labels == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    youden = sensitivity + specificity - 1.0

    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'youden': youden,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def compute_roc(true_labels, all_probs, n_points=200):
    """Compute ROC curve points."""
    thresholds = np.linspace(0, 1, n_points)
    tpr_list = []
    fpr_list = []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (true_labels == 1))
        fp = np.sum((preds == 1) & (true_labels == 0))
        fn = np.sum((preds == 0) & (true_labels == 1))
        tn = np.sum((preds == 0) & (true_labels == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute AUC via trapezoidal rule (sort by FPR ascending)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]
    auc = np.trapz(tpr_sorted, fpr_sorted)

    return fpr_arr, tpr_arr, thresholds, auc


# ============ PLOTTING ============
def plot_roc_curve(fpr, tpr, auc, model_name, output_path):
    """Generate publication-quality ROC curve."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    ax.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved ROC curve: {output_path}")


def plot_threshold_sweep(true_labels, all_probs, model_name, output_path):
    """Plot sensitivity and specificity vs. threshold."""
    thresholds = np.linspace(0, 1, 200)
    sensitivities = []
    specificities = []
    f1_scores = []

    for t in thresholds:
        m = compute_metrics_at_threshold(true_labels, all_probs, t)
        sensitivities.append(m['sensitivity'])
        specificities.append(m['specificity'])
        f1_scores.append(m['f1'])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)

    ax.plot(thresholds, sensitivities, color='#dc2626', linewidth=2, label='Sensitivity (Recall)')
    ax.plot(thresholds, specificities, color='#2563eb', linewidth=2, label='Specificity')
    ax.plot(thresholds, f1_scores, color='#16a34a', linewidth=2, linestyle='--', label='F1 Score')

    # Mark key thresholds
    for t_mark, color, label in [(0.50, '#f59e0b', 't=0.50'), (0.85, '#7c3aed', 't=0.85')]:
        m = compute_metrics_at_threshold(true_labels, all_probs, t_mark)
        ax.axvline(x=t_mark, color=color, linestyle=':', alpha=0.7)
        ax.annotate(f'{label}\nSens={m["sensitivity"]:.3f}\nSpec={m["specificity"]:.3f}',
                    xy=(t_mark, 0.5), fontsize=8, color=color, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

    # Youden optimal
    youdens = np.array(sensitivities) + np.array(specificities) - 1.0
    best_idx = np.argmax(youdens)
    best_t = thresholds[best_idx]
    ax.axvline(x=best_t, color='#059669', linewidth=2, linestyle='-.')
    m_best = compute_metrics_at_threshold(true_labels, all_probs, best_t)
    ax.annotate(f'Youden t={best_t:.2f}\nSens={m_best["sensitivity"]:.3f}\nSpec={m_best["specificity"]:.3f}',
                xy=(best_t, 0.3), fontsize=8, color='#059669', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#059669', alpha=0.8))

    ax.set_xlabel('ALL Probability Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(f'Threshold Analysis — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved threshold sweep: {output_path}")


def plot_confusion_matrix(metrics, model_name, output_path):
    """Plot confusion matrix for a given threshold."""
    cm = np.array([[metrics['tp'], metrics['fn']],
                    [metrics['fp'], metrics['tn']]])
    labels = ['ALL (Blast)', 'HEM (Healthy)']

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred ALL', 'Pred HEM'])
    ax.set_yticklabels(['True ALL', 'True HEM'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    t = metrics['threshold']
    ax.set_title(f'Confusion Matrix (t={t:.2f}) — {model_name}', fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall(true_labels, all_probs, model_name, output_path):
    """Plot Precision-Recall curve."""
    thresholds = np.linspace(0, 1, 200)
    precisions = []
    recalls = []

    for t in thresholds:
        m = compute_metrics_at_threshold(true_labels, all_probs, t)
        precisions.append(m['precision'])
        recalls.append(m['sensitivity'])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    ax.plot(recalls, precisions, color='#7c3aed', linewidth=2.5)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved PR curve: {output_path}")


# ============ MAIN ============
def analyze_model(model_name, model_path, val_dir, output_dir, use_tta=False):
    """Run complete analysis for one model."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
    tta_suffix = "_tta" if use_tta else ""

    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}" + (" [with TTA]" if use_tta else ""))
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    # 1. Collect predictions
    true_labels, all_probs = collect_predictions(model_path, val_dir, use_tta=use_tta)
    print(f"  Total samples: {len(true_labels)} (ALL: {np.sum(true_labels)}, HEM: {np.sum(true_labels == 0)})")

    # 2. ROC curve
    fpr, tpr, roc_thresholds, auc = compute_roc(true_labels, all_probs)
    print(f"  AUC: {auc:.4f}")
    plot_roc_curve(fpr, tpr, auc, model_name + (" + TTA" if use_tta else ""),
                   os.path.join(output_dir, f"roc_{safe_name}{tta_suffix}.png"))

    # 3. Threshold sweep
    plot_threshold_sweep(true_labels, all_probs, model_name + (" + TTA" if use_tta else ""),
                         os.path.join(output_dir, f"threshold_{safe_name}{tta_suffix}.png"))

    # 4. Precision-Recall curve
    plot_precision_recall(true_labels, all_probs, model_name + (" + TTA" if use_tta else ""),
                          os.path.join(output_dir, f"pr_{safe_name}{tta_suffix}.png"))

    # 5. Metrics at detail thresholds
    print(f"\n  {'Threshold':<10} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'Acc':>8} {'Youden':>8}")
    print(f"  {'-'*58}")

    results = []
    best_youden = -1
    best_youden_threshold = 0.5

    for t in DETAIL_THRESHOLDS:
        m = compute_metrics_at_threshold(true_labels, all_probs, t)
        results.append(m)
        print(f"  {t:<10.2f} {m['sensitivity']:>8.4f} {m['specificity']:>8.4f} "
              f"{m['precision']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>8.4f} {m['youden']:>8.4f}")

        if m['youden'] > best_youden:
            best_youden = m['youden']
            best_youden_threshold = t

    # Find continuous Youden optimal
    fine_thresholds = np.linspace(0, 1, 1000)
    youdens = []
    for t in fine_thresholds:
        m = compute_metrics_at_threshold(true_labels, all_probs, t)
        youdens.append(m['youden'])
    optimal_t = fine_thresholds[np.argmax(youdens)]
    optimal_m = compute_metrics_at_threshold(true_labels, all_probs, optimal_t)

    print(f"\n  ★ Youden Optimal Threshold: {optimal_t:.3f}")
    print(f"    Sensitivity: {optimal_m['sensitivity']:.4f}")
    print(f"    Specificity: {optimal_m['specificity']:.4f}")
    print(f"    F1 Score:    {optimal_m['f1']:.4f}")
    print(f"    Accuracy:    {optimal_m['accuracy']:.4f}")
    print(f"    Youden's J:  {optimal_m['youden']:.4f}")

    # 6. Confusion matrices at key thresholds
    for t in [0.50, optimal_t, 0.85]:
        m = compute_metrics_at_threshold(true_labels, all_probs, t)
        t_label = f"{t:.2f}" if t != optimal_t else f"{t:.3f}_youden"
        plot_confusion_matrix(m, model_name,
                             os.path.join(output_dir, f"cm_{safe_name}{tta_suffix}_t{t_label}.png"))

    # 7. Save CSV
    csv_path = os.path.join(output_dir, f"results_{safe_name}{tta_suffix}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        # Add Youden optimal
        writer.writerow(optimal_m)
    print(f"  Saved CSV: {csv_path}")

    return {
        'model': model_name,
        'auc': auc,
        'optimal_threshold': optimal_t,
        'optimal_metrics': optimal_m,
        'all_results': results,
    }


def main():
    print("=" * 60)
    print("  ROC & Threshold Analysis for IEEE Paper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_summaries = []

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  SKIP: {model_path} not found")
            continue

        # Without TTA
        summary = analyze_model(model_name, model_path, VAL_DIR, OUTPUT_DIR, use_tta=False)
        all_summaries.append(summary)

        # With TTA (for the main model only — takes 4x longer)
        if "Large v2" in model_name:
            summary_tta = analyze_model(model_name, model_path, VAL_DIR, OUTPUT_DIR, use_tta=True)
            all_summaries.append(summary_tta)

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for s in all_summaries:
        om = s['optimal_metrics']
        print(f"\n  {s['model']}:")
        print(f"    AUC: {s['auc']:.4f}")
        print(f"    Youden Optimal: t={s['optimal_threshold']:.3f} → "
              f"Sens={om['sensitivity']:.4f}, Spec={om['specificity']:.4f}, F1={om['f1']:.4f}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
