import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_scripts.train import get_model, get_loaders

# ── IEEE "Premium" Refined Style Configuration ───────────────────────────────
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.6,
    'legend.fontsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'grid.color': '#CCCCCC',
    'figure.dpi': 200,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

BASE_DIR = r"C:\Open Source\leukiemea"
OUT_DIR = os.path.join(BASE_DIR, "ieee_figures", "final_three")
os.makedirs(OUT_DIR, exist_ok=True)

MNV3L_RUNS = [
    (1, os.path.join(BASE_DIR, "outputs", "local_mnv3l_final1")),
    (2, os.path.join(BASE_DIR, "outputs", "local_mnv3l_final2")),
    (3, os.path.join(BASE_DIR, "outputs", "local_mnv3l_final3")),
]

COLORS = {'mnv3l': '#3B7DD8', 'effb0': '#E07B39', 'black': '#333333'}

def load_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def plot_fig1_training_curves_v2():
    print("Generating Fig 1: Training Curves (2-Panel, Refined)...")
    # Using MNV3L Fold 2 (Median)
    fold2_csv = os.path.join(MNV3L_RUNS[1][1], "local_mnv3l_final2_fold2_20260315_173338_metrics.csv")
    df = load_metrics(fold2_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))
    
    # --- Panel A: Loss ---
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#666666', lw=1.2, linestyle='--')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color=COLORS['mnv3l'], lw=1.8)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontweight='bold')
    ax1.set_title('(a) Training and Validation Loss')
    ax1.legend(loc='upper right', frameon=True)
    ax1.minorticks_on()
    
    # --- Panel B: AUC ---
    ax2.plot(df['epoch'], df['auc'], label='Val AUC', color='#D62728', lw=1.8)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('ROC Area Under Curve', fontweight='bold')
    ax2.set_title('(b) Validation AUC-ROC')
    ax2.set_ylim(0.85, 1.0)
    ax2.legend(loc='lower right', frameon=True)
    ax2.minorticks_on()
    
    # Shared Unfreezing markers for both panels
    for ax in [ax1, ax2]:
        unfreeze_epochs = [11, 21, 41]
        labels = ['Head', 'Layer4', 'Full']
        for ep, lbl in zip(unfreeze_epochs, labels):
            ax.axvline(x=ep, color='gray', linestyle=':', alpha=0.5, lw=1)
            # Annotate only on ax1 to keep it clean, or both? Let's do both but smaller.
            ax.text(ep + 0.5, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
                    lbl, rotation=90, va='bottom', fontsize=8, color='gray')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig1_v2.png"))
    plt.close(fig)

def get_predictions_for_fold(fold, run_dir):
    print(f"  Evaluating Fold {fold}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import types
    args = types.SimpleNamespace(
        model = "mnv3l",
        fold = fold,
        splits_json = "cv_splits/cv_splits_3fold.json",
        res = 320,
        batch_size = 48,
        num_workers = 0
    )
    
    os.chdir(BASE_DIR)
    _, val_loader, _, _, _ = get_loaders(args)
    model, _, _, _, _ = get_model(args)
    
    pth_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
    ckpt_path = os.path.join(run_dir, pth_files[0])
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    all_targets = []
    all_scores_all = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            variants = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]

            avg_probs = torch.zeros(images.size(0), 2, device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                for v in variants:
                    logits = model(v)
                    avg_probs += torch.softmax(logits.float(), dim=1)

            avg_probs /= len(variants)
            
            all_targets.extend(labels.cpu().numpy())
            all_scores_all.extend(avg_probs[:, 0].cpu().numpy())

    y_true = 1 - np.array(all_targets)
    y_prob = np.array(all_scores_all)

    return y_true, y_prob

def plot_fig2_roc_v2():
    print("Generating Fig 2: Mean ROC Curve (Refined)...")
    all_true = []
    all_preds_prob = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold, run_dir in MNV3L_RUNS:
        y_true, y_prob = get_predictions_for_fold(fold, run_dir)
        all_true.extend(y_true)
        all_preds_prob.extend(y_prob)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
    all_true = np.array(all_true)
    all_preds_prob = np.array(all_preds_prob)
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    ax.plot(mean_fpr, mean_tpr, color=COLORS['mnv3l'], lw=2,
            label=f'Mean ROC (AUC={mean_auc:.3f})')
    
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), 
                    np.clip(mean_tpr + std_tpr, 0, 1), 
                    color=COLORS['mnv3l'], alpha=0.12, label='$\pm$ 1 std. dev.')
    
    # Mark threshold 0.35
    fpr_agg, tpr_agg, thresholds_agg = roc_curve(all_true, all_preds_prob)
    idx = np.argmin(np.abs(thresholds_agg - 0.35))
    op_fpr, op_tpr = fpr_agg[idx], tpr_agg[idx]
    
    ax.plot(op_fpr, op_tpr, marker='o', markersize=6, color='black', label=f'Threshold 0.35')
    ax.annotate(f'TPR={op_tpr:.3f}\nFPR={op_fpr:.3f}', xy=(op_fpr, op_tpr), xytext=(op_fpr+0.1, op_tpr-0.2),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.grid(True, linewidth=0.4, color='#DDDDDD')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig2_v2.png"))
    plt.close(fig)
    return all_true, all_preds_prob

def plot_fig3_cm_v2(all_true, all_preds_prob):
    print("Generating Fig 3: Confusion Matrix (Refined)...")
    y_pred_class = (all_preds_prob >= 0.35).astype(int)
    cm = confusion_matrix(all_true, y_pred_class, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    
    # "Greys" colors like in the good examples
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', 
                xticklabels=['HEM (Neg)', 'ALL (Pos)'], 
                yticklabels=['HEM (Neg)', 'ALL (Pos)'],
                annot_kws={"size": 12, "fontweight": "bold"}, 
                cbar=False, square=True,
                linewidths=1.0, linecolor='0.8', ax=ax)
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    # Use direction=in for ticks
    ax.tick_params(direction='in', length=0)
    
    # Calculate some summary stats to put in title for visual context
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    ax.set_title(f'Aggregated Results (t=0.35)\nAcc: {acc:.1%}, Sens: {sens:.1%}, Spec: {spec:.1%}', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig3_v2.png"))
    plt.close(fig)

if __name__ == "__main__":
    print("Regenerating 3 figures with REDESIGNED Visuals...")
    plot_fig1_training_curves_v2()
    y_t, y_p = plot_fig2_roc_v2()
    plot_fig3_cm_v2(y_t, y_p)
    print(f"Done! Results in {OUT_DIR}")
