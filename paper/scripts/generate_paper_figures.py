"""
generate_paper_figures.py
=========================
Generates all Phase 1 publication-quality figures for the paper.
Output: IEEE_Conference_Template/figures/

Figures:
  fig1_loss_curves.{png,eps}
  fig2_auc_curve.{png,eps}
  fig3_sensitivity_specificity.{png,eps}
  fig4_confusion_matrix.{png,eps}
  fig5_metrics_summary.{png,eps}

Styling follows figure.py (IEEE double-column format):
  - Font: Times New Roman, 12pt
  - Width: 7.16 inches (double-column)
  - No internal titles (filename is the title)
  - Ticks: direction='in', minorticks_on
  - Exports: PNG @ 600 DPI, EPS @ 300 DPI
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── IEEE Style ──────────────────────────────────────────────────────────────
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.linewidth']  = 0.8
plt.rcParams['lines.linewidth'] = 1.8

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'orig_benchmark',
    'orig_benchmark_fold1_20260309_140336_metrics.csv'
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'IEEE_Conference_Template', 'figures'
)
os.makedirs(OUT_DIR, exist_ok=True)

# Validation set class sizes (Fold 1, from training log)
VAL_ALL = 3125   # positive (ALL blasts)
VAL_HEM = 1339   # negative (healthy HEM)

# Best epoch (peak AUC)
BEST_EPOCH = 96
OPT_THRESHOLD = 0.75

# Phase-transition epochs
PHASE1_EP = 6
PHASE2_EP = 31

# ── Helper: Save ─────────────────────────────────────────────────────────────
def save_fig(fig, name):
    png = os.path.join(OUT_DIR, f"{name}.png")
    eps = os.path.join(OUT_DIR, f"{name}.eps")
    fig.savefig(png, format='png', dpi=600, bbox_inches='tight')
    fig.savefig(eps, format='eps', dpi=300, bbox_inches='tight')
    print(f"  Saved {name}.png + .eps")
    plt.close(fig)


# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
epochs = df['epoch'].values

best_row = df[df['epoch'] == BEST_EPOCH].iloc[0]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Training & Validation Loss Curves
# ─────────────────────────────────────────────────────────────────────────────
def fig1_loss_curves():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    ax.plot(epochs, df['train_loss'], color='#2196F3', label='Training Loss',   linewidth=1.8)
    ax.plot(epochs, df['val_loss'],   color='#F44336', label='Validation Loss', linewidth=1.8, linestyle='--')

    ax.axvline(PHASE1_EP,  color='gray', linestyle=':', linewidth=1.0, label=f'Phase 1 (ep {PHASE1_EP})')
    ax.axvline(PHASE2_EP,  color='gray', linestyle='-.', linewidth=1.0, label=f'Phase 2 (ep {PHASE2_EP})')
    ax.axvline(BEST_EPOCH, color='gold', linestyle='--', linewidth=1.4, label=f'Best epoch ({BEST_EPOCH})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9, loc='upper right', framealpha=1.0)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'fig1_loss_curves')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Validation AUC-ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
def fig2_auc_curve():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    ax.plot(epochs, df['auc'], color='#4CAF50', linewidth=1.8, label='Validation AUC-ROC')
    ax.scatter([BEST_EPOCH], [best_row['auc']], color='gold', zorder=5, s=60,
               label=f'Best: {best_row["auc"]:.4f} (ep {BEST_EPOCH})')

    ax.axvline(PHASE1_EP,  color='gray', linestyle=':',  linewidth=1.0, label=f'Phase 1 (ep {PHASE1_EP})')
    ax.axvline(PHASE2_EP,  color='gray', linestyle='-.', linewidth=1.0, label=f'Phase 2 (ep {PHASE2_EP})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.88, 1.0)
    ax.legend(fontsize=9, loc='lower right', framealpha=1.0)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'fig2_auc_curve')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Sensitivity & Specificity
# ─────────────────────────────────────────────────────────────────────────────
def fig3_sensitivity_specificity():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    ax.plot(epochs, df['sensitivity'], color='#E91E63', linewidth=1.8, label='Sensitivity (Recall)')
    ax.plot(epochs, df['specificity'], color='#9C27B0', linewidth=1.8, linestyle='--', label='Specificity')

    ax.axvline(PHASE1_EP,  color='gray', linestyle=':',  linewidth=1.0, label=f'Phase 1 (ep {PHASE1_EP})')
    ax.axvline(PHASE2_EP,  color='gray', linestyle='-.', linewidth=1.0, label=f'Phase 2 (ep {PHASE2_EP})')
    ax.axvline(BEST_EPOCH, color='gold', linestyle='--', linewidth=1.4, label=f'Best epoch ({BEST_EPOCH})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9, loc='lower right', framealpha=1.0)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'fig3_sensitivity_specificity')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Confusion Matrix (best epoch, optimal threshold)
# Derived analytically from confirmed validation set class sizes
# ─────────────────────────────────────────────────────────────────────────────
def fig4_confusion_matrix():
    sens = best_row['sensitivity']   # = TP / (TP+FN)
    spec = best_row['specificity']   # = TN / (TN+FP)

    tp = round(sens * VAL_ALL)
    fn = VAL_ALL - tp
    tn = round(spec * VAL_HEM)
    fp = VAL_HEM - tn
    cm = np.array([[tp, fn],
                   [fp, tn]])

    fig, ax = plt.subplots(figsize=(4.0, 3.4))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', rasterized=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ['ALL (Blast)', 'HEM (Normal)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.tick_params(direction='in', which='both')
    fig.tight_layout()
    save_fig(fig, 'fig4_confusion_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Metrics Summary Bar Chart (best epoch)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_metrics_summary():
    labels  = ['AUC-ROC', 'Sensitivity', 'Specificity', 'F1 Score', 'Accuracy']
    values  = [
        best_row['auc'],
        best_row['sensitivity'],
        best_row['specificity'],
        best_row['f1'],
        best_row['val_acc'],
    ]
    colors  = ['#2196F3', '#E91E63', '#9C27B0', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='black', linewidth=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.008,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis='y', linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'fig5_metrics_summary')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Generating Phase 1 paper figures -> {os.path.abspath(OUT_DIR)}\n")
    fig1_loss_curves()
    fig2_auc_curve()
    fig3_sensitivity_specificity()
    fig4_confusion_matrix()
    fig5_metrics_summary()
    print("\nDone. All figures saved.")
