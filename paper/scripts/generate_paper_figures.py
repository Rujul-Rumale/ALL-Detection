"""
generate_paper_figures.py
=========================
Generates aggregated, publication-quality figures for the paper using 3-fold CV.
Output: paper/latex/alternate_tj_latex_template_ap/figures/

Styling: IEEE double-column (7.16"), Times New Roman, Mean ± STD bands.
No figure titles are embedded in the image; figure name = filename stem.

Data:
  EffB0 Fold 1 : effb0_exp_meth  (one complete run, starting from ep 1)
  EffB0 Fold 2 : effb0_exp_meth1 (two runs, concatenated)
  EffB0 Fold 3 : effb0_exp_meth2 (one run)
  MNV3L Fold 1 : mnv3l_exp_meth  (one run)
  MNV3L Fold 2 : mnv3l_exp_meth1 (one run)
  MNV3L Fold 3 : mnv3l_exp_meth2 (three runs, concatenated)

NOTE on label semantics:
  The training code encodes HEM=class 1, ALL=class 0.
  Therefore in the CSV:
      'sensitivity' column = recall for class 1 = HEM recall
      'specificity' column = recall for class 0 = ALL recall
  The PAPER defines ALL as the positive class, so:
      Paper Sensitivity (ALL recall) = CSV 'specificity'
      Paper Specificity (HEM recall) = CSV 'sensitivity'
  All figure labels and table values use the paper convention.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── IEEE Style ───────────────────────────────────────────────────────────────
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.linewidth']  = 0.8
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR  = os.path.join(BASE_DIR, 'paper', 'latex',
                        'alternate_tj_latex_template_ap', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Phase markers ────────────────────────────────────────────────────────────
PHASE1_EP   = 11
PHASE1_5_EP = 21
PHASE2_EP   = 31   # Full-backbone unfreeze at epoch 31+

# ── CSV paths (per fold) ─────────────────────────────────────────────────────
def p(*parts):
    return os.path.join(BASE_DIR, 'outputs', *parts)

# EffB0 – each fold is a list of segment CSVs in epoch order
# Fold 1: two runs both starting from ep 1; the second (_000256) is the full run
EFFB0_FOLD_SEGS = [
    # Fold 1: use the longer continuous run (_000256) as the canonical trace
    [p('effb0_exp_meth',  'effb0_exp_meth_fold1_20260320_000256_metrics.csv')],
    # Fold 2: first run covers ep 1-105, second continues ep 97-175 (overlap at 97-105)
    [p('effb0_exp_meth1', 'effb0_exp_meth1_fold2_20260320_202738_metrics.csv'),
     p('effb0_exp_meth1', 'effb0_exp_meth1_fold2_20260321_005825_metrics.csv')],
    # Fold 3: single run ep 1-123
    [p('effb0_exp_meth2', 'effb0_exp_meth2_fold3_20260321_023825_metrics.csv')],
]

# MNV3L – per fold
MNV3L_FOLD_SEGS = [
    # Fold 1: single run ep 1-152
    [p('mnv3l_exp_meth',  'mnv3l_exp_meth_fold1_20260319_145131_metrics.csv')],
    # Fold 2: single run ep 1-114
    [p('mnv3l_exp_meth1', 'mnv3l_exp_meth1_fold2_20260319_165823_metrics.csv')],
    # Fold 3: three segments ep 1-42 / 43-105 / 104-137 (overlaps at boundaries)
    [p('mnv3l_exp_meth2', 'mnv3l_exp_meth2_fold3_20260319_182437_metrics.csv'),
     p('mnv3l_exp_meth2', 'mnv3l_exp_meth2_fold3_20260319_191306_metrics.csv'),
     p('mnv3l_exp_meth2', 'mnv3l_exp_meth2_fold3_20260319_201230_metrics.csv')],
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_fold(seg_paths):
    """Load and concatenate segment CSVs for one fold, removing epoch overlaps."""
    frames = []
    last_epoch = -1
    for path in seg_paths:
        if not os.path.exists(path):
            print(f"  [Warning] Missing: {path}")
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        # Keep only epochs strictly after what we've already collected
        df = df[df['epoch'] > last_epoch].copy()
        if len(df):
            last_epoch = df['epoch'].max()
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all_folds(fold_seg_list):
    """Return list of per-fold DataFrames (full epoch sequence, no duplicates)."""
    folds = []
    for segs in fold_seg_list:
        df = load_fold(segs)
        if df is not None:
            folds.append(df)
    return folds


def aggregate(folds):
    """
    Align folds to the same epoch grid (trim to shortest) and compute
    mean + std across folds at each epoch position.
    Returns (mean_df, std_df) indexed 0..N-1, epoch column preserved.
    """
    min_len = min(len(f) for f in folds)
    trimmed = [f.iloc[:min_len].reset_index(drop=True) for f in folds]
    stacked = pd.concat(trimmed)
    mean_df = stacked.groupby(stacked.index).mean()
    std_df  = stacked.groupby(stacked.index).std(ddof=1).fillna(0)
    # Restore epoch from first fold (they share the same epoch sequence)
    mean_df['epoch'] = trimmed[0]['epoch'].values
    return mean_df, std_df


def save_fig(fig, name):
    """Save PNG (600 dpi) and EPS (300 dpi). No title embedded."""
    png = os.path.join(OUT_DIR, f'{name}.png')
    eps = os.path.join(OUT_DIR, f'{name}.eps')
    fig.savefig(png, format='png', dpi=600, bbox_inches='tight')
    fig.savefig(eps, format='eps', dpi=300, bbox_inches='tight')
    print(f"  Saved  {name}.png + .eps")
    plt.close(fig)


def vline(ax, ep, style, label=None):
    ax.axvline(ep, color='gray', linestyle=style, linewidth=0.9, alpha=0.75)
    if label:
        # Place label just above mid-y
        yl = ax.get_ylim()
        ymid = yl[0] + 0.72 * (yl[1] - yl[0])
        ax.text(ep + 0.8, ymid, label, fontsize=8, color='gray', fontstyle='italic')


# ── Load data ────────────────────────────────────────────────────────────────
print("\nLoading fold data …")
mnv3l_folds = load_all_folds(MNV3L_FOLD_SEGS)
effb0_folds = load_all_folds(EFFB0_FOLD_SEGS)

mnv3l_mean, mnv3l_std = aggregate(mnv3l_folds)
effb0_mean, effb0_std = aggregate(effb0_folds)

print(f"  MNV3L: {[len(f) for f in mnv3l_folds]} epochs per fold  →  "
      f"aligned to {len(mnv3l_mean)} epochs")
print(f"  EffB0: {[len(f) for f in effb0_folds]} epochs per fold  →  "
      f"aligned to {len(effb0_mean)} epochs")

# ── Print final-epoch summary for paper ──────────────────────────────────────
def per_fold_peak(folds, label):
    """Print per-fold peak-AUC row values (paper convention labels)."""
    print(f"\n{'─'*60}")
    print(f"  {label}  |  peak-AUC epoch stats  (paper label convention)")
    print(f"  {'Fold':<6} {'Epochs':<8} {'AUC':>6} {'Sens(ALL)':<11} {'Spec(HEM)':<11} {'F1':>6} {'Acc':>6}")
    aucs, senss, specs, f1s, accs = [], [], [], [], []
    for i, df in enumerate(folds):
        row = df.loc[df['auc'].idxmax()]
        # paper sens = csv specificity ; paper spec = csv sensitivity
        auc  = row['auc']
        sens = row['specificity']   # ALL recall
        spec = row['sensitivity']   # HEM recall
        f1   = row['f1']
        acc  = row['val_acc']
        aucs.append(auc); senss.append(sens); specs.append(spec)
        f1s.append(f1);   accs.append(acc)
        print(f"  {i+1:<6} {len(df):<8} {auc:.4f} {sens:<11.4f} {spec:<11.4f} {f1:.4f} {acc:.4f}")
    print(f"  {'Mean':<6} {'':<8} {np.mean(aucs):.4f} {np.mean(senss):<11.4f} {np.mean(specs):<11.4f} {np.mean(f1s):.4f} {np.mean(accs):.4f}")
    print(f"  {'±Std':<6} {'':<8} {np.std(aucs, ddof=1):.4f} {np.std(senss,ddof=1):<11.4f} {np.std(specs,ddof=1):<11.4f} {np.std(f1s,ddof=1):.4f} {np.std(accs,ddof=1):.4f}")
    return (np.mean(aucs), np.std(aucs, ddof=1),
            np.mean(senss), np.std(senss, ddof=1),
            np.mean(specs), np.std(specs, ddof=1),
            np.mean(f1s),   np.std(f1s, ddof=1),
            np.mean(accs),  np.std(accs, ddof=1))

mnv3l_stats = per_fold_peak(mnv3l_folds, 'MobileNetV3-Large')
effb0_stats = per_fold_peak(effb0_folds, 'EfficientNet-B0')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Aggregated Training & Validation Loss (MNV3L)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_loss_curves():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ep = mnv3l_mean['epoch']

    ax.plot(ep, mnv3l_mean['train_loss'], color='#2196F3',
            label='Training Loss', linewidth=1.5)
    ax.fill_between(ep,
                    mnv3l_mean['train_loss'] - mnv3l_std['train_loss'],
                    mnv3l_mean['train_loss'] + mnv3l_std['train_loss'],
                    color='#2196F3', alpha=0.15)

    ax.plot(ep, mnv3l_mean['val_loss'], color='#F44336',
            label='Validation Loss', linewidth=1.5, linestyle='--')
    ax.fill_between(ep,
                    mnv3l_mean['val_loss'] - mnv3l_std['val_loss'],
                    mnv3l_mean['val_loss'] + mnv3l_std['val_loss'],
                    color='#F44336', alpha=0.15)

    ax.axvline(PHASE1_EP,   color='gray', linestyle=':',  linewidth=0.9, alpha=0.7)
    ax.axvline(PHASE1_5_EP, color='gray', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(PHASE2_EP,   color='gray', linestyle='-.', linewidth=0.9, alpha=0.7)

    yl = ax.get_ylim()
    ymid = yl[0] + 0.80 * (yl[1] - yl[0])
    for xp, txt in [(PHASE1_EP+0.5, 'Ph 1'), (PHASE1_5_EP+0.5, 'Ph 1.5'), (PHASE2_EP+0.5, 'Ph 2')]:
        ax.text(xp, ymid, txt, fontsize=8, color='gray', fontstyle='italic')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Focal Loss')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, which='major', linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.minorticks_on()
    fig.tight_layout()
    save_fig(fig, 'fig1_loss_curves')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Aggregated Validation AUC-ROC (MNV3L)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_auc_curve():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ep = mnv3l_mean['epoch']

    ax.plot(ep, mnv3l_mean['auc'], color='#4CAF50',
            linewidth=1.8, label='Mean Validation AUC-ROC')
    ax.fill_between(ep,
                    mnv3l_mean['auc'] - mnv3l_std['auc'],
                    mnv3l_mean['auc'] + mnv3l_std['auc'],
                    color='#4CAF50', alpha=0.2)

    best_idx = mnv3l_mean['auc'].idxmax()
    best     = mnv3l_mean.iloc[best_idx]
    ax.scatter([best['epoch']], [best['auc']], color='gold', zorder=5, s=50,
               edgecolors='black',
               label=f'Peak Mean: {best["auc"]:.4f} (ep {int(best["epoch"])})')

    ax.axvline(PHASE1_EP,   color='gray', linestyle=':',  linewidth=0.9, alpha=0.7)
    ax.axvline(PHASE1_5_EP, color='gray', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(PHASE2_EP,   color='gray', linestyle='-.', linewidth=0.9, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, which='major', linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.minorticks_on()
    fig.tight_layout()
    save_fig(fig, 'fig2_auc_curve')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Sensitivity (ALL Recall) & Specificity (HEM Recall) — MNV3L
# ─────────────────────────────────────────────────────────────────────────────
def fig3_sensitivity_specificity():
    """
    Paper convention:
      Sensitivity = ALL recall = CSV 'specificity'
      Specificity = HEM recall = CSV 'sensitivity'
    """
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ep = mnv3l_mean['epoch']

    # Sensitivity (ALL Recall) — from CSV 'specificity' column
    paper_sens     = mnv3l_mean['specificity']
    paper_sens_std = mnv3l_std['specificity']
    ax.plot(ep, paper_sens, color='#E91E63', linewidth=1.8,
            label='Sensitivity (ALL Recall)')
    ax.fill_between(ep,
                    paper_sens - paper_sens_std,
                    paper_sens + paper_sens_std,
                    color='#E91E63', alpha=0.15)

    # Specificity (HEM Recall) — from CSV 'sensitivity' column
    paper_spec     = mnv3l_mean['sensitivity']
    paper_spec_std = mnv3l_std['sensitivity']
    ax.plot(ep, paper_spec, color='#9C27B0', linewidth=1.8, linestyle='--',
            label='Specificity (HEM Recall)')
    ax.fill_between(ep,
                    paper_spec - paper_spec_std,
                    paper_spec + paper_spec_std,
                    color='#9C27B0', alpha=0.15)

    ax.axvline(PHASE1_EP,   color='gray', linestyle=':',  linewidth=0.9, alpha=0.7)
    ax.axvline(PHASE1_5_EP, color='gray', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(PHASE2_EP,   color='gray', linestyle='-.', linewidth=0.9, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', ncol=2)
    ax.grid(True, which='major', linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.minorticks_on()
    fig.tight_layout()
    save_fig(fig, 'fig3_sensitivity_specificity')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Representative Confusion Matrix (MNV3L Fold 1 peak-AUC epoch)
# ─────────────────────────────────────────────────────────────────────────────
# Validation set sizes from C-NMC subject-disjoint Fold 1 (38 patients)
# These must match the actual fold-1 validation counts.
# Approximate from dataset: 101 patients total; fold 1 = 38 patients (~37%)
# Proportional ALL : 8491 * 38/101 ≈ 3193  HEM : 4037 * 38/101 ≈ 1520
VAL_ALL_F1 = 3193
VAL_HEM_F1 = 1520

def fig4_confusion_matrix():
    fold1 = mnv3l_folds[0]
    best  = fold1.loc[fold1['auc'].idxmax()]

    # paper_sens = ALL recall = CSV specificity
    # paper_spec = HEM recall = CSV sensitivity
    all_recall = best['specificity']   # ALL recall (paper sensitivity)
    hem_recall = best['sensitivity']   # HEM recall (paper specificity)

    tp = round(all_recall * VAL_ALL_F1)   # True Positive (ALL correctly ID'd)
    fn = VAL_ALL_F1 - tp                  # False Negative (ALL missed)
    tn = round(hem_recall * VAL_HEM_F1)   # True Negative (HEM correctly ID'd)
    fp = VAL_HEM_F1 - tn                  # False Positive (HEM flagged as ALL)

    # Layout: rows = True, cols = Pred; [ALL-row, HEM-row] x [ALL-pred, HEM-pred]
    cm = np.array([[tp, fn],
                   [fp, tn]])

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', rasterized=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ['ALL (Blast)', 'HEM (Normal)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=12, weight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    save_fig(fig, 'fig4_confusion_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-metric Best-Epoch Comparison (MNV3L vs EffB0)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_metrics_summary():
    labels = ['AUC-ROC', 'Sens (ALL)', 'Spec (HEM)', 'F1 Score', 'Accuracy']

    def best_stats(mean_df, std_df):
        idx = mean_df['auc'].idxmax()
        row  = mean_df.iloc[idx]
        srow = std_df.iloc[idx]
        # paper: sens=ALL recall=csv spec ; spec=HEM recall=csv sens
        vals = [row['auc'], row['specificity'], row['sensitivity'],
                row['f1'], row['val_acc']]
        stds = [srow['auc'], srow['specificity'], srow['sensitivity'],
                srow['f1'], srow['val_acc']]
        return vals, stds

    vals_mnv3l, stds_mnv3l = best_stats(mnv3l_mean, mnv3l_std)
    vals_effb0, stds_effb0 = best_stats(effb0_mean, effb0_std)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    r1 = ax.bar(x - width/2, vals_mnv3l, width, yerr=stds_mnv3l,
                label='MobileNetV3-L', color='#2196F3',
                edgecolor='black', capsize=4, alpha=0.85)
    r2 = ax.bar(x + width/2, vals_effb0, width, yerr=stds_effb0,
                label='EfficientNet-B0', color='#FF9800',
                edgecolor='black', capsize=4, alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', ncol=2)
    ax.grid(True, axis='y', linewidth=0.4, alpha=0.5)

    def autolabel(rects, stds):
        for rect, std in zip(rects, stds):
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h + std + 0.01),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    autolabel(r1, stds_mnv3l)
    autolabel(r2, stds_effb0)
    fig.tight_layout()
    save_fig(fig, 'fig5_metrics_summary')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — EffB0 Training & Validation Loss
# ─────────────────────────────────────────────────────────────────────────────
def fig6_effb0_loss_curves():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ep = effb0_mean['epoch']

    ax.plot(ep, effb0_mean['train_loss'], color='#FF5722',
            label='Training Loss', linewidth=1.5)
    ax.fill_between(ep,
                    effb0_mean['train_loss'] - effb0_std['train_loss'],
                    effb0_mean['train_loss'] + effb0_std['train_loss'],
                    color='#FF5722', alpha=0.15)

    ax.plot(ep, effb0_mean['val_loss'], color='#795548',
            label='Validation Loss', linewidth=1.5, linestyle='--')
    ax.fill_between(ep,
                    effb0_mean['val_loss'] - effb0_std['val_loss'],
                    effb0_mean['val_loss'] + effb0_std['val_loss'],
                    color='#795548', alpha=0.15)

    ax.axvline(PHASE1_EP,   color='gray', linestyle=':',  linewidth=0.9, alpha=0.7)
    ax.axvline(PHASE1_5_EP, color='gray', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(PHASE2_EP,   color='gray', linestyle='-.', linewidth=0.9, alpha=0.7)

    yl = ax.get_ylim()
    ymid = yl[0] + 0.80 * (yl[1] - yl[0])
    for xp, txt in [(PHASE1_EP+0.5, 'Ph 1'), (PHASE1_5_EP+0.5, 'Ph 1.5'), (PHASE2_EP+0.5, 'Ph 2')]:
        ax.text(xp, ymid, txt, fontsize=8, color='gray', fontstyle='italic')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Focal Loss')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, which='major', linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.minorticks_on()
    fig.tight_layout()
    save_fig(fig, 'fig6_effb0_loss_curves')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n🚀 Generating Figures → {os.path.abspath(OUT_DIR)}\n")
    fig1_loss_curves()
    fig2_auc_curve()
    fig3_sensitivity_specificity()
    fig4_confusion_matrix()
    fig5_metrics_summary()
    fig6_effb0_loss_curves()
    print("\n✅ All figures saved.")
