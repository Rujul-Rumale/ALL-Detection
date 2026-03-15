import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── IEEE Style ──────────────────────────────────────────────────────────────
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.linewidth']  = 0.8
plt.rcParams['lines.linewidth'] = 1.8

OUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'IEEE_Conference_Template', 'figures'
)
os.makedirs(OUT_DIR, exist_ok=True)

# Validation set class sizes (Fold 1)
VAL_ALL = 3125
VAL_HEM = 1339

def save_fig(fig, name):
    png = os.path.join(OUT_DIR, f"{name}.png")
    eps = os.path.join(OUT_DIR, f"{name}.eps")
    fig.savefig(png, format='png', dpi=600, bbox_inches='tight')
    fig.savefig(eps, format='eps', dpi=300, bbox_inches='tight')
    print(f"  Saved {name}.png + .eps")
    plt.close(fig)

def get_latest_metrics(run_dir_name):
    # Find the latest metrics csv in outputs/{run_dir_name}
    search_path = os.path.join(
        os.path.dirname(__file__), '..', 'outputs', run_dir_name, '*_metrics.csv'
    )
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No metrics CSV found in {search_path}")
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading {latest_file}...")
    df = pd.read_csv(latest_file)
    df.columns = df.columns.str.strip()
    return df

def fig1_comparison_loss_curves(df_base, df_hyb):
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    # Baseline Losses
    ax.plot(df_base['epoch'], df_base['val_loss'], color='#F44336', label='Baseline Val Loss', linewidth=1.8, linestyle='--')
    
    # Hybrid Losses
    ax.plot(df_hyb['epoch'], df_hyb['val_loss'], color='#2196F3', label='Hybrid Val Loss', linewidth=1.8, linestyle='-')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.legend(fontsize=9, loc='upper right', framealpha=1.0)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'comparison_loss_curves')


def fig2_comparison_auc_curve(df_base, df_hyb):
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    # Baseline AUC
    ax.plot(df_base['epoch'], df_base['auc'], color='#F44336', linewidth=1.8, linestyle='--', label='Baseline AUC')
    best_base_idx = df_base['auc'].idxmax()
    best_base = df_base.iloc[best_base_idx]
    ax.scatter([best_base['epoch']], [best_base['auc']], color='#F44336', zorder=5, s=60, marker='x',
               label=f'Base Best: {best_base["auc"]:.4f} (ep {best_base["epoch"]})')

    # Hybrid AUC
    ax.plot(df_hyb['epoch'], df_hyb['auc'], color='#4CAF50', linewidth=1.8, label='Hybrid AUC')
    best_hyb_idx = df_hyb['auc'].idxmax()
    best_hyb = df_hyb.iloc[best_hyb_idx]
    ax.scatter([best_hyb['epoch']], [best_hyb['auc']], color='gold', zorder=5, s=60,
               label=f'Hyb Best: {best_hyb["auc"]:.4f} (ep {best_hyb["epoch"]})')


    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.88, 1.0)
    ax.legend(fontsize=9, loc='lower right', framealpha=1.0)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()

    fig.tight_layout()
    save_fig(fig, 'comparison_auc_curve')

def get_cm(best_row):
    sens = best_row['sensitivity']   # = TP / (TP+FN)
    spec = best_row['specificity']   # = TN / (TN+FP)

    tp = round(sens * VAL_ALL)
    fn = VAL_ALL - tp
    tn = round(spec * VAL_HEM)
    fp = VAL_HEM - tn
    cm = np.array([[tp, fn],
                   [fp, tn]])
    return cm

def fig3_comparison_confusion_matrices(df_base, df_hyb):
    best_base = df_base.iloc[df_base['auc'].idxmax()]
    best_hyb = df_hyb.iloc[df_hyb['auc'].idxmax()]

    cm_base = get_cm(best_base)
    cm_hyb = get_cm(best_hyb)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4))

    classes = ['ALL (Blast)', 'HEM (Normal)']
    
    for ax, cm, title in zip(axes, [cm_base, cm_hyb], ['Baseline Confusion Matrix', 'Hybrid Focal Confusion Matrix']):
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', rasterized=True)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title, fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}',
                        ha='center', va='center', fontsize=12,
                        color='white' if cm[i, j] > thresh else 'black')
        
        ax.tick_params(direction='in', which='both')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.tight_layout()
    save_fig(fig, 'comparison_confusion_matrices')

if __name__ == '__main__':
    print(f"Generating Comparative Paper Figures -> {os.path.abspath(OUT_DIR)}\n")
    try:
        # Check original baseline fold1 config that ran in background
        df_baseline = get_latest_metrics('mnv3l_bs48_fold1')
        df_hybrid = get_latest_metrics('mnv3l_hybrid_gc_fold1')
        
        fig1_comparison_loss_curves(df_baseline, df_hybrid)
        fig2_comparison_auc_curve(df_baseline, df_hybrid)
        fig3_comparison_confusion_matrices(df_baseline, df_hybrid)
        print("\nDone. All comparison figures saved.")
    except Exception as e:
        print(f"Error generating comparison figures: {e}")
