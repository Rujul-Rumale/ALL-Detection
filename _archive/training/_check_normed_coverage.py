"""
Pre-flight check: does cnmc_staging_normed cover the same images as the training folds?

Strategy:
- Collect all basenames from cnmc_staging_normed/train/all + val/all + train/hem + val/hem
- Collect all basenames from C-NMC_Dataset training folds (fold_0/1/2 all/hem)
- Compare: what percentage overlap? Are all fold images present in normed?
"""
import os

NORMED = r'c:\Open Source\leukiemea\cnmc_staging_normed'
FOLD_BASE = r'c:\Open Source\leukiemea\C-NMC_Dataset\PKG - C-NMC 2019\C-NMC_training_data'

# ── Collect normed filenames ──────────────────────────────────────────────
normed_files = set()
normed_counts = {}
for split in ['train', 'val']:
    for cls in ['all', 'hem']:
        p = os.path.join(NORMED, split, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if not f.startswith('.')]
            normed_counts[f'{split}/{cls}'] = len(files)
            normed_files.update(files)

print("=== NORMED FILES COUNT ===")
for k, v in normed_counts.items():
    print(f"  {k}: {v}")
print(f"  TOTAL UNIQUE NORMED: {len(normed_files)}")

# ── Collect fold filenames ─────────────────────────────────────────────────
fold_files = {}
fold_names = sorted([d for d in os.listdir(FOLD_BASE)
                     if os.path.isdir(os.path.join(FOLD_BASE, d)) and not d.startswith('.')])
all_fold_files = set()
print("\n=== FOLD FILES COUNT ===")
for fold in fold_names:
    fold_files[fold] = {}
    for cls in ['all', 'hem']:
        p = os.path.join(FOLD_BASE, fold, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if not f.startswith('.')]
            fold_files[fold][cls] = set(files)
            all_fold_files.update(files)
            print(f"  {fold}/{cls}: {len(files)}")
print(f"  TOTAL UNIQUE FOLD FILES: {len(all_fold_files)}")

# ── Cross-comparison ──────────────────────────────────────────────────────
in_both = normed_files & all_fold_files
only_in_normed = normed_files - all_fold_files
only_in_folds = all_fold_files - normed_files

print("\n=== COVERAGE ANALYSIS ===")
print(f"  Files in BOTH normed and folds:  {len(in_both)}")
print(f"  Files ONLY in normed (not in folds): {len(only_in_normed)}")
print(f"  Files ONLY in folds (not in normed): {len(only_in_folds)}")
print(f"  Coverage: {len(in_both)}/{len(all_fold_files)} = {100*len(in_both)/len(all_fold_files):.1f}%")

if len(only_in_folds) == 0:
    print("\n  VERDICT: cnmc_staging_normed FULLY COVERS all training fold images.")
    print("  >>> Copy from normed to cnmc_normed_training — DO NOT re-normalize.")
elif len(in_both) == 0:
    print("\n  VERDICT: ZERO OVERLAP — these are completely different datasets.")
    print("  >>> Must run full Macenko normalization on all fold images.")
else:
    print(f"\n  VERDICT: PARTIAL COVERAGE — {len(only_in_folds)} fold images are missing from normed.")
    print("  >>> Copy matching ones, normalize the rest.")
    print("\n  Sample missing files (first 10):")
    for f in sorted(only_in_folds)[:10]:
        print(f"    {f}")

if only_in_normed:
    print(f"\n  NOTE: {len(only_in_normed)} normed files have no match in folds (may be val split or old data)")
    print("  Sample (first 5):")
    for f in sorted(only_in_normed)[:5]:
        print(f"    {f}")
