import os

TRAIN_BASE = r'c:\Open Source\leukiemea\C-NMC_Dataset\PKG - C-NMC 2019\C-NMC_training_data'
PRELIM_BASE = r'c:\Open Source\leukiemea\C-NMC_Dataset\PKG - C-NMC 2019\C-NMC_test_prelim_phase_data'

# Fold names
folds = sorted([d for d in os.listdir(TRAIN_BASE) if os.path.isdir(os.path.join(TRAIN_BASE, d)) and not d.startswith('.')])
print("FOLD DIRECTORY NAMES:", folds)

# Counts
totals = {'all': 0, 'hem': 0}
for fold in folds:
    for cls in ['all', 'hem']:
        p = os.path.join(TRAIN_BASE, fold, cls)
        n = len([x for x in os.listdir(p) if not x.startswith('.')]) if os.path.isdir(p) else 0
        totals[cls] += n
        print(f"{fold}/{cls}: {n}")

print(f"TOTAL ALL: {totals['all']}  HEM: {totals['hem']}  GRAND: {totals['all']+totals['hem']}")
if totals['all'] == 7272 and totals['hem'] == 3389:
    print("COUNT CHECK: PASSED")
else:
    print(f"COUNT CHECK: FAILED (expected 7272 ALL, 3389 HEM)")

# Sample filenames from each fold/class
for fold in folds:
    for cls in ['all', 'hem']:
        p = os.path.join(TRAIN_BASE, fold, cls)
        if os.path.isdir(p):
            files = sorted([f for f in os.listdir(p) if not f.startswith('.')])[:3]
            print(f"{fold}/{cls}/: {files}")

# Prelim set structure
print("\nPRELIM TEST SET CONTENTS:")
for entry in sorted(os.listdir(PRELIM_BASE)):
    full = os.path.join(PRELIM_BASE, entry)
    if os.path.isdir(full):
        n = len([x for x in os.listdir(full) if not x.startswith('.')])
        print(f"  [DIR] {entry}/ ({n} items)")
        # Show 3 sample files
        samples = sorted([f for f in os.listdir(full) if not f.startswith('.')])[:3]
        for s in samples:
            print(f"    {s}")
    else:
        print(f"  [FILE] {entry}")
