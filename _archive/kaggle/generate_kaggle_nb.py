import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells_content = [
    # Cell 1: GPU Check
    "import subprocess\nprint(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)",
    
    # Cell 2: Install dependencies
    "!pip install -q timm>=0.9.0 albumentations>=1.3.0 rich",
    
    # Cell 3: Path remapping
    """import json
import re

print("Remapping cv_splits/cv_splits_3fold.json for Kaggle...")

with open('cv_splits/cv_splits_3fold.json', 'r') as f:
    data = json.load(f)

# The Windows base path used locally
LOCAL_PREFIX_REGEX = re.compile(r"c:/open source/leukiemea/c-nmc_dataset/", re.IGNORECASE)
KAGGLE_PREFIX = "/kaggle/input/c-nmc-2019-dataset/"

def remap_path(p):
    # Convert backslashes to forward slashes first
    p_forward = p.replace('\\', '/')
    # Also handle the lowercase version for regex
    # The regex matching is case-insensitive, but we need to replace the prefix safely
    # Easiest way: re.sub
    remapped = LOCAL_PREFIX_REGEX.sub(KAGGLE_PREFIX, p_forward)
    
    # Fallback if regex didn't catch due to varying slash amounts
    if not remapped.startswith('/kaggle'):
        # Just in case the regex missed something, do a highly permissive extract
        parts = p_forward.split('C-NMC_Dataset/')
        if len(parts) > 1:
            remapped = KAGGLE_PREFIX + parts[1]
    
    return remapped

for fold_key in data['folds']:
    # Remap train images
    new_train = []
    for pair in data['folds'][fold_key]['train_images']:
        new_train.append([remap_path(pair[0]), pair[1]])
    data['folds'][fold_key]['train_images'] = new_train
    
    # Remap val images
    new_val = []
    for pair in data['folds'][fold_key]['val_images']:
        new_val.append([remap_path(pair[0]), pair[1]])
    data['folds'][fold_key]['val_images'] = new_val

OUT_JSON = '/kaggle/working/cv_splits_kaggle.json'
with open(OUT_JSON, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Saved to {OUT_JSON}")
print("\\nFirst 3 remapped paths (Fold 1 Train):")
for pair in data['folds']['fold_1']['train_images'][:3]:
    print(pair[0])""",

    # Cell 4: Verify Images
    """import os
import json

print("Spot checking 5 paths from the remapped JSON to ensure they exist on disk...")
with open('/kaggle/working/cv_splits_kaggle.json', 'r') as f:
    check_data = json.load(f)

sample_paths = [pair[0] for pair in check_data['folds']['fold_1']['train_images'][:5]]
all_passed = True

for p in sample_paths:
    exists = os.path.exists(p)
    status = "PASS" if exists else "FAIL"
    print(f"[{status}] {p}")
    if not exists:
        all_passed = False

if not all_passed:
    raise FileNotFoundError("One or more sample paths do not exist. Please check the dataset mount path in Kaggle (/kaggle/input/c-nmc-2019-dataset/) and ensure it matches the script.")
else:
    print("All sample paths verified successfully!")""",

    # Cell 5: Train Fold 1
    """!python training_scripts/train_base.py \\
  --model effb4 \\
  --fold 1 \\
  --run_name effb4_cnmc_kaggle \\
  --splits_json /kaggle/working/cv_splits_kaggle.json \\
  --output_dir /kaggle/working/outputs \\
  --res 320 \\
  --batch_size 48 \\
  --num_workers 2 \\
  --epochs 150 \\
  --patience 25 \\
  --lr_backbone 1e-5 \\
  --lr_head 1e-4 \\
  --no_live""",

    # Cell 6: Train Fold 2
    """!python training_scripts/train_base.py \\
  --model effb4 \\
  --fold 2 \\
  --run_name effb4_cnmc_kaggle \\
  --splits_json /kaggle/working/cv_splits_kaggle.json \\
  --output_dir /kaggle/working/outputs \\
  --res 320 \\
  --batch_size 48 \\
  --num_workers 2 \\
  --epochs 150 \\
  --patience 25 \\
  --lr_backbone 1e-5 \\
  --lr_head 1e-4 \\
  --no_live""",

    # Cell 7: Train Fold 3
    """!python training_scripts/train_base.py \\
  --model effb4 \\
  --fold 3 \\
  --run_name effb4_cnmc_kaggle \\
  --splits_json /kaggle/working/cv_splits_kaggle.json \\
  --output_dir /kaggle/working/outputs \\
  --res 320 \\
  --batch_size 48 \\
  --num_workers 2 \\
  --epochs 150 \\
  --patience 25 \\
  --lr_backbone 1e-5 \\
  --lr_head 1e-4 \\
  --no_live""",

    # Cell 8: Checkpoint verification
    """import glob
import os

print("Looking for saved best checkpoints...")
ckpts = glob.glob("/kaggle/working/outputs/effb4_cnmc_kaggle/**/*_best.pth", recursive=True)

if not ckpts:
    print("WARNING: No checkpoints found!")
else:
    for c in ckpts:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        print(f"Found: {c} - {size_mb:.2f} MB")""",

    # Cell 9: Copy out
    """import shutil
import glob
import os

print("Copying checkpoints to root /kaggle/working/ for persistence...")
ckpts = glob.glob("/kaggle/working/outputs/effb4_cnmc_kaggle/**/*_best.pth", recursive=True)
for c in ckpts:
    dest = os.path.join("/kaggle/working/", os.path.basename(c))
    shutil.copy(c, dest)
    print(f"Saved to persistent output: {os.path.basename(dest)}")"""
]

for content in cells_content:
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in content.split("\n")]
    })

# Clean up final newlines
for cell in notebook["cells"]:
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1][:-1]

with open("kaggle_effb4_train.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Notebook generated successfully!")
