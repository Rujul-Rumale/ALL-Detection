"""
Create Balanced C-NMC Dataset for Training
===========================================
Combines all 3 C-NMC folds, applies stain normalization, then balances classes
by oversampling HEM with augmented copies. Creates a clean 80/20 stratified split.

Result: ~5000 ALL + ~5000 HEM for training (balanced)

Usage:
  python training_scripts/balance_dataset.py
"""
import os
import sys
import random
import shutil
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

RAW_DIR = r"c:\Open Source\leukiemea\C-NMC"
NORMED_DIR = r"c:\Open Source\leukiemea\cnmc_staging_normed"
DST_DIR = r"c:\Open Source\leukiemea\cnmc_staging_balanced"

RANDOM_SEED = 42
VAL_RATIO = 0.20  # 80/20 split


# ============ AUGMENTATION ============
def augment_image(img):
    """Apply a random combination of augmentations to create a unique variant."""
    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    if angle > 0:
        img = img.rotate(angle)
    # Random brightness
    if random.random() > 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    # Random contrast
    if random.random() > 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    # Random color shift
    if random.random() > 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    # Random slight blur
    if random.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
    # Random sharpness
    if random.random() > 0.5:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.3))
    return img


def process_copy(args):
    """Copy a single image (original or augmented)."""
    src_path, dst_path, do_augment = args
    try:
        img = Image.open(src_path).convert("RGB")
        if do_augment:
            img = augment_image(img)
        img.save(dst_path, quality=95)
        return True
    except Exception:
        return False


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Creating Balanced C-NMC Dataset")
    print("=" * 60)

    # Step 1: Collect all image paths from stain-normalized data
    # (If normed dir exists, use it. Otherwise use raw C-NMC folds)
    use_normed = os.path.isdir(NORMED_DIR)

    all_files = {'all': [], 'hem': []}

    if use_normed:
        print(f"\n  Using stain-normalized images from: {NORMED_DIR}")
        for split in ['train', 'val']:
            split_dir = os.path.join(NORMED_DIR, split)
            if not os.path.isdir(split_dir):
                continue
            for cls in ['all', 'hem']:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            all_files[cls].append(os.path.join(cls_dir, f))
    else:
        print(f"\n  Using raw C-NMC folds from: {RAW_DIR}")
        for fold in sorted(os.listdir(RAW_DIR)):
            fold_inner = os.path.join(RAW_DIR, fold, fold)
            if not os.path.isdir(fold_inner):
                continue
            for cls in ['all', 'hem']:
                cls_dir = os.path.join(fold_inner, cls)
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            all_files[cls].append(os.path.join(cls_dir, f))

    n_all = len(all_files['all'])
    n_hem = len(all_files['hem'])
    print(f"\n  Original counts:")
    print(f"    ALL: {n_all}")
    print(f"    HEM: {n_hem}")
    print(f"    Total: {n_all + n_hem}")
    print(f"    Ratio: {n_all / n_hem:.2f}:1")

    # Step 2: Create stratified train/val split from ALL original images
    print(f"\n  Creating {int((1-VAL_RATIO)*100)}/{int(VAL_RATIO*100)} stratified split...")

    all_labels = [0] * n_all + [1] * n_hem
    all_paths = all_files['all'] + all_files['hem']

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=VAL_RATIO,
        stratify=all_labels,
        random_state=RANDOM_SEED
    )

    # Count per-class in train
    train_all = sum(1 for l in train_labels if l == 0)
    train_hem = sum(1 for l in train_labels if l == 1)
    val_all = sum(1 for l in val_labels if l == 0)
    val_hem = sum(1 for l in val_labels if l == 1)

    print(f"    Train: ALL={train_all}, HEM={train_hem} (total {len(train_paths)})")
    print(f"    Val:   ALL={val_all}, HEM={val_hem} (total {len(val_paths)})")

    # Step 3: Determine balancing strategy for training set
    deficit = train_all - train_hem
    print(f"\n  Balancing train set: need {deficit} augmented HEM images")
    print(f"    Target: {train_all} ALL + {train_all} HEM = {train_all * 2} total")

    # Step 4: Prepare copy tasks
    tasks = []

    # 4a: Copy all train images
    for path, label in zip(train_paths, train_labels):
        cls = 'all' if label == 0 else 'hem'
        fname = os.path.basename(path)
        dst = os.path.join(DST_DIR, "train", cls, fname)
        tasks.append((path, dst, False))

    # 4b: Create augmented HEM images to balance
    hem_train_paths = [p for p, l in zip(train_paths, train_labels) if l == 1]
    for i in range(deficit):
        src_path = hem_train_paths[i % len(hem_train_paths)]
        base, ext = os.path.splitext(os.path.basename(src_path))
        aug_fname = f"{base}_aug{i:04d}{ext}"
        dst = os.path.join(DST_DIR, "train", "hem", aug_fname)
        tasks.append((src_path, dst, True))

    # 4c: Copy all val images (unchanged, no balancing)
    for path, label in zip(val_paths, val_labels):
        cls = 'all' if label == 0 else 'hem'
        fname = os.path.basename(path)
        dst = os.path.join(DST_DIR, "val", cls, fname)
        tasks.append((path, dst, False))

    # Step 5: Execute
    print(f"\n  Processing {len(tasks)} total images...")
    os.makedirs(os.path.join(DST_DIR, "train", "all"), exist_ok=True)
    os.makedirs(os.path.join(DST_DIR, "train", "hem"), exist_ok=True)
    os.makedirs(os.path.join(DST_DIR, "val", "all"), exist_ok=True)
    os.makedirs(os.path.join(DST_DIR, "val", "hem"), exist_ok=True)

    t0 = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(process_copy, t) for t in tasks]
        for f in as_completed(futures):
            if not f.result():
                failed += 1
            done += 1
            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    [{done:>6d}/{len(tasks)}] {rate:.1f} img/s | "
                      f"elapsed {elapsed:.0f}s | failed {failed}")

    # Step 6: Verify
    print(f"\n  {'=' * 50}")
    print(f"  FINAL BALANCED DATASET")
    print(f"  {'=' * 50}")
    for split in ["train", "val"]:
        for cls in sorted(os.listdir(os.path.join(DST_DIR, split))):
            cls_dir = os.path.join(DST_DIR, split, cls)
            n = len(os.listdir(cls_dir))
            print(f"    {split}/{cls}: {n}")

    train_total = sum(len(os.listdir(os.path.join(DST_DIR, "train", c)))
                      for c in os.listdir(os.path.join(DST_DIR, "train")))
    val_total = sum(len(os.listdir(os.path.join(DST_DIR, "val", c)))
                    for c in os.listdir(os.path.join(DST_DIR, "val")))
    print(f"    Train total: {train_total}")
    print(f"    Val total: {val_total}")
    print(f"    Grand total: {train_total + val_total}")
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s | Failed: {failed}")
    print(f"  Output: {DST_DIR}")


if __name__ == "__main__":
    main()
