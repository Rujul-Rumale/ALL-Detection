"""
normalize_cnmc_full.py
======================
Phase 0 -- Option A: Macenko Stain Normalization

Provenance chain (one unambiguous line for the paper):
  raw .bmp from C-NMC_Dataset -> Macenko normalization -> cnmc_normed_training / cnmc_normed_prelim

Processes:
  1. Training folds: C-NMC_Dataset/.../C-NMC_training_data/fold_{0,1,2}/all|hem/
     -> cnmc_normed_training/fold_{0,1,2}/all|hem/  (saves as .png)

  2. Preliminary test set: C-NMC_Dataset/.../C-NMC_test_prelim_phase_data/  (flat numbered .bmp)
     Labels from C-NMC_test_prelim_phase_data_labels.csv (original UID filenames mapped to numbers)
     -> cnmc_normed_prelim/all|hem/  (saved under original UID filename, .png)

Reference matrix: Macenko et al. (2009) standard HERef -- hardcoded, deterministic.
  HERef  = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
  maxCRef = [1.9705, 1.0308]

Failures:
  Logged to <project_root>/normalization_failures.txt.
  Failed images are EXCLUDED from output. Filenames recorded for CV splits metadata.

Expected counts after normalization:
  Training folds: 7272 ALL + 3389 HEM = 10661 total
  Prelim set:     1219 ALL +  648 HEM =  1867 total  (1868 images in CSV; 1 filename is header)

Usage:
  python training_scripts/normalize_cnmc_full.py
"""

import os
import sys
import csv
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"c:\Open Source\leukiemea"
PKG_BASE       = os.path.join(PROJECT_ROOT, "C-NMC_Dataset", "PKG - C-NMC 2019")
TRAIN_BASE     = os.path.join(PKG_BASE, "C-NMC_training_data")
PRELIM_BASE    = os.path.join(PKG_BASE, "C-NMC_test_prelim_phase_data")
PRELIM_CSV     = os.path.join(PRELIM_BASE, "C-NMC_test_prelim_phase_data_labels.csv")
PRELIM_IMG_DIR = PRELIM_BASE  # images are flat in this dir (1.bmp, 2.bmp, ...)

OUT_TRAIN      = os.path.join(PROJECT_ROOT, "cnmc_normed_training")
OUT_PRELIM     = os.path.join(PROJECT_ROOT, "cnmc_normed_prelim")
FAILURES_LOG   = os.path.join(PROJECT_ROOT, "normalization_failures.txt")

EXPECTED_TRAIN_ALL = 7272
EXPECTED_TRAIN_HEM = 3389
EXPECTED_PRELIM_ALL = 1219
EXPECTED_PRELIM_HEM = 648

FOLDS = ["fold_0", "fold_1", "fold_2"]


# ── Macenko Normalizer ────────────────────────────────────────────────────────
class MacenkoNormalizer:
    """
    Macenko et al. (2009) stain normalization.
    Reference matrix is hardcoded -- fully deterministic, no external template.
    HERef and maxCRef sourced from the standard Macenko reference.
    """
    def __init__(self):
        self.HERef   = np.array([[0.5626, 0.2159],
                                  [0.7201, 0.8012],
                                  [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def _get_stain_matrix(self, I, beta=0.15, alpha=1):
        I      = I.reshape(-1, 3).astype(np.float64)
        OD     = -np.log10(np.clip(I / 255.0, 1e-6, 1.0))
        mask   = np.all(OD > beta, axis=1)
        if mask.sum() < 10:
            return self.HERef, self.maxCRef
        OD_hat = OD[mask]
        try:
            _, _, V = np.linalg.svd(OD_hat, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.HERef, self.maxCRef
        V    = V[:2, :]
        proj = OD_hat @ V.T
        phi  = np.arctan2(proj[:, 1], proj[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        v1 = V.T @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V.T @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
        Y    = OD @ np.linalg.pinv(HE.T)  # OD=(K,3) @ pinv((2,3))=(3,2) -> (K,2)
        maxC = np.percentile(Y, 99, axis=0)
        return HE, maxC

    def normalize(self, img):
        h, w, _ = img.shape
        HE, maxC = self._get_stain_matrix(img)
        OD     = -np.log10(np.clip(img.reshape(-1, 3).astype(np.float64) / 255.0, 1e-6, 1.0))
        C      = OD @ np.linalg.pinv(HE.T)  # OD=(N,3) @ pinv((2,3))=(3,2) -> (N,2)
        C      = C / maxC * self.maxCRef
        OD_norm = C @ self.HERef.T
        I_norm  = np.clip(255.0 * np.power(10, -OD_norm), 0, 255).astype(np.uint8)
        return I_norm.reshape(h, w, 3)


# ── Worker (top-level for multiprocessing) ────────────────────────────────────
def _worker(args):
    """Normalize one image. Returns (success, src_path, error_msg_or_None).
    MacenkoNormalizer is instantiated locally -- no pickling, no shared state."""
    src_path, dst_path = args
    normalizer = MacenkoNormalizer()
    try:
        img = np.array(Image.open(src_path).convert("RGB"))
        normed = normalizer.normalize(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        Image.fromarray(normed).save(dst_path)
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)


# ── Task builder helpers ──────────────────────────────────────────────────────
def build_train_tasks():
    """Build (src, dst) pairs for all training fold images."""
    tasks = []
    for fold in FOLDS:
        for cls in ["all", "hem"]:
            src_dir = os.path.join(TRAIN_BASE, fold, cls)
            dst_dir = os.path.join(OUT_TRAIN, fold, cls)
            if not os.path.isdir(src_dir):
                print(f"  WARNING: missing expected directory {src_dir}")
                continue
            for fname in os.listdir(src_dir):
                if fname.startswith("."):
                    continue
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png")
                tasks.append((src, dst))
    return tasks


def build_prelim_tasks():
    """
    Build (src, dst) pairs for the preliminary test set images.
    The prelim images are flat numbered .bmp files (1.bmp, 2.bmp, ...).
    Their original UID filenames and labels are in the CSV.

    CSV format: Patient_ID (original UID filename), new_names (e.g. 1.bmp), labels (1=ALL, 0=HEM)
    REMAPPING: CSV labels 1=ALL -> project encoding 0, CSV 0=HEM -> project encoding 1.
    """
    tasks = []    # (src_path, dst_path)
    label_map = {}  # new_name (e.g. "1.bmp") -> (original_uid, project_label)

    with open(PRELIM_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_uid  = row["Patient_ID"]     # e.g. UID_57_29_1_all.bmp
            new_name      = row["new_names"]       # e.g. 1.bmp
            csv_label     = int(row["labels"])

            # Prelim CSV: 1=ALL, 0=HEM -- OPPOSITE of project encoding (ALL=0, HEM=1)
            # Remap immediately on load, never use raw CSV label values after this line
            project_label = {1: 0, 0: 1}[csv_label]   # 1->0 (ALL), 0->1 (HEM)
            cls = "all" if project_label == 0 else "hem"

            label_map[new_name] = (original_uid, project_label, cls)

    # Assertion: remapped labels must produce exactly 1219 ALL and 648 HEM
    all_count = sum(1 for v in label_map.values() if v[2] == "all")
    hem_count = sum(1 for v in label_map.values() if v[2] == "hem")
    if all_count != EXPECTED_PRELIM_ALL or hem_count != EXPECTED_PRELIM_HEM:
        raise RuntimeError(
            f"PRELIM LABEL REMAP ASSERTION FAILED: "
            f"expected {EXPECTED_PRELIM_ALL} ALL and {EXPECTED_PRELIM_HEM} HEM, "
            f"got {all_count} ALL and {hem_count} HEM. STOPPING."
        )
    print(f"  Prelim label remap assertion PASSED: {all_count} ALL, {hem_count} HEM")

    for new_name, (original_uid, project_label, cls) in label_map.items():
        src = os.path.join(PRELIM_IMG_DIR, new_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Prelim image not found: {src}")
        # Save under original UID filename (strip extension, add .png)
        out_stem = os.path.splitext(original_uid)[0]
        dst = os.path.join(OUT_PRELIM, cls, out_stem + ".png")
        tasks.append((src, dst))

    return tasks


def run_normalization(tasks, label, failure_log_lines):
    """Process tasks sequentially. Returns (success_count, failure_count).

    Sequential is used because Macenko normalization is numpy/SVD-heavy (CPU-bound)
    and Python's GIL prevents true thread parallelism for this workload.
    """
    total    = len(tasks)
    success  = 0
    failures = 0
    t0       = time.time()
    normalizer = MacenkoNormalizer()

    print(f"\n  Processing {total} images sequentially...")
    for i, (src_path, dst_path) in enumerate(tasks):
        try:
            img    = np.array(Image.open(src_path).convert("RGB"))
            normed = normalizer.normalize(img)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            Image.fromarray(normed).save(dst_path)
            success += 1
        except Exception as e:
            failures += 1
            failure_log_lines.append(f"[{label}] {src_path} | {e}")

        done = i + 1
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta  = (total - done) / rate if rate > 0 else 0
            print(f"    [{done:>6d}/{total}] {rate:.0f} img/s | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | failed {failures}")

    return success, failures


def verify_counts(base_dir, expected_all, expected_hem, label):
    """Assert output counts match expectations. Raises on mismatch."""
    actual_all = actual_hem = 0
    # handle both flat (prelim) and fold-structured (training) layouts
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel = os.path.relpath(root, base_dir).replace("\\", "/")
        pngs = [f for f in files if f.endswith(".png")]
        if rel.endswith("/all") or rel == "all":
            actual_all += len(pngs)
        elif rel.endswith("/hem") or rel == "hem":
            actual_hem += len(pngs)

    grand = actual_all + actual_hem
    print(f"\n  {label} COUNT VERIFICATION:")
    print(f"    ALL: {actual_all}  (expected {expected_all})")
    print(f"    HEM: {actual_hem}  (expected {expected_hem})")
    print(f"    TOTAL: {grand}  (expected {expected_all + expected_hem})")

    if actual_all != expected_all or actual_hem != expected_hem:
        raise RuntimeError(
            f"COUNT ASSERTION FAILED for {label}: "
            f"got {actual_all} ALL / {actual_hem} HEM, "
            f"expected {expected_all} ALL / {expected_hem} HEM. STOPPING."
        )
    print(f"    STATUS: PASSED OK")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    failure_log_lines = []

    print("=" * 70)
    print("  normalize_cnmc_full.py -- Phase 0 Option A")
    print("  Macenko normalization: raw .bmp -> .png")
    print(f"  Training source:  {TRAIN_BASE}")
    print(f"  Prelim source:    {PRELIM_BASE}")
    print(f"  Training output:  {OUT_TRAIN}")
    print(f"  Prelim output:    {OUT_PRELIM}")
    print(f"  Failures log:     {FAILURES_LOG}")
    print("=" * 70)

    # ── Step 1: Training folds ────────────────────────────────────────────
    print("\n[STEP 1] Building training fold task list...")
    train_tasks = build_train_tasks()
    print(f"  Total training images to normalize: {len(train_tasks)}")

    t_start = time.time()
    s1, f1 = run_normalization(train_tasks, "TRAIN", failure_log_lines)
    print(f"\n  Training normalization done in {(time.time()-t_start)/60:.1f} min")
    print(f"  Succeeded: {s1}  |  Failed: {f1}")

    # ── Step 2: Prelim test set ───────────────────────────────────────────
    print("\n[STEP 2] Building prelim test set task list...")
    prelim_tasks = build_prelim_tasks()
    print(f"  Total prelim images to normalize: {len(prelim_tasks)}")

    t_start = time.time()
    s2, f2 = run_normalization(prelim_tasks, "PRELIM", failure_log_lines)
    print(f"\n  Prelim normalization done in {(time.time()-t_start)/60:.1f} min")
    print(f"  Succeeded: {s2}  |  Failed: {f2}")

    # ── Step 3: Write failures log ────────────────────────────────────────
    total_failures = f1 + f2
    if failure_log_lines:
        with open(FAILURES_LOG, "w") as log:
            log.write(f"normalization_failures.txt\n")
            log.write(f"Total failures: {total_failures}\n")
            log.write("=" * 70 + "\n\n")
            for line in failure_log_lines:
                log.write(line + "\n")
        print(f"\n  Failures written to: {FAILURES_LOG}")
    else:
        # Write a clean log to confirm the run completed
        with open(FAILURES_LOG, "w") as log:
            log.write("normalization_failures.txt\n")
            log.write("Total failures: 0\n")
            log.write("All images normalized successfully.\n")
        print(f"\n  No failures. Clean log written to: {FAILURES_LOG}")

    # ── Step 4: Count verification ────────────────────────────────────────
    # Training expected counts account for failures being excluded
    adjusted_train_all = EXPECTED_TRAIN_ALL - sum(
        1 for l in failure_log_lines if "[TRAIN]" in l and "_all" in l)
    adjusted_train_hem = EXPECTED_TRAIN_HEM - sum(
        1 for l in failure_log_lines if "[TRAIN]" in l and "_hem" in l)
    adjusted_prelim_all = EXPECTED_PRELIM_ALL - sum(
        1 for l in failure_log_lines if "[PRELIM]" in l and "_all" in l)
    adjusted_prelim_hem = EXPECTED_PRELIM_HEM - sum(
        1 for l in failure_log_lines if "[PRELIM]" in l and "_hem" in l)

    try:
        verify_counts(OUT_TRAIN,  adjusted_train_all,  adjusted_train_hem,  "cnmc_normed_training")
        verify_counts(OUT_PRELIM, adjusted_prelim_all, adjusted_prelim_hem, "cnmc_normed_prelim")
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # ── Step 5: Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NORMALIZATION COMPLETE")
    print(f"  Training images normalized: {s1}")
    print(f"  Prelim images normalized:   {s2}")
    print(f"  Total failures:             {total_failures}")
    if total_failures > 0:
        print(f"  Failure details: {FAILURES_LOG}")
        print("  NOTE: Failed images are excluded from all subsequent steps.")
        print("        Record their filenames in cv_splits metadata for paper trail.")
    print()
    if total_failures == 0:
        print("  SAFE TO DELETE (after first training run verifies the new pipeline):")
        print("    c:\\Open Source\\leukiemea\\C-NMC\\")
        print("    c:\\Open Source\\leukiemea\\cnmc_staging\\")
        print("    c:\\Open Source\\leukiemea\\cnmc_staging_normed\\")
    print("=" * 70)


if __name__ == "__main__":
    main()
