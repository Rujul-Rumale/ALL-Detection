"""
build_cv_splits.py
==================
Phase 0 Step 0.5 -- Build Patient-Level CV Splits

Produces cv_splits_3fold.json: the ground truth for all training/validation
partitioning in this project. Every accuracy number in the paper derives from
this file. Its correctness is critical.

Data sources (raw .bmp, no normalization -- CNMC data is pre-normalized by creators):
  Training folds:
    C-NMC_Dataset/PKG - C-NMC 2019/C-NMC_training_data/fold_{0,1,2}/all|hem/
    Naming: UID_{P}_{N}_{C}_all.bmp  (ALL, patient ID = token 1)
            UID_H{S}_{N}_{C}_hem.bmp (HEM, patient ID = token 1, strip leading H)

  Preliminary test set (usable for training -- labels provided):
    C-NMC_Dataset/PKG - C-NMC 2019/C-NMC_test_prelim_phase_data/  (flat numbered .bmp)
    Labels: C-NMC_test_prelim_phase_data_labels.csv
    CSV columns: Patient_ID (original UID filename), new_names (e.g. 1.bmp), labels (1=ALL, 0=HEM)
    LABEL REMAP: CSV 1=ALL -> project 0, CSV 0=HEM -> project 1.  Done at load time.

  Final test set: NO LABELS. Not used anywhere.

Fold assignment rules:
  - Training fold images: assigned to fold matching their source directory (fold_0->1, fold_1->2, fold_2->3)
  - Prelim patients (28): distributed round-robin across folds 1,2,3 by sorted patient ID
  - Leakage check: assert zero patient overlap between every fold pair

Output:
  cv_splits/cv_splits_3fold.json      -- machine-readable, used by train_base.py
  cv_splits/cv_splits_audit.txt       -- human-readable, inspect before trusting
  cv_splits/prelim_split_assignment.json -- how the 28 prelim patients were assigned

Usage:
  python training_scripts/build_cv_splits.py
"""

import os
import csv
import json
import sys
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ── Dataset Auto-Detection ──────────────────────────────────────────────────
def find_pkg_base():
    """
    Search for the CNMC dataset root in common environment locations.
    """
    candidates = [
        # Kaggle Official
        "/kaggle/input/c-nmc-2019-dataset/C-NMC 2019 (PKG)",
        "/kaggle/input/c-nmc-leukemia-classification-challenge/C-NMC 2019 (PKG)",
        # Standard Local/Colab structure
        os.path.join(PROJECT_ROOT, "C-NMC_Dataset", "PKG - C-NMC 2019"),
        "/content/ALL-Detection/C-NMC_Dataset/PKG - C-NMC 2019",
        "/content/C-NMC_Dataset/PKG - C-NMC 2019",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            print(f"  FOUND DATASET ROOT: {cand}")
            return cand
            
    # Fallback to default if not found (will error cleanly later)
    return candidates[2]

PKG_BASE     = find_pkg_base()
TRAIN_BASE   = os.path.join(PKG_BASE, "C-NMC_training_data")
PRELIM_BASE  = os.path.join(PKG_BASE, "C-NMC_test_prelim_phase_data")
PRELIM_CSV   = os.path.join(PRELIM_BASE, "C-NMC_test_prelim_phase_data_labels.csv")

OUT_DIR      = os.path.join(PROJECT_ROOT, "cv_splits")
SPLITS_JSON  = os.path.join(OUT_DIR, "cv_splits_3fold.json")
AUDIT_TXT    = os.path.join(OUT_DIR, "cv_splits_audit.txt")
PRELIM_ASSIGN_JSON = os.path.join(OUT_DIR, "prelim_split_assignment.json")

# fold_0 -> CV fold 1, fold_1 -> CV fold 2, fold_2 -> CV fold 3
FOLD_DIR_TO_CV = {"fold_0": 1, "fold_1": 2, "fold_2": 3}
FOLDS = ["fold_0", "fold_1", "fold_2"]

EXPECTED_TRAIN_TOTAL = 10661
EXPECTED_PRELIM_ALL  = 1219
EXPECTED_PRELIM_HEM  = 648
EXPECTED_PRELIM_TOTAL = 1867


# ── Patient ID extraction ──────────────────────────────────────────────────────
def extract_patient_id(filename, cls_prefix=None):
    """
    Extract patient ID from confirmed CNMC filename format.
    Namespaces the ID with cls_prefix (e.g. 'all_11') to avoid false ID collisions
    between independent cohorts (ALL vs HEM).

    ALL:  UID_{P}_{N}_{C}_all.bmp  -> token[1] is P (e.g. UID_11_10_1_all.bmp -> '11')
    HEM:  UID_H{S}_{N}_{C}_hem.bmp -> token[1] is 'H{S}', strip leading 'H' -> S
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse patient ID from filename: {filename!r}")
    token = parts[1]
    pid = token[1:] if token.startswith("H") else token

    if cls_prefix:
        return f"{cls_prefix}_{pid}"
    return pid


# ── Step 1: Index training fold images ────────────────────────────────────────
def collect_training_fold_images():
    """
    Returns dict: {cv_fold_num: {'all': [(filepath, patient_id), ...],
                                  'hem': [(filepath, patient_id), ...]}}
    """
    data = {1: {"all": [], "hem": []},
            2: {"all": [], "hem": []},
            3: {"all": [], "hem": []}}
    total = 0

    for fold_dir, cv_fold in FOLD_DIR_TO_CV.items():
        for cls in ["all", "hem"]:
            dir_path = os.path.join(TRAIN_BASE, fold_dir, cls)
            if not os.path.isdir(dir_path):
                print(f"  ERROR: Missing directory {dir_path}")
                sys.exit(1)
            for fname in sorted(os.listdir(dir_path)):
                if fname.startswith("."):
                    continue
                fpath = os.path.join(dir_path, fname)
                pid = extract_patient_id(fname, cls_prefix=cls)
                data[cv_fold][cls].append((fpath, pid))
                total += 1

    if total != EXPECTED_TRAIN_TOTAL:
        print(f"  ERROR: Expected {EXPECTED_TRAIN_TOTAL} training images, found {total}")
        sys.exit(1)
    print(f"  Training fold images indexed: {total} (expected {EXPECTED_TRAIN_TOTAL}) -- OK")
    return data


# ── Step 2: Index prelim test set images ──────────────────────────────────────
def collect_prelim_images():
    """
    Reads prelim CSV, remaps labels, returns list of:
      {'filepath': ..., 'patient_id': ..., 'cls': 'all'|'hem', 'label': 0|1}

    CSV label remap:
      CSV 1=ALL -> project encoding 0 (ALL=class 0)
      CSV 0=HEM -> project encoding 1 (HEM=class 1)
    Applied immediately on load. Raw CSV label values not used after this.
    """
    records = []
    with open(PRELIM_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_uid = row["Patient_ID"]    # e.g. UID_57_29_1_all.bmp
            new_name     = row["new_names"]      # e.g. 1.bmp
            csv_label    = int(row["labels"])

            # Prelim CSV: 1=ALL, 0=HEM -- OPPOSITE of project encoding (ALL=0, HEM=1)
            # Remap immediately on load, never use raw CSV label values after this line
            project_label = {1: 0, 0: 1}[csv_label]
            cls = "all" if project_label == 0 else "hem"

            # Patient ID: from the original UID filename (same format as training)
            # NAMESPACING: Prefix with 'prelim_' to avoid false leakage alarms
            # because prelim patients are a separate cohort from training.
            pid = "prelim_" + extract_patient_id(original_uid)

            filepath = os.path.join(PRELIM_BASE, new_name)
            if not os.path.isfile(filepath):
                print(f"  ERROR: Prelim image not found: {filepath}")
                sys.exit(1)

            records.append({
                "filepath":   filepath,
                "filename":   new_name,
                "patient_id": pid,
                "cls":        cls,
                "label":      project_label
            })

    # Assertion: remapped counts must match exactly
    all_count = sum(1 for r in records if r["cls"] == "all")
    hem_count = sum(1 for r in records if r["cls"] == "hem")
    if all_count != EXPECTED_PRELIM_ALL or hem_count != EXPECTED_PRELIM_HEM:
        print(f"  ERROR: Prelim label remap assertion FAILED: "
              f"got {all_count} ALL, {hem_count} HEM; "
              f"expected {EXPECTED_PRELIM_ALL} ALL, {EXPECTED_PRELIM_HEM} HEM")
        sys.exit(1)
    print(f"  Prelim label remap PASSED: {all_count} ALL, {hem_count} HEM -- OK")

    if len(records) != EXPECTED_PRELIM_TOTAL:
        print(f"  ERROR: Expected {EXPECTED_PRELIM_TOTAL} prelim records, got {len(records)}")
        sys.exit(1)
    print(f"  Prelim images indexed: {len(records)} -- OK")
    return records


# ── Step 3: Distribute prelim patients round-robin ────────────────────────────
def assign_prelim_patients(prelim_records):
    """
    Distribute the 28 prelim patients across folds 1, 2, 3.
    Rule: sort patient IDs alphabetically (then numerically), assign round-robin.
    Deterministic: no randomness.
    Returns: {patient_id: cv_fold_num}
    """
    # Get unique patient IDs with their dominant class
    patient_cls = {}
    for r in prelim_records:
        pid = r["patient_id"]
        if pid not in patient_cls:
            patient_cls[pid] = {"all": 0, "hem": 0}
        patient_cls[pid][r["cls"]] += 1

    # Sort: first numerically if all digits, else lexicographically
    def sort_key(pid):
        return (0, int(pid)) if pid.isdigit() else (1, pid)

    sorted_patients = sorted(patient_cls.keys(), key=sort_key)
    print(f"  Prelim unique patients: {len(sorted_patients)}")

    assignment = {}
    fold_cycle = [1, 2, 3]
    for i, pid in enumerate(sorted_patients):
        assignment[pid] = fold_cycle[i % 3]

    # Report distribution
    fold_counts = {1: 0, 2: 0, 3: 0}
    for pid, f in assignment.items():
        fold_counts[f] += 1
    print(f"  Prelim patient assignment: fold1={fold_counts[1]}, "
          f"fold2={fold_counts[2]}, fold3={fold_counts[3]}")

    return assignment, sorted_patients, patient_cls


# ── Step 4: Leakage check ─────────────────────────────────────────────────────
def check_leakage(fold_patients):
    """
    fold_patients: {fold_num: set(patient_ids)}
    Asserts zero overlap between every fold pair. Exits on failure.
    """
    pairs = [(1, 2), (1, 3), (2, 3)]
    passed = True
    for fa, fb in pairs:
        overlap = fold_patients[fa] & fold_patients[fb]
        if overlap:
            print(f"  LEAKAGE CHECK: FAILED -- folds {fa}&{fb} share patients: {overlap}")
            passed = False
        else:
            print(f"  LEAKAGE CHECK: fold{fa} vs fold{fb} -- PASSED (0 shared patients)")
    if not passed:
        print("  HALTING: Patient leakage detected. Fix split assignment before proceeding.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  build_cv_splits.py -- Phase 0 Step 0.5")
    print("  Patient-level 3-fold CV splits for CNMC 12k dataset")
    print("=" * 70)

    # ── Collect data ──────────────────────────────────────────────────────────
    print("\n[STEP 1] Indexing training fold images...")
    fold_data = collect_training_fold_images()

    print("\n[STEP 2] Indexing preliminary test set images...")
    prelim_records = collect_prelim_images()

    print("\n[STEP 3] Assigning prelim patients to folds (round-robin)...")
    prelim_assignment, sorted_prelim_patients, prelim_patient_cls = \
        assign_prelim_patients(prelim_records)

    # ── Merge prelim into fold data ───────────────────────────────────────────
    print("\n[STEP 4] Merging prelim images into fold assignments...")
    for r in prelim_records:
        cv_fold = prelim_assignment[r["patient_id"]]
        fold_data[cv_fold][r["cls"]].append((r["filepath"], r["patient_id"]))

    # ── Build fold patient sets ────────────────────────────────────────────────
    fold_patients = {}
    for cv_fold in [1, 2, 3]:
        pids = set()
        for cls in ["all", "hem"]:
            for fpath, pid in fold_data[cv_fold][cls]:
                pids.add(pid)
        fold_patients[cv_fold] = pids

    # ── Leakage check ─────────────────────────────────────────────────────────
    print("\n[STEP 5] Running leakage check...")
    check_leakage(fold_patients)

    # ── Build CV splits JSON ──────────────────────────────────────────────────
    print("\n[STEP 6] Building cv_splits_3fold.json...")

    # Total images after merging
    total_images = sum(
        len(fold_data[f][cls]) for f in [1, 2, 3] for cls in ["all", "hem"]
    )
    total_patients = len(set(
        pid for f in [1, 2, 3] for cls in ["all", "hem"]
        for _, pid in fold_data[f][cls]
    ))

    splits_out = {
        "metadata": {
            "total_patients": total_patients,
            "total_images": total_images,
            "folds": 3,
            "seed": "official_dataset_folds_plus_roundrobin_prelim",
            "fold_dir_to_cv_fold": {"fold_0": 1, "fold_1": 2, "fold_2": 3},
            "prelim_assignment_rule": "sort_patient_ids_numerically_then_roundrobin_1_2_3",
            "label_encoding": {"all": 0, "hem": 1},
            "data_source": "raw_bmp_from_cnmc_dataset_creator_prenormalized",
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "folds": {}
    }

    # For each CV configuration: fold N is val, the other two are train
    for val_fold in [1, 2, 3]:
        train_folds = [f for f in [1, 2, 3] if f != val_fold]

        val_images = [(fp, 0) for fp, _ in fold_data[val_fold]["all"]] + \
                     [(fp, 1) for fp, _ in fold_data[val_fold]["hem"]]
        val_patients = sorted(fold_patients[val_fold])

        train_images = []
        train_counts = {"all": 0, "hem": 0}
        for tf in train_folds:
            train_images += [(fp, 0) for fp, _ in fold_data[tf]["all"]]
            train_counts["all"] += len(fold_data[tf]["all"])
            train_images += [(fp, 1) for fp, _ in fold_data[tf]["hem"]]
            train_counts["hem"] += len(fold_data[tf]["hem"])
        
        train_patients = sorted(
            set(pid for tf in train_folds for pid in fold_patients[tf])
        )

        splits_out["folds"][f"fold_{val_fold}"] = {
            "val_fold_source": val_fold,
            "train_fold_sources": train_folds,
            "val_patients":   val_patients,
            "train_patients": train_patients,
            "val_images":   val_images,
            "train_images": train_images,
            "train_counts": {
                "all": train_counts["all"],
                "hem": train_counts["hem"],
                "total": train_counts["all"] + train_counts["hem"]
            },
            "val_counts": {
                "all": len(fold_data[val_fold]["all"]),
                "hem": len(fold_data[val_fold]["hem"]),
                "total": len(fold_data[val_fold]["all"]) + len(fold_data[val_fold]["hem"])
            }
        }

    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)
    print(f"  Written: {SPLITS_JSON}")

    # ── Prelim assignment JSON ─────────────────────────────────────────────────
    prelim_assign_out = {
        "description": "Round-robin assignment of 28 prelim patients to CV folds",
        "rule": "Sort patient IDs numerically, assign fold 1/2/3 cyclically",
        "assignment": {pid: prelim_assignment[pid] for pid in sorted_prelim_patients}
    }
    with open(PRELIM_ASSIGN_JSON, "w", encoding="utf-8") as f:
        json.dump(prelim_assign_out, f, indent=2)
    print(f"  Written: {PRELIM_ASSIGN_JSON}")

    # ── Audit text file ────────────────────────────────────────────────────────
    print("\n[STEP 7] Writing audit file...")
    with open(AUDIT_TXT, "w", encoding="utf-8") as out:
        out.write("CV Splits Audit Report\n")
        out.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        out.write(f"Total images: {total_images}\n")
        out.write(f"Total patients: {total_patients}\n")
        out.write("=" * 70 + "\n\n")

        for val_fold in [1, 2, 3]:
            fd = splits_out["folds"][f"fold_{val_fold}"]
            out.write(f"FOLD {val_fold} as VALIDATION (train on folds {fd['train_fold_sources']})\n")
            out.write(f"  Train: {fd['train_counts']['all']} ALL + "
                      f"{fd['train_counts']['hem']} HEM = {fd['train_counts']['total']} total\n")
            out.write(f"  Val:   {fd['val_counts']['all']} ALL + "
                      f"{fd['val_counts']['hem']} HEM = {fd['val_counts']['total']} total\n")
            out.write(f"  Train patients: {len(fd['train_patients'])}\n")
            out.write(f"  Val patients:   {len(fd['val_patients'])}\n")

            # Leakage check line (already verified above, just document)
            overlap = set(fd["train_patients"]) & set(fd["val_patients"])
            if overlap:
                out.write(f"  LEAKAGE CHECK: FAILED -- {len(overlap)} shared patients!\n")
            else:
                out.write(f"  LEAKAGE CHECK: PASSED (0 shared patients)\n")
            out.write("\n")

        out.write("=" * 70 + "\n")
        out.write("ALL LEAKAGE CHECKS: PASSED\n")

    print(f"  Written: {AUDIT_TXT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CV SPLITS COMPLETE")
    print(f"  Total images in splits: {total_images}")
    print(f"  Total unique patients:  {total_patients}")
    print()
    for val_fold in [1, 2, 3]:
        fd = splits_out["folds"][f"fold_{val_fold}"]
        print(f"  Fold {val_fold} val:   {fd['val_counts']['all']} ALL + "
              f"{fd['val_counts']['hem']} HEM = {fd['val_counts']['total']}")
        print(f"  Fold {val_fold} train: {fd['train_counts']['all']} ALL + "
              f"{fd['train_counts']['hem']} HEM = {fd['train_counts']['total']}")
        print()
    print(f"  Output: {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
