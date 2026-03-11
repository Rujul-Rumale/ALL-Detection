#!/usr/bin/env python3
"""
verify_env.py
Checks all dependencies required for train_original_cpu_baseline.py
Run from project root: python linux_training/verify_env.py
"""

import sys

results = []

def check(name, fn):
    try:
        result = fn()
        results.append((name, True, result))
    except Exception as e:
        results.append((name, False, str(e)))

# Python version
check("Python >= 3.9", lambda: (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9) else (_ for _ in ()).throw(
        RuntimeError(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old")
    )
))

# PyTorch
check("torch", lambda: __import__("torch").__version__)

# CUDA
check("CUDA available", lambda: (
    __import__("torch").cuda.get_device_name(0)
    if __import__("torch").cuda.is_available()
    else (_ for _ in ()).throw(RuntimeError("CUDA not available"))
))

# AMP
check("torch.amp.autocast", lambda: (
    str(__import__("torch").amp.autocast)
))

# timm
check("timm >= 0.9", lambda: __import__("timm").__version__)

# albumentations
check("albumentations >= 1.3", lambda: __import__("albumentations").__version__)

# OpenCV
check("cv2", lambda: __import__("cv2").__version__)

# numpy
check("numpy", lambda: __import__("numpy").__version__)

# scikit-learn
check("sklearn", lambda: __import__("sklearn").__version__)

# PIL
check("Pillow", lambda: __import__("PIL").__version__)

# psutil
check("psutil", lambda: __import__("psutil").__version__)

# rich
check("rich", lambda: __import__("rich").__version__)

# nvidia-smi accessible
import subprocess
check("nvidia-smi", lambda: (
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
))

# cv_splits file exists
import os
check("cv_splits_3fold.json exists", lambda: (
    "FOUND" if os.path.exists("cv_splits/cv_splits_3fold.json")
    else (_ for _ in ()).throw(RuntimeError("File not found — run from project root"))
))

# training script exists
check("train_original_cpu_baseline.py exists", lambda: (
    "FOUND" if os.path.exists(
        "training_scripts/train_original_cpu_baseline.py")
    else (_ for _ in ()).throw(RuntimeError("File not found"))
))

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Environment Verification")
print("=" * 55)
all_passed = True
for name, passed, detail in results:
    status = "PASS" if passed else "FAIL"
    marker = "✓" if passed else "✗"
    print(f"  [{marker}] {status:<6} {name:<35} {detail}")
    if not passed:
        all_passed = False

print("=" * 55)
if all_passed:
    print("  All checks passed. Ready to train.")
else:
    print("  Some checks failed. Run install_deps.sh first.")
print("=" * 55 + "\n")
sys.exit(0 if all_passed else 1)
