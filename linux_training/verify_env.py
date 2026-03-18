#!/usr/bin/env python3
"""
verify_env.py
Checks the Fedora/Linux environment against the current training path.
Run from project root: python linux_training/verify_env.py
"""

import json
import os
import subprocess
import sys

results = []


def check(name, fn, optional=False):
    try:
        result = fn()
        results.append((name, optional, True, result))
    except Exception as e:
        results.append((name, optional, False, str(e)))


check(
    "Python >= 3.10",
    lambda: (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 10)
        else (_ for _ in ()).throw(
            RuntimeError(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is too old"
            )
        )
    ),
)

check("torch", lambda: __import__("torch").__version__)
check("torchvision", lambda: __import__("torchvision").__version__)
check("timm", lambda: __import__("timm").__version__)
check("albumentations", lambda: __import__("albumentations").__version__)
check("cv2", lambda: __import__("cv2").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("sklearn", lambda: __import__("sklearn").__version__)
check("skimage", lambda: __import__("skimage").__version__)
check("Pillow", lambda: __import__("PIL").__version__)
check("psutil", lambda: __import__("psutil").__version__)
check("rich", lambda: __import__("rich").__version__)
check("pandas", lambda: __import__("pandas").__version__)
check("tqdm", lambda: __import__("tqdm").__version__)
check("torch.amp.autocast", lambda: str(__import__("torch").amp.autocast))

check(
    "CUDA available",
    lambda: (
        __import__("torch").cuda.get_device_name(0)
        if __import__("torch").cuda.is_available()
        else (_ for _ in ()).throw(RuntimeError("CUDA not available"))
    ),
    optional=True,
)

check(
    "nvidia-smi",
    lambda: (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ),
    optional=True,
)

check("tensorflow", lambda: __import__("tensorflow").__version__, optional=True)
check("transformers", lambda: __import__("transformers").__version__, optional=True)
check("customtkinter", lambda: __import__("customtkinter").__version__, optional=True)
check("tkinter", lambda: "FOUND", optional=True)
check("onnx", lambda: __import__("onnx").__version__, optional=True)
check("onnx_tf", lambda: __import__("onnx_tf").__version__, optional=True)
check("pytest", lambda: __import__("pytest").__version__, optional=True)

check(
    "cv_splits_3fold.json exists",
    lambda: (
        "FOUND"
        if os.path.exists("cv_splits/cv_splits_3fold.json")
        else (_ for _ in ()).throw(RuntimeError("File not found - run from project root"))
    ),
)


def splits_need_regen():
    with open("cv_splits/cv_splits_3fold.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = data["folds"]["fold_1"]["train_images"][0][0]
    if ":" in sample or "\\" in sample:
        raise RuntimeError(
            f"Split file still contains non-Linux paths: {sample}. "
            "Regenerate with python training_scripts/build_cv_splits.py"
        )
    return "portable paths"


check("cv_splits paths are Linux-safe", splits_need_regen, optional=True)

check(
    "train.py exists",
    lambda: (
        "FOUND"
        if os.path.exists("training_scripts/train.py")
        else (_ for _ in ()).throw(RuntimeError("File not found"))
    ),
)

print("\n" + "=" * 55)
print("  Environment Verification")
print("=" * 55)

all_required_passed = True
for name, optional, passed, detail in results:
    if passed:
        status = "PASS"
        marker = "+"
    elif optional:
        status = "WARN"
        marker = "!"
    else:
        status = "FAIL"
        marker = "x"
    print(f"  [{marker}] {status:<6} {name:<35} {detail}")
    if not passed and not optional:
        all_required_passed = False

print("=" * 55)
if all_required_passed:
    print("  Required checks passed. Ready for Fedora training setup.")
else:
    print("  Some required checks failed. Run install_deps.sh first.")
print("=" * 55 + "\n")
sys.exit(0 if all_required_passed else 1)
