#!/usr/bin/env python3
"""
test_tflite_inference.py
========================
Standalone sanity-test for all 6 TFLite models.
Run on Pi immediately after file transfer to confirm everything loads.

No project imports required — fully self-contained.

Usage:
  python3 test_tflite_inference.py

Expected output (per model):
  mnv3l_fold1_best.tflite | [1,3,320,320] float32 | [1,2] float32 | P(ALL)=0.xxx P(HEM)=0.xxx | PASS
"""

import os
import sys
import numpy as np

# Locate self and models/ dir relative to this script
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models", "tflite_final")

# All 6 model files (hardcoded relative to project root)
TFLITE_MODELS = [
    os.path.join(_MODELS_DIR, "effb0_fold1_best.tflite"),
    os.path.join(_MODELS_DIR, "effb0_fold2_best.tflite"),
    os.path.join(_MODELS_DIR, "effb0_fold3_best.tflite"),
    os.path.join(_MODELS_DIR, "mnv3l_fold1_best.tflite"),
    os.path.join(_MODELS_DIR, "mnv3l_fold2_best.tflite"),
    os.path.join(_MODELS_DIR, "mnv3l_fold3_best.tflite"),
]

# ── tflite-runtime import ─────────────────────────────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using tensorflow.lite.Interpreter (tflite_runtime not found)")
    except ImportError:
        print("[ERROR] Neither tflite_runtime nor tensorflow found.")
        print("        Install with: pip install tflite-runtime")
        sys.exit(1)

# ── ImageNet normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def make_random_input(shape: tuple) -> np.ndarray:
    """
    Random float32 tensor of `shape`, ImageNet-normalised (NCHW).
    shape is typically (1, 3, 320, 320).
    """
    rng     = np.random.default_rng(seed=42)
    c, h, w = shape[1], shape[2], shape[3]
    img     = rng.random((h, w, c), dtype=np.float32)
    img     = (img - IMAGENET_MEAN) / IMAGENET_STD
    img     = np.transpose(img, (2, 0, 1))              # HWC → CHW
    return np.expand_dims(img, axis=0).astype(np.float32)


def test_model(model_path: str) -> bool:
    """
    Load one TFLite model, run one inference, print result.
    Returns True on success.
    """
    model_name = os.path.basename(model_path)

    if not os.path.exists(model_path):
        print(f"  {model_name:<35} | [FILE NOT FOUND]  FAIL")
        return False

    try:
        # ── a. Load interpreter ───────────────────────────────────────────────
        interp = Interpreter(model_path=model_path)

        # ── b. Allocate tensors ───────────────────────────────────────────────
        interp.allocate_tensors()

        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        in_shape  = tuple(in_det["shape"])
        out_shape = tuple(out_det["shape"])
        in_dtype  = in_det["dtype"]

        # ── c. Random input ───────────────────────────────────────────────────
        input_tensor = make_random_input(in_shape)

        # Handle quantized input boundary (should not occur with float32 I/O,
        # but guard in case a non-standard model is loaded)
        if in_dtype == np.int8:
            scale, zero_point = in_det["quantization"]
            if scale != 0:
                input_tensor = (input_tensor / scale + zero_point).clip(-128, 127).astype(np.int8)
            else:
                input_tensor = input_tensor.astype(np.int8)

        # ── d. Run inference ──────────────────────────────────────────────────
        interp.set_tensor(in_det["index"], input_tensor)
        interp.invoke()

        # ── e. Get output ─────────────────────────────────────────────────────
        output = interp.get_tensor(out_det["index"]).astype(np.float32)

        # ── f. Softmax ────────────────────────────────────────────────────────
        probs   = softmax(output)[0]          # shape (2,)
        p_all   = float(probs[0])             # class 0 = ALL (blast)
        p_hem   = float(probs[1])             # class 1 = HEM (normal)

        # ── g. Print result ───────────────────────────────────────────────────
        in_str  = f"{in_shape}"
        out_str = f"{out_shape}"
        print(
            f"  {model_name:<35} | "
            f"{in_str} {in_det['dtype'].__name__} | "
            f"{out_str} {out_det['dtype'].__name__} | "
            f"P(ALL)={p_all:.3f} P(HEM)={p_hem:.3f} | "
            f"PASS"
        )
        return True

    except Exception as exc:
        print(f"  {model_name:<35} | FAIL — {exc}")
        return False


def main() -> None:
    print(f"\n{'='*80}")
    print(f"  TFLite Sanity Check — {len(TFLITE_MODELS)} models")
    print(f"  Models dir: {_MODELS_DIR}")
    print(f"{'='*80}\n")
    print(
        f"  {'Model':<35}   {'Input':^25}   {'Output':^17}   "
        f"{'Probabilities':^25}   Status"
    )
    print(f"  {'─'*35}   {'─'*25}   {'─'*17}   {'─'*25}   {'─'*6}")

    passed = 0
    for model_path in TFLITE_MODELS:
        ok = test_model(model_path)
        if ok:
            passed += 1

    print(f"\n{'='*80}")
    print(f"  Result: {passed}/{len(TFLITE_MODELS)} models passed.")
    if passed < len(TFLITE_MODELS):
        print(f"  Check that convert_to_tflite.py completed without errors.")
    print(f"{'='*80}\n")

    sys.exit(0 if passed == len(TFLITE_MODELS) else 1)


if __name__ == "__main__":
    main()
