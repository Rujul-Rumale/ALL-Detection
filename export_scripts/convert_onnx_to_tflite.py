"""
convert_onnx_to_tflite.py
=========================
Converts 6 ONNX models (EfficientNet-B0 × 3 folds + MobileNetV3-Large × 3 folds)
to INT8-quantized TFLite files for Raspberry Pi 5 deployment.

Run from project root:
    python export_scripts/convert_onnx_to_tflite.py

Requirements:
    pip install onnx onnx-tf tensorflow
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "models" / "tflite_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_RES     = 320

MODELS = [
    {
        "onnx": PROJECT_ROOT / "outputs/local_effb0_final1/local_effb0_final1_fold1_20260316_220343_best.onnx",
        "out":  OUTPUT_DIR / "effb0_fold1_best.tflite",
        "arch": "effb0", "fold": 1,
    },
    {
        "onnx": PROJECT_ROOT / "outputs/local_effb0_final2/local_effb0_final2_fold2_20260317_010239_best.onnx",
        "out":  OUTPUT_DIR / "effb0_fold2_best.tflite",
        "arch": "effb0", "fold": 2,
    },
    {
        "onnx": PROJECT_ROOT / "outputs/local_effb0_final3/local_effb0_final3_fold3_20260317_035939_best.onnx",
        "out":  OUTPUT_DIR / "effb0_fold3_best.tflite",
        "arch": "effb0", "fold": 3,
    },
    {
        "onnx": PROJECT_ROOT / "outputs/local_mnv3l_final1/local_mnv3l_final1_fold1_20260315_152609_best.onnx",
        "out":  OUTPUT_DIR / "mnv3l_fold1_best.tflite",
        "arch": "mnv3l", "fold": 1,
    },
    {
        "onnx": PROJECT_ROOT / "outputs/local_mnv3l_final2/local_mnv3l_final2_fold2_20260315_173338_best.onnx",
        "out":  OUTPUT_DIR / "mnv3l_fold2_best.tflite",
        "arch": "mnv3l", "fold": 2,
    },
    {
        "onnx": PROJECT_ROOT / "outputs/local_mnv3l_final3/local_mnv3l_final3_fold3_20260315_194014_best.onnx",
        "out":  OUTPUT_DIR / "mnv3l_fold3_best.tflite",
        "arch": "mnv3l", "fold": 3,
    },
]


def make_representative_dataset(n=100):
    """Yields properly normalized float32 tensors in NCHW format."""
    rng = np.random.default_rng(42)
    def generator():
        for _ in range(n):
            img = rng.random((1, INPUT_RES, INPUT_RES, 3), dtype=np.float32)
            # ImageNet normalize
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            # NHWC → NCHW for the model's ONNX input
            img = img.transpose(0, 3, 1, 2).astype(np.float32)
            yield [img]
    return generator


def convert_one(cfg: dict):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_path = str(cfg["onnx"])
    out_path  = str(cfg["out"])
    arch      = cfg["arch"]
    fold      = cfg["fold"]

    print(f"\n{'='*60}")
    print(f"  Converting: {arch.upper()} Fold {fold}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  TFLite: {out_path}")
    print(f"{'='*60}")

    if not os.path.exists(onnx_path):
        print(f"  ✗  ONNX file not found — skipping.")
        return False

    # ── Step 1: ONNX → TF SavedModel ──────────────────────────────────────────
    print("  [1/3] Converting ONNX → TF SavedModel ...")
    tf.config.set_visible_devices([], 'GPU')  # force CPU for conversion stability
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    saved_model_dir = os.path.join(tempfile.mkdtemp(), f"{arch}_fold{fold}_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"     SavedModel → {saved_model_dir}")

    # ── Step 2: TFLiteConverter + INT8 quantization ────────────────────────────
    print("  [2/3] Converting SavedModel → INT8 TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations            = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset   = make_representative_dataset(n=100)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type     = tf.int8
    converter.inference_output_type    = tf.float32

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"  ✗  INT8 conversion failed ({e}). Falling back to float16 ...")
        converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter2.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter2.target_spec.supported_types = [tf.float16]
        tflite_model = converter2.convert()

    # ── Step 3: Save + verify ──────────────────────────────────────────────────
    print("  [3/3] Saving and verifying ...")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    # Verify by loading and checking input dtype
    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    in_dtype  = in_detail["dtype"].__name__ if hasattr(in_detail["dtype"], "__name__") else str(in_detail["dtype"])
    in_shape  = in_detail["shape"].tolist()

    print(f"  ✓  Saved: {out_path}")
    print(f"     Size:          {size_mb:.2f} MB")
    print(f"     Input dtype:   {in_dtype}  (expected: int8)")
    print(f"     Input shape:   {in_shape}")

    if "int8" not in in_dtype.lower():
        print(f"  ⚠  WARNING: input dtype is {in_dtype}, not int8. Check quantization.")

    # Cleanup temp SavedModel
    shutil.rmtree(os.path.dirname(saved_model_dir), ignore_errors=True)
    return True


def main():
    print("\n🔬  ONNX → INT8 TFLite Conversion Pipeline")
    print(f"    Output dir: {OUTPUT_DIR}\n")

    results = []
    for cfg in MODELS:
        success = convert_one(cfg)
        results.append((cfg["arch"], cfg["fold"], success))

    print(f"\n{'='*60}")
    print("  CONVERSION SUMMARY")
    print(f"{'='*60}")
    for arch, fold, ok in results:
        status = "✓  OK" if ok else "✗  FAILED"
        print(f"  {arch.upper():10s} Fold {fold}  →  {status}")

    ok_count = sum(1 for _, _, ok in results if ok)
    print(f"\n  {ok_count}/{len(results)} models converted successfully.")
    print(f"  TFLite files: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
