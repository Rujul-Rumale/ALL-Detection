"""
convert_to_tflite.py
====================
Converts the 6 trained ONNX models to INT8-weight TFLite format.

Pipeline per model:
  ONNX  →  TF SavedModel (via onnx-tf)
        →  TFLite INT8   (via TFLiteConverter)

Quantization settings:
  - Optimizations    : DEFAULT (INT8 weight quantization)
  - Representative   : 200 random float32 tensors [1,3,320,320], ImageNet norm
  - Supported ops    : TFLITE_BUILTINS_INT8
  - Input  type      : float32  (client passes normalised float32; weights INT8)
  - Output type      : float32

Output: models/tflite_final/<arch>_fold<n>_best.tflite

Requirements:
  pip install onnx onnx-tf tensorflow numpy
"""

import os
import sys
import shutil
import tempfile

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR      = os.path.join(PROJECT_ROOT, "models", "tflite_final")
TMP_DIR      = os.path.join(PROJECT_ROOT, "_tmp_savedmodels")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = [
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "effb0_exp_meth",
            "effb0_exp_meth_fold1_20260320_000256_best.onnx"
        ),
        "arch": "effb0", "fold": 1,
        "tflite": "effb0_fold1_best.tflite",
    },
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "effb0_exp_meth1",
            "effb0_exp_meth1_fold2_20260321_005825_best.onnx"
        ),
        "arch": "effb0", "fold": 2,
        "tflite": "effb0_fold2_best.tflite",
    },
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "effb0_exp_meth2",
            "effb0_exp_meth2_fold3_20260321_023825_best.onnx"
        ),
        "arch": "effb0", "fold": 3,
        "tflite": "effb0_fold3_best.tflite",
    },
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "mnv3l_exp_meth",
            "mnv3l_exp_meth_fold1_20260319_145131_best.onnx"
        ),
        "arch": "mnv3l", "fold": 1,
        "tflite": "mnv3l_fold1_best.tflite",
    },
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "mnv3l_exp_meth1",
            "mnv3l_exp_meth1_fold2_20260319_165823_best.onnx"
        ),
        "arch": "mnv3l", "fold": 2,
        "tflite": "mnv3l_fold2_best.tflite",
    },
    {
        "onnx": os.path.join(
            PROJECT_ROOT, "outputs", "mnv3l_exp_meth2",
            "mnv3l_exp_meth2_fold3_20260319_201230_best.onnx"
        ),
        "arch": "mnv3l", "fold": 3,
        "tflite": "mnv3l_fold3_best.tflite",
    },
]

# ── ImageNet normalisation constants ──────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SHAPE   = (1, 3, 320, 320)   # NCHW — matches training and ONNX graph
N_REPR        = 200                 # representative dataset size for INT8 cal


def representative_dataset_gen():
    """
    Yields 200 random float32 tensors shaped [1, 3, 320, 320].
    Values are ImageNet-normalised (channel-wise mean subtracted, std divided).
    This covers the expected input distribution seen during deployment.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(N_REPR):
        # Simulate a normalised image in [0,1] pixel range, then standardise
        img = rng.random((1, 320, 320, 3), dtype=np.float32)   # NHWC for TF
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # TFLite converter expects a list of input arrays per call
        yield [img.astype(np.float32)]


def onnx_to_savedmodel(onnx_path: str, savedmodel_dir: str) -> str:
    """
    Convert ONNX → TF SavedModel using onnx-tf.
    Returns the SavedModel directory path.
    """
    import onnx
    from onnx_tf.backend import prepare

    print(f"  Loading ONNX: {os.path.basename(onnx_path)}")
    model = onnx.load(onnx_path)

    print(f"  Preparing TF backend …")
    tf_rep = prepare(model)

    if os.path.exists(savedmodel_dir):
        shutil.rmtree(savedmodel_dir)

    print(f"  Exporting SavedModel → {savedmodel_dir}")
    tf_rep.export_graph(savedmodel_dir)
    return savedmodel_dir


def savedmodel_to_tflite_int8(savedmodel_dir: str, tflite_out: str) -> None:
    """
    Convert TF SavedModel → TFLite with INT8 weight quantization.

    Key settings (per task specification):
      - float32 input / float32 output  (INT8 weights only)
      - TFLITE_BUILTINS_INT8 ops
      - 200-sample representative dataset for calibration
    """
    import tensorflow as tf

    print(f"  Building TFLiteConverter from SavedModel …")
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)

    # ── Optimisation: default = INT8 weight quantization ──────────────────────
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ── Representative dataset for calibration ────────────────────────────────
    # Required for full INT8 quantization of activations.
    # Input generator yields NHWC tensors — the ONNX-TF graph expects this
    # layout because onnx-tf transposes NCHW → NHWC when converting.
    converter.representative_dataset = representative_dataset_gen

    # ── Op set: INT8 builtins only ─────────────────────────────────────────────
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # ── Keep float32 at I/O boundary ──────────────────────────────────────────
    # inference_input_type = float32: client passes normalised float32 arrays
    # without needing to handle quantization scaling themselves.
    # inference_output_type = float32: raw logits come back as float32.
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    print(f"  Quantizing … (this may take ~30–90 s per model)")
    tflite_model = converter.convert()

    with open(tflite_out, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_out) / (1024 ** 2)
    print(f"  Saved: {tflite_out}  ({size_mb:.2f} MB)")


def verify_tflite(tflite_path: str) -> None:
    """
    Reload the .tflite file and confirm:
      - File is loadable
      - Input dtype is float32
      - Output dtype is float32
      - A single forward pass succeeds
    """
    import tensorflow as tf

    print(f"  Verifying: {os.path.basename(tflite_path)}")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    assert in_det["dtype"]  == np.float32, \
        f"Expected float32 input, got {in_det['dtype']}"
    assert out_det["dtype"] == np.float32, \
        f"Expected float32 output, got {out_det['dtype']}"

    # Quick forward pass with a zero-filled tensor
    dummy = np.zeros(in_det["shape"], dtype=np.float32)
    interp.set_tensor(in_det["index"], dummy)
    interp.invoke()
    out = interp.get_tensor(out_det["index"])

    size_mb = os.path.getsize(tflite_path) / (1024 ** 2)
    print(f"    Input dtype : {in_det['dtype']}  shape={in_det['shape']}")
    print(f"    Output dtype: {out_det['dtype']} shape={out_det['shape']}")
    print(f"    Output sample: {out}")
    print(f"    File size   : {size_mb:.2f} MB")
    print(f"    PASS ✓")


def convert_one(cfg: dict) -> bool:
    """
    Full conversion pipeline for one model.
    Returns True on success, False on failure.
    """
    onnx_path    = cfg["onnx"]
    tflite_name  = cfg["tflite"]
    tflite_out   = os.path.join(OUT_DIR, tflite_name)
    savedmodel_d = os.path.join(TMP_DIR, tflite_name.replace(".tflite", "_sm"))

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {cfg['arch'].upper()} Fold {cfg['fold']}  →  {tflite_name}")
    print(sep)

    if not os.path.exists(onnx_path):
        print(f"  [ERROR] ONNX file not found: {onnx_path}")
        return False

    try:
        onnx_to_savedmodel(onnx_path, savedmodel_d)
        savedmodel_to_tflite_int8(savedmodel_d, tflite_out)
        verify_tflite(tflite_out)
        return True
    except Exception as e:
        print(f"  [FAIL] {cfg['arch']} fold {cfg['fold']}: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ONNX → TFLite INT8 Converter")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Representative samples: {N_REPR}")
    print(f"{'='*60}")

    results = []
    for cfg in MODELS:
        ok = convert_one(cfg)
        results.append((cfg["tflite"], ok))

    # Cleanup temp SavedModel directories
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
        print(f"\nCleaned up temp dir: {TMP_DIR}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    print(f"\n  {passed}/{len(results)} models converted successfully.")
    print(f"  TFLite files: {OUT_DIR}")

    sys.exit(0 if passed == len(results) else 1)
