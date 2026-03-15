import os
import torch
import torch.nn as nn
from torchvision import models

# ============ CONFIG ============
from pathlib import Path
OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "models")
IMG_SIZE = 224
DEVICE = torch.device("cpu") # Export on CPU

def build_model():
    """MobileNetV3-Small with custom 2-class head."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),  # 2 classes: ALL, HEM
    )
    return model

def export():
    print("Loading PyTorch Checkpoint...")
    model = build_model().to(DEVICE)
    checkpoint_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc_best.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_val_recall = checkpoint.get('best_val_recall', 'Unknown')
    print(f"Model loaded. Best Val Recall: {best_val_recall}")

    # ===== EXPORT TO ONNX =====
    print("Exporting model to ONNX...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc.onnx")
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model saved: {onnx_path} ({onnx_size:.2f} MB)")

    # ===== CONVERT ONNX TO TFLITE =====
    print("Converting ONNX to TFLite via ONNX-TF...")
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    
    # Workaround for TF protobuf compatibility issues if any
    tf.config.set_visible_devices([], 'GPU') # Force CPU for conversion
    
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_saved_model_path = os.path.join(OUTPUT_DIR, "tf_saved_model")
    tf_rep.export_graph(tf_saved_model_path)

    print("Converting SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
    # Important: Enable default optimizations for quantization (weight quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"TFLite model saved: {tflite_path} ({tflite_size:.2f} MB)")
    print("Export Complete! Model is ready for Raspberry Pi 5.")

if __name__ == "__main__":
    export()
