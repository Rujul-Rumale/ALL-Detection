import os
import torch
import torch.nn as nn
from torchvision import models

# ============ CONFIG ============
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
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

if __name__ == "__main__":
    export()
