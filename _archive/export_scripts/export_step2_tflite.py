import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
onnx_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc.onnx")

def export_tflite():
    print("Converting ONNX to TFLite via ONNX-TF...")
    
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
    export_tflite()
