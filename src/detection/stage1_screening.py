"""
Stage 1: ALL Screening using TFLite
Uses Peter Moss pretrained model for image-level classification
"""

import numpy as np
import cv2
import os

# Use tflite_runtime for Pi compatibility (falls back to TensorFlow if unavailable)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class ALLScreener:
    """
    Screen blood smear images for Acute Lymphoblastic Leukemia.
    Uses Peter Moss pretrained TFLite model.
    
    Input: Any size RGB image (will be resized to 100x100)
    Output: Classification (ALL/Healthy) with confidence
    """
    
    def __init__(self, model_path=None):
        """
        Initialize screener with TFLite model.
        
        Args:
            model_path: Path to .tflite model. If None, uses default location.
        """
        if model_path is None:
            # Default: assume model is in project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(
                base_dir, 
                "ALL-Arduino-Nano-33-BLE-Sense-Classifier",
                "model",
                "all_nano_33_ble_sense.tflite"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Expected input shape (batch, height, width, channels)
        self.input_shape = self.input_details[0]['shape'][1:3]  # (100, 100)
        
    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image (OpenCV format) or path to image
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Apply Histogram Equalization to fix contrast (Matches augmentation.py logic)
        if len(image.shape) == 3:
            img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
            image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

        # Resize to model input size
        image = cv2.resize(image, tuple(self.input_shape))
        
        # Check expected input type
        input_dtype = self.input_details[0]['dtype']
        
        if input_dtype == np.int8:
            # Quantized model: scale to INT8 range [-128, 127]
            # Get quantization params
            input_scale = self.input_details[0]['quantization'][0]
            input_zero_point = self.input_details[0]['quantization'][1]
            
            # Scale image to [0, 255] then quantize
            image = image.astype(np.float32) / 255.0
            image = (image / input_scale) + input_zero_point
            image = np.clip(image, -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            # UINT8 quantized model
            image = image.astype(np.uint8)
        else:
            # Float model: normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def predict(self, image):
        """
        Classify image as ALL-positive or Healthy.
        
        Args:
            image: BGR image or path to image
            
        Returns:
            dict with 'positive' (bool), 'confidence' (float), 'raw_output' (list)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        # DEBUG: Raw output
        # print(f"DEBUG: Raw Output Tensor: {output_tensor}")
        
        output = output_tensor # Keep original for reference
        
        # Dequantize if needed
        output_dtype = self.output_details[0]['dtype']
        if output_dtype in [np.int8, np.uint8]:
            output_scale = self.output_details[0]['quantization'][0]
            output_zero_point = self.output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Apply softmax - REMOVED: Model output is already probabilities (quantized 0.0-1.0)
        probabilities = output[0]
        
        # Interpret output
        # Model outputs 2 classes: [Healthy, ALL] (inferred from output saturation)
        # Class 0: Healthy, Class 1: ALL
        healthy_prob = float(probabilities[0])
        all_prob = float(probabilities[1])
        
        is_positive = all_prob > healthy_prob
        confidence = max(all_prob, healthy_prob)
        
        return {
            'positive': is_positive,
            'classification': 'ALL' if is_positive else 'Healthy',
            'confidence': round(confidence, 4),
            'all_probability': round(all_prob, 4),
            'healthy_probability': round(healthy_prob, 4)
        }
    
    def screen_batch(self, image_paths):
        """
        Screen multiple images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of prediction dicts
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'image': path,
                    'error': str(e)
                })
        return results


def main():
    """CLI interface for screening."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='ALL Screening (Stage 1)')
    parser.add_argument('image', help='Path to blood smear image')
    parser.add_argument('--model', '-m', help='Path to TFLite model')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    screener = ALLScreener(model_path=args.model)
    result = screener.predict(args.image)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "⚠️ ALL POSITIVE" if result['positive'] else "✓ Healthy"
        print(f"\nScreening Result: {status}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"ALL Probability: {result['all_probability']:.4f}")
        print(f"Healthy Probability: {result['healthy_probability']:.4f}")


if __name__ == "__main__":
    main()
