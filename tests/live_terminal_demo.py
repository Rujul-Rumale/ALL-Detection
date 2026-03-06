import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detection.blast_detector_v5 import detect_blasts
from src.detection.stage1_screening import ALLScreener

def run_demo(image_path):
    print("="*60)
    print("      EDGE-AI LEUKEMIA DETECTION SYSTEM - LIVE DEMO")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Hardware: Raspberry Pi 5 / IoT Edge Node")
    print(f"Processing Image: {os.path.basename(image_path)}")
    print("-" * 60)

    # 1. Pipeline Initialization
    start_init = time.time()
    print("[1/3] Initializing Embedded AI Pipeline...")
    try:
        screener = ALLScreener()
        init_time = (time.time() - start_init) * 1000
        print(f"      -> Model Loaded: MobileNetV3 (TFLite)")
        print(f"      -> Optimization: Float16 Quantized")
        print(f"      -> Init Latency: {init_time:.1f}ms")
    except Exception as e:
        print(f"      -> Error: {e}")
        return

    # 2. Advanced Nucleus Analysis (L1 Scoring)
    print("\n[2/3] Running Nucleus Segmentation & Feature Analysis...")
    start_det = time.time()
    results_v5 = detect_blasts(image_path, visualize=False)
    det_time = (time.time() - start_det) * 1000
    
    # Summary of Analysis from V5 detector is already printed by its internal logic
    # But we add a stylized count summary here:
    print(f"      -> Performance: {det_time:.1f}ms")
    print(f"      -> Cell Count: {results_v5['total_cells']}")

    # 3. Deep Learning Screening
    print("\n[3/3] Performing Stage-1 AI Screening...")
    start_screen = time.time()
    screening_result = screener.predict(image_path)
    screen_time = (time.time() - start_screen) * 1000

    status_str = "!!! ALL POSITIVE !!!" if screening_result['positive'] else "OK - HEALTHY"
    
    print(f"      -> Classification: {screening_result['classification']}")
    print(f"      -> Confidence: {screening_result['confidence']*100:.1f}%")
    print(f"      -> Latency: {screen_time:.1f}ms")

    # Final Diagnostic Report
    print("\n" + "="*60)
    print("                FINAL DIAGNOSTIC VERDICT")
    print("="*60)
    print(f"  RESULT:      {status_str}")
    print(f"  CONFIDENCE:  {screening_result['confidence']*100:.2f}%")
    print(f"  TOTAL TIME:  {init_time + det_time + screen_time:.1f} ms")
    print("="*60)
    print("             Clinical verification recommended.")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Use a default sample if none provided
    sample = r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L1\Im064_0.jpg"
    if len(sys.argv) > 1:
        sample = sys.argv[1]
    
    if os.path.exists(sample):
        run_demo(sample)
    else:
        print(f"Error: Could not find image at {sample}")
