import os
import sys
import time
from pathlib import Path

# Fix path before importing src
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.detection.demo_pipeline import DemoPipeline

def run_benchmarks():
    pipeline = DemoPipeline()
    # Sample image from L2
    img_path = r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im001_1.jpg"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    configs = [
        "EfficientNet-B0  | Fold 1",
        "MobileNetV3-Large | Fold 1",
        "EfficientNet-B0  | Ensemble",
        "MobileNetV3-Large | Ensemble"
    ]

    print("Starting Hardware Benchmarks...")
    print("=" * 50)

    for config in configs:
        printable_config = config.replace("│", "|")
        print(f"\nBenchmarking: {printable_config}")
        try:
            pipeline.load_classifier(config)
            # Warmup
            _ = pipeline.process_image(img_path, config)
            
            # Actual bench (3 runs)
            for i in range(3):
                start = time.time()
                results = pipeline.process_image(img_path, config)
                end = time.time()
                print(f"  Run {i+1}: {end-start:.3f}s total pipeline time. Cells found: {len(results['detections'])}")
        except Exception as e:
            print(f"  Error benchmarking {config}: {e}")

    print("\nBenchmarks complete. Checking log directory: outputs/pi_benchmarks")
    log_dir = Path("outputs/pi_benchmarks")
    if log_dir.exists():
        for f in log_dir.glob("*.csv"):
            print(f"Found log: {f.name}")
    else:
        print("Log directory not found.")

if __name__ == "__main__":
    run_benchmarks()
