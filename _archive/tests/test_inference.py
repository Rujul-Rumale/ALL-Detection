import sys
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    sys.path.insert(0, 'c:/Open Source/leukiemea')
    from src.detection.demo_pipeline import DemoPipeline
    
    dp = DemoPipeline()
    res = dp.process_image(r'c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im064_0.jpg', 'MobileNetV3-Small (Weighted)')
    print(f"SUCCESS! Found {res['total_cells']} cells, {res['blast_count']} blasts.")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
