"""
Test script: Compare improved watershed centroid detection on L2 images.
Generates a visual output showing detected centroids on the image.
"""
import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detection.generate_cell_crops_sam import get_watershed_centroids

# Test images — pick a few clumpy L2 images
TEST_IMAGES = [
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im003_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im001_1.jpg",
]

OUTPUT_DIR = r"c:\Open Source\leukiemea\temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue
        
    basename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*50}")
    print(f"Processing: {basename}")
    
    centroids, original_img = get_watershed_centroids(img_path)
    
    print(f"  Centroids found: {len(centroids)}")
    
    if original_img is not None:
        # Draw centroids on the image
        vis = original_img.copy()
        for pt in centroids:
            cx, cy = pt[0]
            cv2.circle(vis, (cx, cy), 8, (255, 0, 0), 2)
            cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)
        
        # Save    
        out_path = os.path.join(OUTPUT_DIR, f"watershed_improved_{basename}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path}")

print(f"\n{'='*50}")
print("Done!")
