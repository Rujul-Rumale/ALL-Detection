"""Test improved filters on a wider range of L2 images."""
import os, sys, cv2, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detection.generate_cell_crops_sam import get_watershed_centroids

# Pick images with likely debris/smeared cells
TEST_IMAGES = [
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im003_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im001_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im004_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im009_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im011_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im015_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im025_1.jpg",
]
OUTPUT_DIR = r"c:\Open Source\leukiemea\temp_outputs"

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"Skipping (not found): {os.path.basename(img_path)}")
        continue
    basename = os.path.splitext(os.path.basename(img_path))[0]
    centroids, original_img = get_watershed_centroids(img_path)
    print(f"{basename}: {len(centroids)} WBCs")
    if original_img is not None:
        vis = original_img.copy()
        for pt in centroids:
            cx, cy = pt[0]
            cv2.circle(vis, (cx, cy), 8, (255, 0, 0), 2)
            cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"filtered_{basename}.jpg"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print("Done!")
