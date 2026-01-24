"""
Demo: Test segmentation pipeline on ALL-IDB1 image
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.wbc_segmenter import segment_image
from utils.visualization import visualize_segmentation, visualize_cell_montage


def main():
    # Example: segment an ALL-IDB1 image
    # User should update this path to an actual image
    image_path = Path("ALL_IDB/ALL_IDB Dataset/L1/Im003_1.tif")
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path in this script.")
        return
    
    print(f"Processing: {image_path}")
    
    # Segment image
    results = segment_image(
        image_path,
        output_dir=image_path.parent / "extracted_cells",
        visualize=True
    )
    
    print(f"\nResults:")
    print(f"  Detected {results['count']} WBCs")
    if 'saved_to' in results:
        print(f"  Cells saved to: {results['saved_to']}")
    
    # Visualize
    # Load original for visualization
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    
    # 6-panel segmentation view
    visualize_segmentation(img, results)
    
    # Cell montage
    if results['cells']:
        visualize_cell_montage(results['cells'])
    
    plt.show()


if __name__ == '__main__':
    main()
