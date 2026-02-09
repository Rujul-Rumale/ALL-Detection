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
    # Allow user to select an image
    from tkinter import Tk, filedialog
    
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file dialog starting in ALL_IDB directory
    initial_dir = Path("ALL_IDB/ALL_IDB Dataset")
    if not initial_dir.exists():
        initial_dir = Path(".")
    
    print("Opening file dialog...")
    file_path = filedialog.askopenfilename(
        title="Select an ALL-IDB Image",
        initialdir=str(initial_dir),
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.tif *.tiff *.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()  # Clean up
    
    if not file_path:
        print("No file selected. Exiting.")
        return
    
    image_path = Path(file_path)
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
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
