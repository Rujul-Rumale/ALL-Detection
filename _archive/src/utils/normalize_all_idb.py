
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Create a symlink to src to allow imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from tqdm import tqdm
from preprocessing import normalize_staining, resize_with_padding, mask_background_kmeans, normalize_hist_match
from src.segmentation.wbc_segmenter import WBCSegmenter

def process_all_idb(input_dir, output_dir, reference_vectors):
    """
    Process ALL-IDB dataset:
    1. Segment WBCs from full images
    2. Normalize staining
    3. Pad and resize
    4. Save individual cells
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    segmenter = WBCSegmenter(padding=20)
    
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.tif'))
    print(f"Found {len(images)} images in {input_dir}")
    
    if len(images) == 0:
        print("No images found. Check format (.jpg, .tif)")
        return

    # Load template for histogram matching fallback
    # Assuming C-NMC structure is consistent
    template_path = Path(r"c:\Open Source\leukiemea\C-NMC_Dataset\fold_0\fold_0\all\UID_11_10_1_all.bmp")
    if template_path.exists():
        template_img = cv2.imread(str(template_path))
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        print("Loaded template image for histogram matching.")
    else:
        print("Warning: Template image not found!")
        template_img = None

    count = 0
    
    for img_file in tqdm(images):
        try:
            # 1. Read Image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Segment
            # Resize for faster segmentation, but map back to original? 
            # For simplicity, we segment at full res or half res
            results = segmenter.segment(img)
            cells = results['cells']
            
            for i, cell in enumerate(cells):
                if cell.size == 0: continue
                
                # 3. New Pipeline: Mask -> Normalize (Hist Match)
                
                # a. Mask Background (K-Means)
                masked_cell = mask_background_kmeans(cell, k=2)
                
                # b. Normalize 
                # Use Histogram Matching as primary since Macenko failed
                if template_img is not None:
                     norm_cell = normalize_hist_match(masked_cell, template_img)
                else:
                    # Fallback to just masked if no template (shouldn't happen)
                    norm_cell = masked_cell
                
                # 4. Resize & Pad
                processed_cell = resize_with_padding(norm_cell, target_size=(224, 224))
                
                # 5. Save
                save_name = f"{img_file.stem}_cell_{i}.jpg"
                save_full_path = output_path / save_name
                
                # Convert back to BGR for OpenCV saving
                cv2.imwrite(str(save_full_path), cv2.cvtColor(processed_cell, cv2.COLOR_RGB2BGR))
                count += 1
                
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print(f"Processing complete. Saved {count} single-cell images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input directory (ALL-IDB1)")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--vectors', default='data/stain_vectors.npz', help="Path to stain vectors")
    args = parser.parse_args()
    
    process_all_idb(args.input, args.output, args.vectors)
