
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.preprocessing import normalize_staining, resize_with_padding, normalize_hist_match
from src.segmentation.wbc_segmenter import WBCSegmenter

def verify_pipeline():
    # 1. Setup paths
    # ALL-IDB image (Test Domain) - Pre-segmented
    # Using cell_008.jpg as it has a reasonable file size (likely good quality)
    all_idb_path = r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L1\extracted_cells\cell_008.jpg"
    # C-NMC image (Training Domain - Target)
    cnmc_path = r"c:\Open Source\leukiemea\C-NMC_Dataset\fold_0\fold_0\all\UID_11_10_1_all.bmp"
    # Reference vectors
    vectors_path = r"data/stain_vectors.npz"

    output_file = "normalization_proof_v6.png"

    # 2. Extract a cell from ALL-IDB
    print(f"Reading {all_idb_path}...")
    raw_cell = cv2.imread(all_idb_path)
    if raw_cell is None:
        print(f"Failed to load {all_idb_path}")
        return
    raw_cell = cv2.cvtColor(raw_cell, cv2.COLOR_BGR2RGB)
    
    # No segmentation needed for pre-segmented cells
    print("Using pre-segmented cell directly.")

    # Load C-NMC Target (Template)
    print(f"Reading template {cnmc_path}...")
    template_img = cv2.imread(cnmc_path)
    if template_img is None:
        print(f"Failed to load template {cnmc_path}")
        return
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

    # 3. Apply Background Masking (Smart K-Means)
    print("Applying Background Masking (K-Means K=2)...")
    
    # Use K-Means to separate Cell (Nucleus+Cyto) vs Background
    # Convert to LAB for better color separation
    lab_cell = cv2.cvtColor(raw_cell, cv2.COLOR_RGB2LAB)
    pixel_values = lab_cell.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # K=2: Cluster 0 (Background?), Cluster 1 (Cell?)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    try:
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Determine which cluster is background (higher L value)
        # centers[:, 0] is the L channel
        l_centers = centers[:, 0]
        bg_cluster = np.argmax(l_centers)
        cell_cluster = np.argmin(l_centers)
        
        # Create mask for the cell cluster
        labels = labels.flatten()
        mask = (labels == cell_cluster).astype(np.uint8) * 255
        mask = mask.reshape(raw_cell.shape[:2])
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
    except Exception as e:
        print(f"K-Means masking failed: {e}. Falling back to crude threshold.")
        gray = cv2.cvtColor(raw_cell, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # Apply mask
    masked_cell = cv2.bitwise_and(raw_cell, raw_cell, mask=mask)
    
    # Now try histogram matching on the MASKED cell vs MASKED template
    norm_cell = normalize_hist_match(masked_cell, template_img)
    
    # 5. Create Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw
    axes[0].imshow(raw_cell)
    axes[0].set_title("1. Original ALL-IDB Cell\n(Light Background)", fontsize=12)
    axes[0].axis('off')
    
    # Masked + Normalized
    axes[1].imshow(norm_cell)
    axes[1].set_title("2. Processed Cell\n(K-Means Mask + Hist Match)", fontsize=12, color='green', fontweight='bold')
    axes[1].axis('off')
    
    # Target
    axes[2].imshow(template_img)
    axes[2].set_title("3. Target C-NMC Domain\n(Template)", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Verification image saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    verify_pipeline()
