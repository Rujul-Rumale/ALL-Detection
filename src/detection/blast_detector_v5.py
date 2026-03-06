"""
Blast Detector V5 - Python Port
Combines K-Means segmentation with L1-tuned classification
Port of: test/blast_detector_v5.m
"""

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, binary_opening
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
import json
import argparse


# === PARAMETERS ===
RESIZE_FACTOR = 0.5
MIN_NUC_AREA = 500
BLAST_SCORE_CUTOFF = 3.2
ECCENTRICITY_THRESH = 0.85


def detect_blasts(image_path, visualize=False, output_json=None, return_crops=False, return_all_cells=False):
    """
    Detect blast cells in a blood smear image.
    
    Args:
        image_path: Path to input image
        visualize: If True, display annotated result
        output_json: If provided, save detections to this JSON path
        return_crops: If True, include cropped cell images in results
        return_all_cells: If True, include normal cells too (for UI display)
    
    Returns:
        dict with 'blasts', 'normal', 'total' counts, 'detections' list, and optionally 'annotated_image'
    """
    
    # Load and resize
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    
    print(f"Processing: {image_path}")
    
    # =========================================================================
    # SEGMENTATION (K-Means on L*a*b*)
    # =========================================================================
    
    lab = rgb2lab(img_rgb)
    ab = lab[:, :, 1:3].reshape(-1, 2)
    
    # K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)
    labels = kmeans.fit_predict(ab).reshape(rows, cols)
    
    # Find nucleus cluster (darkest in L channel)
    L = lab[:, :, 0]
    avg_L = [L[labels == i].mean() for i in range(3)]
    nuc_cluster = np.argmin(avg_L)
    
    # Create nucleus mask
    mask = (labels == nuc_cluster).astype(np.uint8)
    
    # Morphological cleanup
    mask = binary_opening(mask, disk(2))
    mask = binary_fill_holes(mask)
    
    # Remove small objects
    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area < MIN_NUC_AREA:
            mask[labeled_mask == region.label] = 0
    
    # =========================================================================
    # FEATURE EXTRACTION & CLASSIFICATION
    # =========================================================================
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    detections = []
    blast_count = 0
    normal_count = 0
    
    print("\n--- Nucleus Analysis (L1 Blast Detection) ---")
    print("ID\tArea\tCirc.\tEcc.\tHom.\tSCORE\tClass")
    print("-" * 60)
    
    for i, region in enumerate(regions):
        area = region.area
        perimeter = region.perimeter
        ecc = region.eccentricity
        
        # Bounding box (minr, minc, maxr, maxc)
        minr, minc, maxr, maxc = region.bbox
        
        # Circularity
        circ = (4 * np.pi * area) / (perimeter**2 + 1e-6)
        
        # Texture (GLCM Homogeneity)
        roi_gray = gray[minr:maxr, minc:maxc]
        try:
            # Quantize to 16 levels
            roi_quant = (roi_gray // 16).astype(np.uint8)
            glcm = graycomatrix(roi_quant, distances=[1], angles=[0], 
                               levels=16, symmetric=True, normed=True)
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        except:
            homogeneity = 0.5
        
        # L1 Scoring
        s_area = min(area / 1500, 1.2)
        s_circ = circ
        s_tex = homogeneity
        
        total_score = (s_area * 1.0) + (s_circ * 1.5) + (s_tex * 1.2)
        
        # Classification
        if ecc > ECCENTRICITY_THRESH:
            is_blast = False
            cls_str = "Debris"
        elif total_score > BLAST_SCORE_CUTOFF:
            is_blast = True
            cls_str = "BLAST (L1)"
            blast_count += 1
        else:
            is_blast = False
            cls_str = "Normal"
            normal_count += 1
        
        print(f"{i+1}\t{area}\t{circ:.2f}\t{ecc:.2f}\t{homogeneity:.2f}\t{total_score:.2f}\t{cls_str}")
        
        # Include cell if it's a blast OR if we want all cells for UI
        if is_blast or return_all_cells:
            detection = {
                'id': i + 1,
                'bbox': [int(minc), int(minr), int(maxc - minc), int(maxr - minr)],
                'area': int(area),
                'circularity': round(circ, 3),
                'eccentricity': round(ecc, 3),
                'homogeneity': round(homogeneity, 3),
                'score': round(total_score, 2),
                'classification': cls_str,
                'is_blast': is_blast
            }
            
            # Crop cell image if requested
            if return_crops:
                # Add proportional padding (50% of bounding box) to capture cytoplasm
                # This is critical for TFLite model to see N:C ratio context
                w_box = maxc - minc
                h_box = maxr - minr
                pad = int(max(w_box, h_box) * 0.5)
                r1 = max(0, minr - pad)
                r2 = min(rows, maxr + pad)
                c1 = max(0, minc - pad)
                c2 = min(cols, maxc + pad)
                cell_crop = img_rgb[r1:r2, c1:c2].copy()
                detection['crop'] = cell_crop  # numpy array
            
            detections.append(detection)
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"Total Cells: {len(regions)}")
    print(f"Suspected Blasts: {blast_count}")
    print(f"Normal WBCs: {normal_count}")
    
    # Create annotated image for UI display
    annotated_img = img_rgb.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        color = (255, 0, 0) if det.get('is_blast', False) else (0, 255, 0)  # Red for blast, Green for normal
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
    
    result = {
        'image': image_path,
        'total_cells': len(regions),
        'blast_count': blast_count,
        'normal_count': normal_count,
        'detections': detections,
        'annotated_image': annotated_img  # RGB numpy array with bounding boxes
    }
    
    # Save JSON output (removing non-serializable objects)
    if output_json:
        # Create a serializable copy of the result
        serializable_result = result.copy()
        if 'annotated_image' in serializable_result:
            del serializable_result['annotated_image']
        
        # Also clean up detections if they have crops (numpy arrays)
        serializable_result['detections'] = []
        for det in result['detections']:
            clean_det = det.copy()
            if 'crop' in clean_det:
                del clean_det['crop']
            serializable_result['detections'].append(clean_det)

        with open(output_json, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        print(f"\nDetections saved to: {output_json}")
    
    # Visualization
    if visualize:
        vis_img = img.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis_img, f"BLAST {det['score']:.1f}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.imshow('Blast Detection V5', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ALL Blast Detector V5')
    parser.add_argument('image', help='Path to blood smear image')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Display annotated result')
    parser.add_argument('--output', '-o', help='Output JSON path')
    
    args = parser.parse_args()
    
    result = detect_blasts(args.image, visualize=args.visualize, output_json=args.output)
    
    if result['blast_count'] > 0:
        print(f"\n⚠️  ALERT: {result['blast_count']} Suspected Blast Cells Detected!")
    else:
        print("\n✓ No obvious blasts detected.")
