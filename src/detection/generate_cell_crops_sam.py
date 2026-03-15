import os
import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import time
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROJECT_ROOT

DATASET_DIR = str(PROJECT_ROOT / "ALL_IDB" / "ALL_IDB Dataset" / "L2")  # Testing L2 first
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "processed_crops_sam")
RAW_CROP_SIZE = 176  # Size to extract from original image (to ensure wide margins)
CROP_SIZE = 128      # Final output dimensions
RESIZE_FACTOR = 0.5  # For faster processing
MIN_NUC_AREA = 50    # Min area (at resize scale) — lowered to catch small lymphocytes
MAX_NUC_AREA = 8000  # Max area (at resize scale) — rejects massive merged blobs
MIN_CIRCULARITY = 0.40  # Debris rejection — nuclei are roughly circular
MIN_SOLIDITY = 0.50  # Debris rejection — deformed spread debris has low solidity
MAX_MEAN_L = 65      # Debris rejection — nuclei are deeply stained (dark in LAB L)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# IMPROVED WATERSHED CENTROID FINDER
# Hybrid approach:
#   - LAB K-Means to identify WBC nuclei (not RBCs)
#   - Paper pipeline for splitting clumps:
#     Opening → EDT → peak_local_max → Watershed
# ==========================================
def get_watershed_centroids(image_path):
    """
    Pinpoints WBC nuclei to be used as point prompts for SAM.
    
    Uses a hybrid pipeline:
      1. LAB K-Means to identify the WBC nucleus cluster (darkest L)
      2. Morphological opening + closing to clean the mask
      3. Euclidean Distance Transform
      4. peak_local_max to find individual nuclei peaks (splits clumps)
      5. Marker-controlled watershed to label regions
      6. Extract centroids from labeled regions with size filtering
    """
    # --- Read image (handles unicode paths on Windows) ---
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], None

    # Keep original image for SAM and cropping
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Scale down for faster processing
    img_small = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    rows, cols = img_small.shape[:2]

    # ── Step 1: LAB K-Means to find WBC nuclei ──
    # This reliably separates WBCs from RBCs (darkest cluster in L channel)
    lab = rgb2lab(img_rgb_small)
    ab = lab[:, :, 1:3].reshape(-1, 2)
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)
    labels = kmeans.fit_predict(ab).reshape(rows, cols)

    L = lab[:, :, 0]
    avg_L = [L[labels == i].mean() for i in range(3)]
    nuc_cluster = np.argmin(avg_L)  # Darkest L channel is nucleus
    mask = (labels == nuc_cluster).astype(np.uint8) * 255

    # ── Step 2: Morphological Cleanup ──
    # Opening removes small noise artifacts
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing fills small holes inside nuclei
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # ── Step 3: Euclidean Distance Transform ──
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # ── Step 4: peak_local_max to find individual nuclei ──
    # Key improvement over old code: each nucleus in a clump gets its own peak
    # even when the binary mask merges them into one blob
    coords = peak_local_max(
        dist_transform,
        min_distance=20,      # min pixel distance between peaks — prevents
                              # multiple centroids within one spread-out nucleus
        threshold_abs=5.0,    # min distance value — rejects shallow debris peaks
        labels=cleaned,       # only inside foreground
    )

    if len(coords) == 0:
        return [], original_img

    # ── Step 5: Marker-controlled Watershed ──
    # Create markers from peaks
    markers = np.zeros((rows, cols), dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 2  # labels start at 2 (1 = background)

    # Dilate markers so watershed has seed regions
    markers = cv2.dilate(
        markers.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(np.int32)
    
    # Define background and unknown regions
    sure_bg = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, cleaned)
    
    markers[sure_bg == 0] = 1       # definite background
    markers[unknown > 0] = 0        # let watershed decide

    # Watershed needs a 3-channel image
    img_ws = img_rgb_small.copy()
    cv2.watershed(img_ws, markers)

    # ── Step 6: Extract centroids — separate cells from debris ──
    centroids = []
    debris_centroids = []  # Debris gets tracked, not discarded
    unique_labels = np.unique(markers)
    L_channel = lab[:, :, 0]  # LAB lightness for intensity check
    
    for label in unique_labels:
        if label <= 1:  # skip background and boundary
            continue
            
        region_mask = (markers == label).astype(np.uint8)
        area = np.sum(region_mask)
        
        # Size filtering — skip only truly tiny specs
        if area < MIN_NUC_AREA:
            continue
        if area > MAX_NUC_AREA:
            continue
        
        # Compute centroid
        ys, xs = np.where(region_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        scaled_pt = [[int(cx / RESIZE_FACTOR), int(cy / RESIZE_FACTOR)]]
        
        # Shape and intensity checks — classify as cell or debris
        is_debris = False
        debris_reason = ""
        
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            
            # Circularity check
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < MIN_CIRCULARITY:
                    is_debris = True
                    debris_reason = "irregular shape"
            
            # Solidity check
            if not is_debris:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < MIN_SOLIDITY:
                        is_debris = True
                        debris_reason = "deformed/spread"
        
        # Intensity check
        if not is_debris:
            mean_L = L_channel[region_mask > 0].mean()
            if mean_L > MAX_MEAN_L:
                is_debris = True
                debris_reason = "too light"
        
        if is_debris:
            debris_centroids.append({"point": scaled_pt, "reason": debris_reason})
        else:
            centroids.append(scaled_pt)

    return centroids, debris_centroids, original_img

# ==========================================
# CROP EXTRACTION LOGIC
# ==========================================
def extract_128_crop(original_img, mask, centroid, crop_id, basename):
    """
    Given a single SAM mask, isolates the cell by blacking out the background,
    builds a bounding box around the mask, expands it to exactly 128x128,
    and returns the final crop image.
    """
    h, w = original_img.shape[:2]
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # 1. Mask out background (keep only this specific cell)
    cell_isolated = cv2.bitwise_and(original_img, original_img, mask=binary_mask)

    # 2. Find bounding box of the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    # Get combined bounding box of all mask contours
    all_pts = np.vstack(contours)
    x, y, w_box, h_box = cv2.boundingRect(all_pts)
    
    # 3. Calculate center of the bounding box
    cx, cy = x + w_box // 2, y + h_box // 2

    # 4. Calculate crop coordinates for an expanded RAW window
    half_size = RAW_CROP_SIZE // 2
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size

    # 5. Handle Boundary Cases (Padding)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)

    # Adjust coordinates to valid image boundaries
    valid_x1, valid_y1 = max(0, x1), max(0, y1)
    valid_x2, valid_y2 = min(w, x2), min(h, y2)

    # Extract valid region
    crop_region = cell_isolated[valid_y1:valid_y2, valid_x1:valid_x2]

    # Pad if crop went outside image bounds
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop_region = cv2.copyMakeBorder(
            crop_region, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # Verify extracted size
    if crop_region.shape[:2] != (RAW_CROP_SIZE, RAW_CROP_SIZE):
        print(f"Warning: Bad crop size {crop_region.shape} on {basename}")
        crop_region = cv2.resize(crop_region, (RAW_CROP_SIZE, RAW_CROP_SIZE)) # Safety fallback
        
    # Resize down to requested final target size
    final_crop = cv2.resize(crop_region, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

    # 6. Save crop
    # Convert RGB back to BGR for cv2.imwrite
    crop_bgr = cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_crop_{crop_id}.jpg")
    cv2.imwrite(out_path, crop_bgr)
    return True

# ==========================================
# MAIN PIPELINE
# ==========================================
def process_dataset():
    print("Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print(f"SAM loaded on {device}.")

    # Gather images
    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {DATASET_DIR}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    
    total_crops_generated = 0
    start_time = time.time()

    for filename in tqdm(image_files, desc="Images Processed"):
        filepath = os.path.join(DATASET_DIR, filename)
        basename = os.path.splitext(filename)[0]

        # Stage 1: Watershed finds prompts
        centroids, original_img = get_watershed_centroids(filepath)
        
        if not centroids:
            continue
            
        # Format prompts for SAM
        batched_points = [[pt[0]] for pt in centroids]
        image_pil = Image.fromarray(original_img)

        # Stage 2: SAM Segmentation
        inputs = processor(image_pil, input_points=[batched_points], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu().numpy()

        # Stage 3: Crop Extraction
        for i, centroid in enumerate(centroids):
            # Extract highest confidence mask for this single prompt
            best_idx = scores[0][i].argmax()
            best_mask = masks[0][i][best_idx].numpy()
            
            success = extract_128_crop(original_img, best_mask, centroid[0], i, basename)
            if success:
                total_crops_generated += 1

    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print(f"Total Images Processed: {len(image_files)}")
    print(f"Total 128x128 Crops Generated: {total_crops_generated}")
    print(f"Total Time Elapsed: {elapsed:.1f}s")
    print(f"Avg Time Per Image: {elapsed/len(image_files):.2f}s")
    print("="*50)

if __name__ == "__main__":
    process_dataset()
