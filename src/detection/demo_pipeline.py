import os
import time
import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

# Add parent directory to path for imports if needed
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.generate_cell_crops_sam import get_watershed_centroids, extract_128_crop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 128
BLAST_CONF_THRESHOLD = 0.85  # Only classify as ALL if confidence >= this (model has ALL bias)
from src.config import INPUT_RES, IMAGENET_MEAN, IMAGENET_STD
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_MODELS_DIR = str(_PROJECT_ROOT / "models")

MODELS = {
    "MobileNetV3-Large v2 (Balanced)": {
        "path": os.path.join(_MODELS_DIR, "mobilenetv3_large_v2.tflite"),
        "class_to_idx": {'all': 0, 'hem': 1}
    },
    "MobileNetV3-Small (Weighted)": {
        "path": os.path.join(_MODELS_DIR, "mobilenetv3_cnmc.tflite"),
        "class_to_idx": {'all': 0, 'hem': 1}
    },
    "MobileNetV3-Small (Unweighted)": {
        "path": os.path.join(_MODELS_DIR, "mobilenetv3_cnmc_unweighted.tflite"),
        "class_to_idx": {'all': 0, 'hem': 1}
    },
    "MobileNetV3-Large (Weighted)": {
        "path": os.path.join(_MODELS_DIR, "mobilenetv3_large_cnmc.tflite"),
        "class_to_idx": {'all': 0, 'hem': 1}
    }
}

class DemoPipeline:
    def __init__(self):
        self.sam_model = None
        self.sam_processor = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.current_model_name = None

    def initialize_sam(self):
        if self.sam_model is None:
            print("Loading SAM model...")
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            print(f"SAM loaded on {DEVICE}.")

    def load_classifier(self, model_name):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        if self.current_model_name == model_name and self.interpreter is not None:
            return # Already loaded
            
        print(f"Loading TFLite classifier: {model_name}...")
        config = MODELS[model_name]
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=config["path"])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.current_model_name = model_name
            print(f"TFLite Classifier loaded.")
        except Exception as e:
            print(f"Failed to load TFLite model at {config['path']}: {e}")
            raise

    def mock_extract_128_crop_in_memory(self, original_img, mask, centroid, crop_id, basename):
        """
        CNMC-matched crop extraction. Applies 4 standardization fixes:
          1. Single connected component (discard mask fragments)
          2. Bad mask rejection (too small, extreme aspect ratio, edge-touching)
          3. Mask centroid centering (not watershed centroid)
          4. Scale normalization (cell fills ~75% of frame, like CNMC)
        Returns BGR numpy array or None if rejected.
        """
        TARGET = CROP_SIZE  # 128
        FILL_RATIO = 0.75   # Target: cell fills ~75% of frame
        MIN_MASK_AREA = 200  # Minimum mask area in pixels
        MAX_ASPECT = 3.0     # Max aspect ratio before rejection
        
        h, w = original_img.shape[:2]
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # ── Fix 4: Keep only largest connected component ──
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels <= 1:
            return None  # no foreground
        # Component 0 is background; find largest foreground component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = (labels == largest_label).astype(np.uint8)
        
        # ── Fix 3: Reject bad masks ──
        mask_area = int(np.sum(binary_mask))
        if mask_area < MIN_MASK_AREA:
            return None  # too small — debris
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        all_pts = np.vstack(contours)
        x, y, w_box, h_box = cv2.boundingRect(all_pts)
        
        # Aspect ratio check
        aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
        if aspect > MAX_ASPECT:
            return None  # extremely elongated — not a cell
        
        # Edge-touching check — likely partial cell
        if x <= 1 or y <= 1 or (x + w_box) >= w - 1 or (y + h_box) >= h - 1:
            return None
        
        # ── Fix 2: Center on mask centroid (mean of mask pixels) ──
        ys, xs = np.where(binary_mask > 0)
        mask_cx = int(np.mean(xs))
        mask_cy = int(np.mean(ys))
        
        # ── Fix 1: Normalize cell-to-frame ratio ──
        # Isolate cell
        cell_isolated = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
        
        # Compute scale: largest cell dimension should fill FILL_RATIO of TARGET
        cell_max_dim = max(w_box, h_box)
        target_dim = int(TARGET * FILL_RATIO)
        scale = target_dim / max(cell_max_dim, 1)
        
        # Crop a generous region around the mask centroid
        margin = int(cell_max_dim * 0.8)  # extra margin around cell
        cx1 = max(0, mask_cx - margin)
        cy1 = max(0, mask_cy - margin)
        cx2 = min(w, mask_cx + margin)
        cy2 = min(h, mask_cy + margin)
        
        cell_region = cell_isolated[cy1:cy2, cx1:cx2]
        mask_region = binary_mask[cy1:cy2, cx1:cx2]
        
        if cell_region.size == 0:
            return None
        
        # Scale the cell region
        new_h = int(cell_region.shape[0] * scale)
        new_w = int(cell_region.shape[1] * scale)
        if new_h < 1 or new_w < 1:
            return None
        
        scaled_cell = cv2.resize(cell_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask to scaled version (clean edges)
        scaled_cell = cv2.bitwise_and(scaled_cell, scaled_cell, mask=scaled_mask)
        
        # Paste into center of TARGET×TARGET black canvas
        canvas = np.zeros((TARGET, TARGET, 3), dtype=np.uint8)
        
        # Compute paste position (centered)
        paste_y = max(0, (TARGET - new_h) // 2)
        paste_x = max(0, (TARGET - new_w) // 2)
        
        # Source region (may need clipping if scaled larger than target)
        src_y1 = max(0, (new_h - TARGET) // 2)
        src_x1 = max(0, (new_w - TARGET) // 2)
        src_y2 = src_y1 + min(new_h, TARGET)
        src_x2 = src_x1 + min(new_w, TARGET)
        
        dst_h = src_y2 - src_y1
        dst_w = src_x2 - src_x1
        
        canvas[paste_y:paste_y + dst_h, paste_x:paste_x + dst_w] = scaled_cell[src_y1:src_y2, src_x1:src_x2]
        
        # Return as BGR
        crop_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        return crop_bgr

    def process_image(self, image_path, model_name):
        """
        1. Watershed to find centroids + debris
        2. SAM to segment all (cells + debris)
        3. TFLite Classifier on cells only
        Returns dictionary of results for UI.
        """
        self.initialize_sam()
        self.load_classifier(model_name)
        
        # 1. Watershed — now returns debris separately
        t0 = time.time()
        centroids, debris_list, original_img = get_watershed_centroids(image_path)
        watershed_time = time.time() - t0
        
        num_cells = len(centroids)
        num_debris = len(debris_list)
        
        if num_cells == 0 and num_debris == 0:
            return {
                "total_cells": 0,
                "blast_count": 0,
                "healthy_count": 0,
                "debris_count": 0,
                "detections": [],
                "annotated_image": original_img,
                "watershed_time": watershed_time,
                "sam_time": 0,
                "classify_time": 0,
                "total_time": watershed_time,
            }
        
        # Combine cell + debris centroids for a single SAM pass
        all_points = []
        for pt in centroids:
            all_points.append(pt)
        for d in debris_list:
            all_points.append(d["point"])
        
        batched_points = [[pt[0]] for pt in all_points]
        image_pil = Image.fromarray(original_img)
        
        # 2. SAM Segmentation (all at once)
        t1 = time.time()
        inputs = self.sam_processor(image_pil, input_points=[batched_points], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            sam_outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = sam_outputs.iou_scores.cpu().numpy()
        sam_time = time.time() - t1

        results = {
            "total_cells": 0,
            "blast_count": 0,
            "healthy_count": 0,
            "debris_count": num_debris,
            "detections": []
        }
        
        annotated_img = original_img.copy()
        
        class_to_idx = MODELS[model_name]['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # 3. Process cells (classify) + debris (label only)
        t2 = time.time()
        cell_id = 0
        
        for i in range(len(all_points)):
            best_idx = scores[0][i].argmax()
            best_mask = masks[0][i][best_idx].numpy()
            is_debris_item = i >= num_cells  # first num_cells are real cells
            
            centroid_pt = all_points[i][0]
            crop_bgr = self.mock_extract_128_crop_in_memory(original_img, best_mask, centroid_pt, i, "mem")
            
            if crop_bgr is None:
                continue
            
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            cell_id += 1
            
            # Contour overlay
            binary_mask = (best_mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if is_debris_item:
                # Debris: yellow contour, no classification
                debris_info = debris_list[i - num_cells]
                if contours:
                    cv2.drawContours(annotated_img, contours, -1, (255, 170, 0), 1)  # Yellow, thin
                
                results["detections"].append({
                    "id": cell_id,
                    "classification": "DEBRIS",
                    "is_blast": False,
                    "is_debris": True,
                    "confidence": 0.0,
                    "score": 0.0,
                    "debris_reason": debris_info.get("reason", "unknown"),
                    "circularity": 0.0,
                    "crop": crop_rgb
                })
            else:
                # Real cell: classify with TFLite + TTA (4 orientations)
                results["total_cells"] += 1
                
                crop_resized = cv2.resize(crop_rgb, (INPUT_RES, INPUT_RES))
                
                # Generate 4 TTA variants: original, h-flip, v-flip, both
                variants = [
                    crop_resized,
                    np.fliplr(crop_resized),           # horizontal flip
                    np.flipud(crop_resized),           # vertical flip
                    np.flipud(np.fliplr(crop_resized)) # both flips
                ]
                
                # Run inference on all variants, average probabilities
                all_probs = []
                for variant in variants:
                    v_norm = variant.astype(np.float32) / 255.0
                    v_norm = (v_norm - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                    v_trans = np.transpose(v_norm, (2, 0, 1))
                    v_tensor = np.expand_dims(v_trans, axis=0).astype(np.float32)
                    
                    self.interpreter.set_tensor(self.input_details[0]['index'], v_tensor)
                    self.interpreter.invoke()
                    raw_out = self.interpreter.get_tensor(self.output_details[0]['index'])
                    
                    exp_out = np.exp(raw_out - np.max(raw_out))
                    all_probs.append(exp_out / np.sum(exp_out))
                
                # Average softmax across all TTA variants
                avg_probs = np.mean(all_probs, axis=0)
                pred_class_idx = np.argmax(avg_probs[0])
                confidence = float(avg_probs[0][pred_class_idx])
                
                class_name = idx_to_class[pred_class_idx].upper()
                
                # ALL probability from averaged predictions
                all_idx = class_to_idx.get('all', 0)
                all_prob = float(avg_probs[0][all_idx])
                
                # Confidence threshold: if model says ALL but isn't confident enough,
                # override to HEM to reduce false positives
                if class_name == 'ALL' and confidence < BLAST_CONF_THRESHOLD:
                    class_name = 'HEM'
                    confidence = 1.0 - all_prob
                
                is_blast = class_name == 'ALL'
                
                if is_blast:
                    results["blast_count"] += 1
                else:
                    results["healthy_count"] += 1
                
                if contours:
                    color = (255, 77, 77) if is_blast else (64, 192, 87)
                    cv2.drawContours(annotated_img, contours, -1, color, 2)
                
                area = cv2.contourArea(contours[0]) if contours else 0
                perimeter = cv2.arcLength(contours[0], True) if contours else 0
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                results["detections"].append({
                    "id": cell_id,
                    "classification": class_name,
                    "is_blast": is_blast,
                    "is_debris": False,
                    "confidence": confidence,
                    "score": confidence,
                    "circularity": circularity,
                    "crop": crop_rgb
                })

        classify_time = time.time() - t2
        total_time = time.time() - t0

        results["annotated_image"] = annotated_img
        results["watershed_time"] = round(watershed_time, 2)
        results["sam_time"] = round(sam_time, 2)
        results["classify_time"] = round(classify_time, 2)
        results["total_time"] = round(total_time, 2)
        return results

