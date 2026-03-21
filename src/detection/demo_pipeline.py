import os
import time
import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

import sys
from pathlib import Path as _Path

# Insert project root so 'src.*' imports resolve correctly regardless of CWD
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.detection.generate_cell_crops_sam import get_watershed_centroids, extract_128_crop
from src.utils.benchmark_logger import BenchmarkLogger
from src.config import INPUT_RES, IMAGENET_MEAN, IMAGENET_STD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 128
BLAST_CONF_THRESHOLD = 0.85  # Only classify as ALL if confidence >= this

_MODELS_DIR = str(_PROJECT_ROOT / "models" / "tflite_final")

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  (8 entries: 6 individual folds + 2 ensembles)
# ═════════════════════════════════════════════════════════════════════════════
MODELS = {
    # ── Individual folds ─────────────────────────────────────────────────────
    "EfficientNet-B0  | Fold 1": {
        "paths": [os.path.join(_MODELS_DIR, "effb0_fold1_best.tflite")],
        "arch": "effb0", "fold": 1,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "EfficientNet-B0  | Fold 2": {
        "paths": [os.path.join(_MODELS_DIR, "effb0_fold2_best.tflite")],
        "arch": "effb0", "fold": 2,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "EfficientNet-B0  | Fold 3": {
        "paths": [os.path.join(_MODELS_DIR, "effb0_fold3_best.tflite")],
        "arch": "effb0", "fold": 3,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "MobileNetV3-Large | Fold 1": {
        "paths": [os.path.join(_MODELS_DIR, "mnv3l_fold1_best.tflite")],
        "arch": "mnv3l", "fold": 1,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "MobileNetV3-Large | Fold 2": {
        "paths": [os.path.join(_MODELS_DIR, "mnv3l_fold2_best.tflite")],
        "arch": "mnv3l", "fold": 2,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "MobileNetV3-Large | Fold 3": {
        "paths": [os.path.join(_MODELS_DIR, "mnv3l_fold3_best.tflite")],
        "arch": "mnv3l", "fold": 3,
        "class_to_idx": {"all": 0, "hem": 1},
    },
    # ── Ensembles ─────────────────────────────────────────────────────────────
    "EfficientNet-B0  | Ensemble": {
        "paths": [
            os.path.join(_MODELS_DIR, "effb0_fold1_best.tflite"),
            os.path.join(_MODELS_DIR, "effb0_fold2_best.tflite"),
            os.path.join(_MODELS_DIR, "effb0_fold3_best.tflite"),
        ],
        "arch": "effb0", "fold": "ensemble",
        "class_to_idx": {"all": 0, "hem": 1},
    },
    "MobileNetV3-Large | Ensemble": {
        "paths": [
            os.path.join(_MODELS_DIR, "mnv3l_fold1_best.tflite"),
            os.path.join(_MODELS_DIR, "mnv3l_fold2_best.tflite"),
            os.path.join(_MODELS_DIR, "mnv3l_fold3_best.tflite"),
        ],
        "arch": "mnv3l", "fold": "ensemble",
        "class_to_idx": {"all": 0, "hem": 1},
    },
}


class DemoPipeline:
    def __init__(self):
        self.sam_model          = None
        self.sam_processor      = None
        self.interpreters       = []   # list — supports ensemble
        self.current_model_name = None
        self.logger             = BenchmarkLogger()

    def initialize_sam(self):
        if self.sam_model is None:
            print("Loading SAM model...")
            self.sam_model     = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            print(f"SAM loaded on {DEVICE}.")

    def load_classifier(self, model_name: str):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        if self.current_model_name == model_name and self.interpreters:
            return  # already loaded

        print(f"Loading TFLite classifier(s): {model_name} ...")
        config = MODELS[model_name]
        self.interpreters = []

        for path in config["paths"]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TFLite file not found: {path}\n"
                    f"Run export_scripts/convert_onnx_to_tflite.py first."
                )
            interp = tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            self.interpreters.append(interp)

        self.current_model_name = model_name
        print(f"Loaded {len(self.interpreters)} interpreter(s).")

    # ── crop extraction ───────────────────────────────────────────────────────

    def mock_extract_128_crop_in_memory(self, original_img, mask, centroid, crop_id, basename):
        """
        CNMC-matched crop extraction.
         1. Keep largest connected component.
         2. Reject bad masks (too small / extreme aspect / edge-touching).
         3. Centre on mask centroid.
         4. Normalise cell-to-frame ratio (~75% fill).
        Returns BGR numpy array (128×128) or None if rejected.
        """
        TARGET     = CROP_SIZE   # 128
        FILL_RATIO = 0.75
        MIN_MASK_AREA = 200
        MAX_ASPECT = 3.0

        h, w = original_img.shape[:2]
        binary_mask = (mask > 0.5).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels <= 1:
            return None
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = (labels == largest_label).astype(np.uint8)

        mask_area = int(np.sum(binary_mask))
        if mask_area < MIN_MASK_AREA:
            return None

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        all_pts = np.vstack(contours)
        x, y, w_box, h_box = cv2.boundingRect(all_pts)

        aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
        if aspect > MAX_ASPECT:
            return None

        if x <= 1 or y <= 1 or (x + w_box) >= w - 1 or (y + h_box) >= h - 1:
            return None

        ys, xs = np.where(binary_mask > 0)
        mask_cx = int(np.mean(xs))
        mask_cy = int(np.mean(ys))

        cell_isolated = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
        cell_max_dim = max(w_box, h_box)
        target_dim   = int(TARGET * FILL_RATIO)
        scale        = target_dim / max(cell_max_dim, 1)

        margin = int(cell_max_dim * 0.8)
        cx1 = max(0, mask_cx - margin); cy1 = max(0, mask_cy - margin)
        cx2 = min(w, mask_cx + margin); cy2 = min(h, mask_cy + margin)

        cell_region = cell_isolated[cy1:cy2, cx1:cx2]
        mask_region = binary_mask[cy1:cy2, cx1:cx2]

        if cell_region.size == 0:
            return None

        new_h = int(cell_region.shape[0] * scale)
        new_w = int(cell_region.shape[1] * scale)
        if new_h < 1 or new_w < 1:
            return None

        scaled_cell = cv2.resize(cell_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        scaled_cell = cv2.bitwise_and(scaled_cell, scaled_cell, mask=scaled_mask)

        canvas  = np.zeros((TARGET, TARGET, 3), dtype=np.uint8)
        paste_y = max(0, (TARGET - new_h) // 2)
        paste_x = max(0, (TARGET - new_w) // 2)
        src_y1  = max(0, (new_h - TARGET) // 2)
        src_x1  = max(0, (new_w - TARGET) // 2)
        src_y2  = src_y1 + min(new_h, TARGET)
        src_x2  = src_x1 + min(new_w, TARGET)
        dst_h   = src_y2 - src_y1
        dst_w   = src_x2 - src_x1
        canvas[paste_y:paste_y + dst_h, paste_x:paste_x + dst_w] = scaled_cell[src_y1:src_y2, src_x1:src_x2]

        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # ── inference with ensemble averaging ─────────────────────────────────────

    def _run_classifier(self, crop_rgb: np.ndarray, input_details_list, output_details_list) -> np.ndarray:
        """
        Resizes crop to 320×320, runs 4-TTA across all interpreters,
        returns averaged softmax probability vector [n_classes].
        """
        crop_resized = cv2.resize(crop_rgb, (INPUT_RES, INPUT_RES))  # explicitly 320×320

        # TTA variants: original, h-flip, v-flip, both
        variants = [
            crop_resized,
            np.fliplr(crop_resized),
            np.flipud(crop_resized),
            np.flipud(np.fliplr(crop_resized)),
        ]

        all_probs = []
        for interp, in_det, out_det in zip(self.interpreters, input_details_list, output_details_list):
            for variant in variants:
                v_norm = variant.astype(np.float32) / 255.0
                v_norm = (v_norm - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                v_trans = np.transpose(v_norm, (2, 0, 1))                    # HWC→CHW
                v_tensor = np.expand_dims(v_trans, axis=0).astype(np.float32)

                # Handle INT8 quantized input
                in_dtype = in_det[0]["dtype"]
                if in_dtype == np.int8:
                    scale, zero_point = in_det[0]["quantization"]
                    if scale != 0:
                        v_tensor = (v_tensor / scale + zero_point).clip(-128, 127).astype(np.int8)
                    else:
                        v_tensor = v_tensor.astype(np.int8)

                interp.set_tensor(in_det[0]["index"], v_tensor)
                interp.invoke()
                raw_out = interp.get_tensor(out_det[0]["index"]).astype(np.float32)

                exp_out = np.exp(raw_out - np.max(raw_out))
                all_probs.append(exp_out / np.sum(exp_out))

        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs[0]   # shape [n_classes]

    # ── main pipeline ─────────────────────────────────────────────────────────

    def process_image(self, image_path: str, model_name: str) -> dict:
        """
        1. Watershed   → centroids + debris
        2. SAM         → segment all points
        3. TFLite (+ TTA + ensemble) → classify cells
        Returns results dict for UI.
        """
        self.initialize_sam()
        self.load_classifier(model_name)

        config          = MODELS[model_name]
        class_to_idx    = config["class_to_idx"]
        idx_to_class    = {v: k for k, v in class_to_idx.items()}
        input_details   = [interp.get_input_details()  for interp in self.interpreters]
        output_details  = [interp.get_output_details() for interp in self.interpreters]

        # ── Start benchmark session ────────────────────────────────────────────
        self.logger.start_session(
            image_filename=image_path,
            model_name=model_name,
            arch=config["arch"],
            fold=config["fold"],
        )

        # ── Stage 1: Watershed ─────────────────────────────────────────────────
        t0 = time.time()
        centroids, debris_list, original_img = get_watershed_centroids(image_path)
        watershed_time = time.time() - t0

        num_cells  = len(centroids)
        num_debris = len(debris_list)

        if num_cells == 0 and num_debris == 0:
            results = {
                "total_cells": 0, "blast_count": 0, "healthy_count": 0,
                "debris_count": 0, "detections": [],
                "annotated_image": original_img,
                "watershed_time": round(watershed_time, 2),
                "sam_time": 0, "classify_time": 0,
                "total_time": round(watershed_time, 2),
            }
            self.logger.finish_session(results, [])
            return results

        all_points   = [pt  for pt in centroids] + [d["point"] for d in debris_list]
        batched_pts  = [[pt[0]] for pt in all_points]
        image_pil    = Image.fromarray(original_img)

        # ── Stage 2: SAM ──────────────────────────────────────────────────────
        t1 = time.time()
        inputs = self.sam_processor(
            image_pil, input_points=[batched_pts], return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            sam_outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores    = sam_outputs.iou_scores.cpu().numpy()
        sam_time  = time.time() - t1

        results = {
            "total_cells": 0, "blast_count": 0,
            "healthy_count": 0, "debris_count": num_debris,
            "detections": [],
        }
        annotated_img     = original_img.copy()
        per_cell_times_ms = []   # per-cell inference time in ms

        # ── Stage 3: Classify ─────────────────────────────────────────────────
        t2 = time.time()
        cell_id = 0

        for i in range(len(all_points)):
            best_idx  = scores[0][i].argmax()
            best_mask = masks[0][i][best_idx].numpy()
            is_debris_item = i >= num_cells

            centroid_pt = all_points[i][0]
            crop_bgr    = self.mock_extract_128_crop_in_memory(
                original_img, best_mask, centroid_pt, i, "mem"
            )
            if crop_bgr is None:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            cell_id += 1

            binary_mask = (best_mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if is_debris_item:
                debris_info = debris_list[i - num_cells]
                if contours:
                    cv2.drawContours(annotated_img, contours, -1, (255, 170, 0), 1)
                results["detections"].append({
                    "id": cell_id, "classification": "DEBRIS",
                    "is_blast": False, "is_debris": True,
                    "confidence": 0.0, "score": 0.0,
                    "debris_reason": debris_info.get("reason", "unknown"),
                    "circularity": 0.0, "crop": crop_rgb,
                })
            else:
                results["total_cells"] += 1

                tc0      = time.perf_counter()
                avg_prob = self._run_classifier(crop_rgb, input_details, output_details)
                tc_ms    = (time.perf_counter() - tc0) * 1000
                per_cell_times_ms.append(tc_ms)

                pred_idx   = int(np.argmax(avg_prob))
                confidence = float(avg_prob[pred_idx])
                class_name = idx_to_class[pred_idx].upper()

                all_idx  = class_to_idx.get("all", 0)
                all_prob = float(avg_prob[all_idx])

                if class_name == "ALL" and confidence < BLAST_CONF_THRESHOLD:
                    class_name = "HEM"
                    confidence = 1.0 - all_prob

                is_blast = class_name == "ALL"

                if is_blast:
                    results["blast_count"] += 1
                else:
                    results["healthy_count"] += 1

                if contours:
                    color = (255, 77, 77) if is_blast else (64, 192, 87)
                    cv2.drawContours(annotated_img, contours, -1, color, 2)

                area       = cv2.contourArea(contours[0]) if contours else 0
                perimeter  = cv2.arcLength(contours[0], True) if contours else 0
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

                results["detections"].append({
                    "id": cell_id, "classification": class_name,
                    "is_blast": is_blast, "is_debris": False,
                    "confidence": confidence, "score": confidence,
                    "circularity": circularity, "crop": crop_rgb,
                })

        classify_time = time.time() - t2
        total_time    = time.time() - t0

        results["annotated_image"] = annotated_img
        results["watershed_time"]  = round(watershed_time,  2)
        results["sam_time"]        = round(sam_time,        2)
        results["classify_time"]   = round(classify_time,   2)
        results["total_time"]      = round(total_time,      2)

        # ── Finish benchmark ───────────────────────────────────────────────────
        self.logger.finish_session(results, per_cell_times_ms)

        return results
