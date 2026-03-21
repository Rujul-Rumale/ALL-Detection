"""
visualize_pipeline_stages.py
============================
Generates a publication-quality 8-panel figure showing every step
of the segmentation pipeline, from raw blood smear to final masked cells.

Usage (from project root):
    python scripts/visualize_pipeline_stages.py <image_path> [--out outputs/pipeline_stages/]

Panels:
    01  Raw blood smear
    02  LAB K-Means nucleus mask
    03  Morphologically cleaned binary mask
    04  Euclidean Distance Transform heatmap
    05  Watershed labelled regions + peak markers
    06  Centroid point-prompts on original
    07  SAM per-cell masks (semi-transparent overlay)
    08  128×128 cell crops grid (black background)

Outputs:
    stage_01_raw.png ... stage_08_crops_grid.png  (individual panels)
    pipeline_stages_composite.png                  (2×4 grid, 300 DPI)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

# ── Project root paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Watershed internals (we re-run the pipeline step by step to capture intermediates)
from skimage.color import rgb2lab
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from transformers import SamModel, SamProcessor

# Crop extraction logic from demo_pipeline
from src.detection.demo_pipeline import DemoPipeline

# Config constants
RESIZE_FACTOR   = 0.5
MIN_NUC_AREA    = 50
MAX_NUC_AREA    = 8000
MIN_CIRCULARITY = 0.40
MIN_SOLIDITY    = 0.50
MAX_MEAN_L      = 65
INPUT_RES       = 320    # model input used for caption notes only


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_watershed_stages(image_path: str) -> dict:
    """
    Runs the full watershed pipeline step-by-step, returning intermediate
    arrays for each stage so they can be visualized.
    Returns a dict with all stage arrays.
    """
    img_array = np.fromfile(image_path, np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_small     = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    rows, cols    = img_small.shape[:2]

    # ── Stage 2: LAB K-Means ──────────────────────────────────────────────────
    lab = rgb2lab(img_rgb_small)
    ab  = lab[:, :, 1:3].reshape(-1, 2)
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)
    labels = kmeans.fit_predict(ab).reshape(rows, cols)
    L = lab[:, :, 0]
    avg_L = [L[labels == i].mean() for i in range(3)]
    nuc_cluster = int(np.argmin(avg_L))
    kmeans_mask = (labels == nuc_cluster).astype(np.uint8) * 255

    # Colour-overlay for display
    kmeans_vis = img_rgb_small.copy()
    kmeans_vis[kmeans_mask > 0] = [255, 80, 80]  # red highlight

    # ── Stage 3: Morphological cleanup ───────────────────────────────────────
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened  = cv2.morphologyEx(kmeans_mask, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    cleaned = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, kernel_close)

    # ── Stage 4: Distance Transform ──────────────────────────────────────────
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # ── Stage 5: Watershed ────────────────────────────────────────────────────
    coords = peak_local_max(
        dist_transform,
        min_distance=20,
        threshold_abs=5.0,
        labels=cleaned,
    )

    markers = np.zeros((rows, cols), dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 2

    markers_dilated = cv2.dilate(
        markers.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(np.int32)

    sure_bg = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, cleaned)
    markers_dilated[sure_bg == 0] = 1
    markers_dilated[unknown > 0]  = 0

    img_ws = img_rgb_small.copy()
    cv2.watershed(img_ws, markers_dilated)

    # ── Stage 6: Centroids ────────────────────────────────────────────────────
    centroids    = []
    debris_list  = []
    L_channel    = lab[:, :, 0]

    for label in np.unique(markers_dilated):
        if label <= 1:
            continue
        region_mask = (markers_dilated == label).astype(np.uint8)
        area = int(np.sum(region_mask))
        if area < MIN_NUC_AREA or area > MAX_NUC_AREA:
            continue

        ys, xs = np.where(region_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        scaled_pt = [[int(cx / RESIZE_FACTOR), int(cy / RESIZE_FACTOR)]]

        is_debris = False
        debris_reason = ""
        conts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            cnt = conts[0]
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circ = 4 * np.pi * area / (perimeter * perimeter)
                if circ < MIN_CIRCULARITY:
                    is_debris, debris_reason = True, "irregular shape"
            if not is_debris:
                hull      = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and area / hull_area < MIN_SOLIDITY:
                    is_debris, debris_reason = True, "deformed/spread"
        if not is_debris:
            mean_L = L_channel[region_mask > 0].mean()
            if mean_L > MAX_MEAN_L:
                is_debris, debris_reason = True, "too light"

        if is_debris:
            debris_list.append({"point": scaled_pt, "reason": debris_reason})
        else:
            centroids.append(scaled_pt)

    return {
        "original_img":    original_img,
        "img_small_rgb":   img_rgb_small,
        "kmeans_vis":      kmeans_vis,
        "kmeans_mask":     kmeans_mask,
        "cleaned":         cleaned,
        "dist_transform":  dist_transform,
        "markers":         markers_dilated,
        "peaks":           coords,
        "centroids":       centroids,
        "debris_list":     debris_list,
    }


def build_watershed_vis(markers: np.ndarray, peaks: np.ndarray, img_rgb_small: np.ndarray) -> np.ndarray:
    """Rainbow-coloured watershed regions with white peak markers."""
    unique = [l for l in np.unique(markers) if l > 1]
    cmap   = plt.get_cmap("tab20", max(len(unique), 1))
    vis    = img_rgb_small.copy().astype(np.float32) / 255.0
    overlay = np.zeros_like(vis)

    for i, label in enumerate(unique):
        color = np.array(cmap(i % 20)[:3])
        mask  = (markers == label)
        overlay[mask] = color

    vis = (vis * 0.5 + overlay * 0.5)
    vis = np.clip(vis * 255, 0, 255).astype(np.uint8)

    # Draw peak markers
    for (y, x) in peaks:
        cv2.circle(vis, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(vis, (x, y), 5, (0, 0, 0), 1)
    return vis


def build_centroid_overlay(original_img: np.ndarray, centroids: list, debris_list: list) -> np.ndarray:
    """Red dots for cell centroids, orange for debris."""
    vis = original_img.copy()
    for pt in centroids:
        x, y = pt[0]
        cv2.circle(vis, (x, y), 10, (255, 60, 60), -1)
        cv2.circle(vis, (x, y), 11, (255, 255, 255), 1)
    for d in debris_list:
        x, y = d["point"][0]
        cv2.circle(vis, (x, y), 8, (255, 170, 0), -1)
    return vis


def build_sam_overlay(original_img: np.ndarray, masks_list: list, scores: np.ndarray,
                      n_cells: int) -> np.ndarray:
    """Semi-transparent coloured SAM masks on the original image."""
    vis  = original_img.copy().astype(np.float32) / 255.0
    cmap = plt.get_cmap("tab20", max(len(masks_list), 1))

    for i, (best_mask, is_debris) in enumerate(masks_list):
        color = np.array([1.0, 0.6, 0.0] if is_debris else cmap(i % 20)[:3])
        bin_mask = (best_mask > 0.5)
        vis[bin_mask] = vis[bin_mask] * 0.45 + color * 0.55

        # Border
        contours, _ = cv2.findContours(
            bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            color_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            cv2.drawContours((vis * 255).astype(np.uint8), contours, -1, color_255, 2)

    return np.clip(vis * 255, 0, 255).astype(np.uint8)


def build_crops_grid(crops: list, grid_cols: int = 6) -> np.ndarray:
    """Mosaic of 128×128 cell crops on a black background."""
    if not crops:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    cell_size = 128
    n         = len(crops)
    grid_rows = (n + grid_cols - 1) // grid_cols
    canvas    = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)

    for idx, crop_rgb in enumerate(crops):
        r, c = divmod(idx, grid_cols)
        canvas[r * cell_size:(r + 1) * cell_size,
               c * cell_size:(c + 1) * cell_size] = crop_rgb

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

PANEL_TITLES = [
    "Stage 1 — Raw Blood Smear",
    "Stage 2 — LAB K-Means\nNucleus Mask",
    "Stage 3 — Morphological\nBinary Mask",
    "Stage 4 — Euclidean Distance\nTransform",
    "Stage 5 — Watershed Regions\n+ Peak Markers",
    "Stage 6 — Centroid\nPoint-Prompts",
    "Stage 7 — SAM Per-Cell\nMask Overlay",
    "Stage 8 — 128×128 Cell Crops\n(resized to 320×320 for inference)",
]

PANEL_FNAMES = [
    "stage_01_raw.png",
    "stage_02_kmeans.png",
    "stage_03_binary_mask.png",
    "stage_04_distance_transform.png",
    "stage_05_watershed.png",
    "stage_06_centroids.png",
    "stage_07_sam_masks.png",
    "stage_08_crops_grid.png",
]


def save_panel(img: np.ndarray, title: str, path: str, cmap=None):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor("#1A1B1E")
    ax.set_facecolor("#25262B")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
    if img.ndim == 2:
        ax.imshow(img, cmap=cmap or "jet")
    else:
        ax.imshow(img)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize segmentation pipeline stages")
    p.add_argument("image_path", help="Path to blood smear image")
    p.add_argument("--out", default="outputs/pipeline_stages/",
                   help="Output directory")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔬  Pipeline Visualizer")
    print(f"    Image:  {args.image_path}")
    print(f"    Output: {out_dir}\n")

    # ── Run watershed stages ──────────────────────────────────────────────────
    print("[1/4] Running watershed pipeline ...")
    stages = run_watershed_stages(args.image_path)

    original_img   = stages["original_img"]
    img_small_rgb  = stages["img_small_rgb"]
    kmeans_vis     = stages["kmeans_vis"]
    cleaned        = stages["cleaned"]
    dist_transform = stages["dist_transform"]
    markers        = stages["markers"]
    peaks          = stages["peaks"]
    centroids      = stages["centroids"]
    debris_list    = stages["debris_list"]

    watershed_vis  = build_watershed_vis(markers, peaks, img_small_rgb)
    centroid_vis   = build_centroid_overlay(original_img, centroids, debris_list)

    print(f"    → {len(centroids)} cell centroids, {len(debris_list)} debris")

    # ── Run SAM ──────────────────────────────────────────────────────────────
    print("[2/4] Running SAM segmentation ...")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model  = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_proc   = SamProcessor.from_pretrained("facebook/sam-vit-base")

    all_points  = [pt for pt in centroids] + [d["point"] for d in debris_list]
    n_cells     = len(centroids)

    sam_vis     = original_img.copy()
    crops       = []
    masks_info  = []

    if all_points:
        batched_pts = [[pt[0]] for pt in all_points]
        image_pil   = Image.fromarray(original_img)
        inputs      = sam_proc(image_pil, input_points=[batched_pts],
                               return_tensors="pt").to(device)
        with torch.no_grad():
            sam_out = sam_model(**inputs)

        masks_raw = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores_raw = sam_out.iou_scores.cpu().numpy()

        # Extract crops using DemoPipeline logic
        pipeline = DemoPipeline()

        for i in range(len(all_points)):
            best_idx  = scores_raw[0][i].argmax()
            best_mask = masks_raw[0][i][best_idx].numpy()
            is_debris = i >= n_cells

            masks_info.append((best_mask, is_debris))

            crop_bgr = pipeline.mock_extract_128_crop_in_memory(
                original_img, best_mask, all_points[i][0], i, "vis"
            )
            if crop_bgr is not None and not is_debris:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crops.append(crop_rgb)

        sam_vis = build_sam_overlay(original_img, masks_info, scores_raw, n_cells)

    print(f"    → {len(crops)} cell crops extracted")

    # ── Build crops grid ──────────────────────────────────────────────────────
    crops_grid = build_crops_grid(crops, grid_cols=min(6, max(len(crops), 1)))

    # ── All panel images ──────────────────────────────────────────────────────
    panel_images = [
        (original_img,            None),       # 1 Raw
        (kmeans_vis,              None),       # 2 K-Means
        (cleaned,                 "gray"),     # 3 Binary mask
        (dist_transform,          "jet"),      # 4 EDT
        (watershed_vis,           None),       # 5 Watershed
        (centroid_vis,            None),       # 6 Centroids
        (sam_vis,                 None),       # 7 SAM masks
        (crops_grid,              None),       # 8 Crops grid
    ]

    # ── Save individual panels ────────────────────────────────────────────────
    print("[3/4] Saving individual stage PNGs ...")
    for (img, cmap), title, fname in zip(panel_images, PANEL_TITLES, PANEL_FNAMES):
        save_panel(img, title, str(out_dir / fname), cmap=cmap)

    # ── Save 2×4 composite ───────────────────────────────────────────────────
    print("[4/4] Generating composite figure ...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=300)
    fig.patch.set_facecolor("#1A1B1E")
    img_name = os.path.basename(args.image_path)
    fig.suptitle(
        f"Segmentation Pipeline — {img_name}",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (img, cmap), title in zip(axes.flatten(), panel_images, PANEL_TITLES):
        ax.set_facecolor("#25262B")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=8, fontweight="bold", pad=4)
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "jet")
        else:
            ax.imshow(img)

    fig.tight_layout(pad=0.4)
    composite_path = out_dir / "pipeline_stages_composite.png"
    fig.savefig(str(composite_path), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n✅  Done!")
    print(f"   Composite: {composite_path}")
    print(f"   Individual panels: {out_dir}")


if __name__ == "__main__":
    main()
