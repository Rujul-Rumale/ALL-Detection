"""
Full Watershed+SAM Pipeline Visualization
Shows all intermediate stages for multiple L2 images:
  Row 1: Original | LAB Nucleus Mask | Distance Transform | Peaks/Centroids
  Row 2: SAM Masks | SAM Boundaries | Annotated Output
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from skimage.color import rgb2lab
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detection.generate_cell_crops_sam import (
    get_watershed_centroids, RESIZE_FACTOR, MIN_NUC_AREA, MAX_NUC_AREA
)

# ── Config ──
TEST_IMAGES = [
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im003_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im001_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im004_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im010_1.jpg",
    r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im020_1.jpg",
]
OUTPUT_DIR = r"c:\Open Source\leukiemea\temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_watershed_stages(image_path):
    """
    Runs the watershed pipeline and returns all intermediate stage images
    for visualization, plus the list of centroids.
    """
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    rows, cols = img_small.shape[:2]

    # Step 1: LAB K-Means
    lab = rgb2lab(img_rgb_small)
    ab = lab[:, :, 1:3].reshape(-1, 2)
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)
    labels = kmeans.fit_predict(ab).reshape(rows, cols)
    L = lab[:, :, 0]
    avg_L = [L[labels == i].mean() for i in range(3)]
    nuc_cluster = np.argmin(avg_L)
    mask = (labels == nuc_cluster).astype(np.uint8) * 255

    # Step 2: Morphological cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Step 3: Distance transform
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Step 4: peak_local_max
    coords = peak_local_max(
        dist_transform, min_distance=15, threshold_abs=4.0, labels=cleaned
    )

    # Step 5: Watershed
    markers = np.zeros((rows, cols), dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 2
    markers = cv2.dilate(
        markers.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(np.int32)
    sure_bg = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, cleaned)
    markers[sure_bg == 0] = 1
    markers[unknown > 0] = 0
    img_ws = img_rgb_small.copy()
    cv2.watershed(img_ws, markers)

    # Extract centroids
    centroids = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        region_mask = (markers == label)
        area = np.sum(region_mask)
        if area < MIN_NUC_AREA or area > MAX_NUC_AREA:
            continue
        ys, xs = np.where(region_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        centroids.append([[int(cx / RESIZE_FACTOR), int(cy / RESIZE_FACTOR)]])

    # ── Build stage images (all at original resolution) ──
    h, w = original_img.shape[:2]

    # Stage 1: Nucleus mask (upscale to original size, colorize)
    mask_vis = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask_vis > 0] = [255, 255, 255]

    # Stage 2: Distance transform heatmap
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    dist_color = cv2.cvtColor(dist_color, cv2.COLOR_BGR2RGB)
    # Mask out background
    bg = (cleaned == 0)
    dist_color[bg] = [80, 0, 0]  # Dark red background
    dist_vis = cv2.resize(dist_color, (w, h), interpolation=cv2.INTER_NEAREST)

    # Stage 3: Peaks + centroids on original
    peaks_vis = original_img.copy()
    for pt in centroids:
        cx, cy = pt[0]
        cv2.drawMarker(peaks_vis, (cx, cy), (255, 0, 0),
                       cv2.MARKER_STAR, markerSize=15, thickness=2)

    # Stage 4: Watershed regions colored
    np.random.seed(42)
    region_colors = np.random.randint(80, 255, size=(len(coords) + 3, 3))
    ws_vis = np.zeros((rows, cols, 3), dtype=np.uint8)
    for label in np.unique(markers):
        if label == -1:
            ws_vis[markers == -1] = [255, 255, 0]  # boundaries in yellow
        elif label == 1:
            ws_vis[markers == 1] = [30, 30, 30]  # background
        elif label > 1:
            ws_vis[markers == label] = region_colors[label]
    ws_vis = cv2.resize(ws_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    return {
        'original': original_img,
        'mask': mask_rgb,
        'distance': dist_vis,
        'peaks': peaks_vis,
        'watershed_regions': ws_vis,
        'centroids': centroids,
    }


def run_sam(original_img, centroids, model, processor, device):
    """Run SAM on the centroids and return mask/boundary visuals."""
    batched_points = [[pt[0]] for pt in centroids]
    image_pil = Image.fromarray(original_img)

    inputs = processor(image_pil, input_points=[batched_points], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores.cpu().numpy()

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(centroids), 3))

    # SAM masks overlay
    sam_masks_vis = original_img.copy()
    sam_bounds_vis = original_img.copy()

    for i in range(len(centroids)):
        best_idx = scores[0][i].argmax()
        best_mask = masks[0][i][best_idx].numpy()
        binary = (best_mask > 0.5).astype(np.uint8)
        color = colors[i].tolist()

        # Mask overlay
        overlay = np.zeros_like(sam_masks_vis)
        overlay[binary == 1] = color
        sam_masks_vis = cv2.addWeighted(sam_masks_vis, 1.0, overlay, 0.5, 0)

        # Boundary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sam_bounds_vis, contours, -1, color, 2)

    return sam_masks_vis, sam_bounds_vis


def compose_visualization(stages, sam_masks, sam_bounds, basename, total_time):
    """
    Create a 2-row visualization:
      Row 1: Original | Nucleus Mask | Distance Transform | Peaks
      Row 2: Watershed Regions | SAM Masks | SAM Boundaries
    """
    h, w = stages['original'].shape[:2]
    
    # Target panel size (scale down for reasonable output)
    pw, ph = 480, int(480 * h / w)
    
    panels_row1 = [
        ("Original", stages['original']),
        ("LAB Nucleus Mask", stages['mask']),
        ("Distance Transform", stages['distance']),
        (f"Centroids ({len(stages['centroids'])} WBCs)", stages['peaks']),
    ]
    panels_row2 = [
        ("Watershed Regions", stages['watershed_regions']),
        ("SAM Masks", sam_masks),
        ("SAM Boundaries", sam_bounds),
    ]

    gap = 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Row 1: 4 panels
    n1 = len(panels_row1)
    row1_w = n1 * pw + (n1 + 1) * gap
    row1_h = ph + 50

    # Row 2: 3 panels, centered
    n2 = len(panels_row2)
    row2_w = n2 * pw + (n2 + 1) * gap

    canvas_w = max(row1_w, row2_w)
    canvas_h = row1_h + ph + 50 + 80  # rows + title + footer
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

    # Title
    title = f"Improved Watershed + SAM Pipeline  |  {basename}  |  {total_time:.1f}s"
    ts = cv2.getTextSize(title, font, 0.8, 2)[0]
    cv2.putText(canvas, title, ((canvas_w - ts[0]) // 2, 30), font, 0.8, (30, 30, 30), 2)

    y_start = 45

    # Draw Row 1
    for i, (label, panel) in enumerate(panels_row1):
        resized = cv2.resize(panel, (pw, ph))
        x = gap + i * (pw + gap)
        y = y_start

        # Label
        ts = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.putText(canvas, label, (x + (pw - ts[0])//2, y + 15), font, 0.5, (60, 60, 60), 1)
        canvas[y + 20:y + 20 + ph, x:x + pw] = resized

    y_start2 = y_start + ph + 35

    # Draw Row 2 (centered)
    x_offset_r2 = (canvas_w - row2_w) // 2
    for i, (label, panel) in enumerate(panels_row2):
        resized = cv2.resize(panel, (pw, ph))
        x = x_offset_r2 + gap + i * (pw + gap)
        y = y_start2

        ts = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.putText(canvas, label, (x + (pw - ts[0])//2, y + 15), font, 0.5, (60, 60, 60), 1)
        canvas[y + 20:y + 20 + ph, x:x + pw] = resized

    return canvas


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════
print("Loading SAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print(f"SAM loaded on {device}.\n")

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue

    basename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"{'='*60}")
    print(f"Processing: {basename}")

    t0 = time.time()

    # Stage 1: Watershed with all intermediate visuals
    print("  Running watershed stages...")
    stages = get_watershed_stages(img_path)
    if stages is None:
        print("  Failed to read image!")
        continue
    print(f"  Found {len(stages['centroids'])} centroids")

    if not stages['centroids']:
        print("  No centroids, skipping SAM.")
        continue

    # Stage 2: SAM
    print("  Running SAM...")
    sam_masks, sam_bounds = run_sam(
        stages['original'], stages['centroids'], model, processor, device
    )

    total_time = time.time() - t0

    # Compose and save
    canvas = compose_visualization(stages, sam_masks, sam_bounds, basename, total_time)
    out_path = os.path.join(OUTPUT_DIR, f"pipeline_stages_{basename}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out_path} ({total_time:.1f}s)")

print(f"\n{'='*60}")
print("All done!")
