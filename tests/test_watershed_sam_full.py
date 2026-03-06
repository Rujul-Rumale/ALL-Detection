"""
Full Watershed + SAM pipeline test.
Generates a 3-panel visual: Centroids | SAM Masks | WBC Boundaries
Similar to the existing evaluation images.
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from detection.generate_cell_crops_sam import get_watershed_centroids

# Test image
IMAGE_PATH = r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L2\Im003_1.jpg"
OUTPUT_DIR = r"c:\Open Source\leukiemea\temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# ── Stage 1: Watershed Centroids ──
print("Running improved watershed...")
t0 = time.time()
centroids, original_img = get_watershed_centroids(IMAGE_PATH)
t_ws = time.time() - t0
print(f"  Found {len(centroids)} centroids in {t_ws:.1f}s")

if not centroids:
    print("No centroids found!")
    sys.exit(1)

# Panel 1: Centroids on original
panel_centroids = original_img.copy()
for pt in centroids:
    cx, cy = pt[0]
    cv2.drawMarker(panel_centroids, (cx, cy), (255, 0, 0), 
                   cv2.MARKER_STAR, markerSize=15, thickness=2)

# ── Stage 2: SAM Segmentation ──
print("Loading SAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print(f"SAM loaded on {device}.")

batched_points = [[pt[0]] for pt in centroids]
image_pil = Image.fromarray(original_img)

print("Running SAM inference...")
t0 = time.time()
inputs = processor(image_pil, input_points=[batched_points], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores.cpu().numpy()
t_sam = time.time() - t0
print(f"  SAM inference done in {t_sam:.1f}s")

# Panel 2: SAM masks overlay (colored)
panel_masks = original_img.copy()
np.random.seed(42)
colors = np.random.randint(50, 255, size=(len(centroids), 3))

for i in range(len(centroids)):
    best_idx = scores[0][i].argmax()
    best_mask = masks[0][i][best_idx].numpy()
    binary = (best_mask > 0.5).astype(np.uint8)
    
    color = colors[i].tolist()
    overlay = np.zeros_like(panel_masks)
    overlay[binary == 1] = color
    panel_masks = cv2.addWeighted(panel_masks, 1.0, overlay, 0.5, 0)

# Panel 3: Boundaries on original
panel_boundaries = original_img.copy()
for i in range(len(centroids)):
    best_idx = scores[0][i].argmax()
    best_mask = masks[0][i][best_idx].numpy()
    binary = (best_mask > 0.5).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[i].tolist()
    cv2.drawContours(panel_boundaries, contours, -1, color, 2)

# ── Compose 3-panel output ──
h, w = original_img.shape[:2]
gap = 20
canvas_w = w * 3 + gap * 4
canvas_h = h + 100
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

# Labels
font = cv2.FONT_HERSHEY_SIMPLEX
titles = [
    f"Watershed Centroids ({len(centroids)} WBCs)",
    f"SAM Prompted Masks ({len(centroids)} WBCs only)",
    "WBC Boundaries (Watershed+SAM)"
]

for i, (panel, title) in enumerate(zip(
    [panel_centroids, panel_masks, panel_boundaries], titles)):
    x_offset = gap + i * (w + gap)
    
    # Title
    text_size = cv2.getTextSize(title, font, 0.7, 2)[0]
    text_x = x_offset + (w - text_size[0]) // 2
    cv2.putText(canvas, title, (text_x, 35), font, 0.7, (0, 0, 0), 2)
    
    # Panel image
    canvas[50:50+h, x_offset:x_offset+w] = panel

# Title bar
total_time = t_ws + t_sam
main_title = f"Improved Watershed -> SAM Pipeline | {basename}.jpg | {total_time:.1f}s total"
text_size = cv2.getTextSize(main_title, font, 0.6, 1)[0]
# Put at bottom
cv2.putText(canvas, main_title, 
            ((canvas_w - text_size[0]) // 2, canvas_h - 20),
            font, 0.6, (80, 80, 80), 1)

out_path = os.path.join(OUTPUT_DIR, f"watershed_sam_improved_{basename}.jpg")
cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"\nSaved: {out_path}")
print(f"Total time: {total_time:.1f}s")
