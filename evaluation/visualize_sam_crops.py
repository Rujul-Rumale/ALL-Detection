import os
import cv2
import matplotlib.pyplot as plt
import random

CROP_DIR = r"c:\Open Source\leukiemea\data\processed_crops_sam"
OUT_IMG = r"c:\Open Source\leukiemea\evaluation\sam_crop_gallery.jpg"

crops = [os.path.join(CROP_DIR, f) for f in os.listdir(CROP_DIR) if f.endswith('.jpg')]
random.seed(42)
sample_crops = random.sample(crops, min(16, len(crops)))

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle("SAM Isolated 128x128 WBC Nucleus Crops", fontsize=16)

for ax, crop_path in zip(axes.flatten(), sample_crops):
    img = cv2.imread(crop_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(os.path.basename(crop_path)[:15], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150)
print(f"Gallery saved to {OUT_IMG}")
