
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def visualize_classes():
    base_dir = r"c:\Open Source\leukiemea\C-NMC_Dataset\fold_0\fold_0"
    all_path = os.path.join(base_dir, "all")
    hem_path = os.path.join(base_dir, "hem")
    
    # Get all image files
    all_images = glob.glob(os.path.join(all_path, "*.bmp"))
    hem_images = glob.glob(os.path.join(hem_path, "*.bmp"))
    
    print(f"Found {len(all_images)} ALL images and {len(hem_images)} HEM images.")
    
    if not all_images or not hem_images:
        print("Error: Could not find images in one or more folders.")
        return

    # Select random samples
    k = 5
    selected_all = random.sample(all_images, k)
    selected_hem = random.sample(hem_images, k)
    
    # Create plot
    fig, axes = plt.subplots(2, k, figsize=(15, 6))
    plt.suptitle(f"C-NMC Dataset: Class Comparison (Fold 0)", fontsize=16)
    
    # Plot ALL
    for i, img_path in enumerate(selected_all):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        if i == 0:
            axes[0, i].set_ylabel("ALL (Malignant)", fontsize=14, fontweight='bold')
        axes[0, i].set_title(os.path.basename(img_path), fontsize=8)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
    # Plot HEM
    for i, img_path in enumerate(selected_hem):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        if i == 0:
            axes[1, i].set_ylabel("HEM (Healthy)", fontsize=14, fontweight='bold')
        axes[1, i].set_title(os.path.basename(img_path), fontsize=8)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
    plt.tight_layout()
    plt.savefig("cnmc_class_comparison.png")
    print("Saved comparison to cnmc_class_comparison.png")

if __name__ == "__main__":
    visualize_classes()
