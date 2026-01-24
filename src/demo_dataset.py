"""
Demo: Inspect C-NMC dataset
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.dataset import create_train_val_test_split
from utils.visualization import plot_class_distribution
import matplotlib.pyplot as plt


def main():
    # Path to C-NMC dataset
    dataset_root = Path("C-NMC_Dataset")
    
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return
    
    print("Loading C-NMC dataset...")
    train_ds, val_ds, test_ds = create_train_val_test_split(dataset_root)
    
    # Visualize distributions
    print("\nVisualizing class distributions...")
    
    fig1 = plot_class_distribution(train_ds.get_class_distribution())
    plt.title('Training Set Distribution')
    
    fig2 = plot_class_distribution(val_ds.get_class_distribution())
    plt.title('Validation Set Distribution')
    
    fig3 = plot_class_distribution(test_ds.get_class_distribution())
    plt.title('Test Set Distribution')
    
    # Show sample images
    print("\nLoading sample images...")
    fig4, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig4.suptitle('Sample Images (Top: Healthy, Bottom: ALL)')
    
    # Healthy samples
    for i in range(5):
        img, label = train_ds[i]
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
    
    # ALL samples (find first ALL image)
    all_idx = [i for i, lbl in enumerate(train_ds.labels) if lbl == 1][:5]
    for i, idx in enumerate(all_idx):
        img, label = train_ds[idx]
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
