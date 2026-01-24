"""
Visualization utilities for segmentation and classification results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_segmentation(image, results, save_path=None):
    """
    Create MATLAB-style 6-panel visualization of segmentation results.
    
    Args:
        image: Original RGB image
        results: Output dict from WBCSegmenter.segment()
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Panel 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Input')
    ax1.axis('on')
    
    # Panel 2: a* channel
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['intermediate']['lab'][:, :, 1], cmap='gray')
    ax2.set_title('a* channel')
    ax2.axis('on')
    
    # Panel 3: b* channel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results['intermediate']['lab'][:, :, 2], cmap='gray')
    ax3.set_title('b* channel')
    ax3.axis('on')
    
    # Panel 4: K-Means labels
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(results['intermediate']['labels'], cmap='jet')
    wbc_cluster = results['intermediate']['wbc_cluster']
    ax4.set_title(f'Clusters (WBC=#{wbc_cluster})')
    ax4.axis('on')
    
    # Panel 5: Cleaned mask
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(results['mask'], cmap='gray')
    ax5.set_title('Cleaned mask')
    ax5.axis('on')
    
    # Panel 6: Detected boundaries
    ax6 = fig.add_subplot(gs[1, 2])
    overlay = image.copy()
    
    # Draw bounding boxes
    for bbox in results['bboxes']:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    ax6.imshow(overlay)
    ax6.set_title(f"Found {results['count']} WBCs")
    ax6.axis('on')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_cell_montage(cells, max_cols=6, save_path=None):
    """
    Display extracted cells in a grid montage.
    
    Args:
        cells: List of cell images
        max_cols: Maximum columns in grid
        save_path: Optional path to save figure
    """
    n_cells = len(cells)
    n_cols = min(n_cells, max_cols)
    n_rows = int(np.ceil(n_cells / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    
    if n_cells == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, cell in enumerate(cells):
        axes[i].imshow(cell)
        axes[i].set_title(f'Cell {i+1}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_cells, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_class_distribution(dataset_dict):
    """
    Plot class distribution bar chart.
    
    Args:
        dataset_dict: Dict like {'hem': 5000, 'all': 5000}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    classes = list(dataset_dict.keys())
    counts = list(dataset_dict.values())
    
    bars = ax.bar(classes, counts, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
