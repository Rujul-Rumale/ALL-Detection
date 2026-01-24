"""
WBC Segmentation - K-Means on LAB color space
Based on the MATLAB version
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
from pathlib import Path


class WBCSegmenter:
    # segments white blood cells from microscope images
    
    def __init__(self, num_clusters=3, min_cell_size=200, padding=10):
        self.num_clusters = num_clusters
        self.min_cell_size = min_cell_size
        self.padding = padding
    
    def segment(self, image):
        # main segmentation function
        # returns dict with mask, cells, bboxes etc
        
        # convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        ab = lab[:, :, 1:3].astype(np.float32)
        rows, cols = ab.shape[:2]
        ab_vec = ab.reshape(-1, 2)
        
        # run kmeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(ab_vec)
        labels = labels.reshape(rows, cols)
        
        # find which cluster has WBCs (darkest L channel)
        L = lab[:, :, 0]
        avg_L = []
        for i in range(self.num_clusters):
            mask_i = (labels == i)
            avg_L.append(np.mean(L[mask_i]))
        
        wbc_cluster = np.argmin(avg_L)
        wbc_mask = (labels == wbc_cluster).astype(np.uint8)
        
        # morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(wbc_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = ndimage.binary_fill_holes(clean_mask).astype(np.uint8)
        
        # remove small stuff
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
            clean_mask, connectivity=8
        )
        
        final_mask = np.zeros_like(clean_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_cell_size:
                final_mask[labeled_mask == i] = 1
        
        # get individual cells
        cells, bboxes = self._extract_cells(image, final_mask)
        
        return {
            'mask': final_mask,
            'cells': cells,
            'bboxes': bboxes,
            'count': len(cells),
            'intermediate': {
                'lab': lab,
                'labels': labels,
                'wbc_cluster': wbc_cluster,
                'raw_mask': wbc_mask,
                'clean_mask': clean_mask
            }
        }
    
    def _extract_cells(self, image, mask):
        # crop out individual cells from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        bboxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # add some padding
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(image.shape[1] - x_pad, w + 2 * self.padding)
            h_pad = min(image.shape[0] - y_pad, h + 2 * self.padding)
            
            if w_pad > 20 and h_pad > 20:  # skip tiny cells
                cell_img = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                cells.append(cell_img)
                bboxes.append((x_pad, y_pad, w_pad, h_pad))
        
        return cells, bboxes
    
    def save_cells(self, cells, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, cell in enumerate(cells):
            filename = f"cell_{i+1:03d}.jpg"
            cell_bgr = cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path / filename), cell_bgr)
        
        return output_path


def segment_image(image_path, output_dir=None, visualize=False):
    # quick wrapper to segment a single image
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)  # resize to speed up
    
    segmenter = WBCSegmenter()
    results = segmenter.segment(image)
    
    if output_dir:
        save_path = segmenter.save_cells(results['cells'], output_dir)
        results['saved_to'] = str(save_path)
    
    return results
