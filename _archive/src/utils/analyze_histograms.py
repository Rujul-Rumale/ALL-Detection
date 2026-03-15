
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze():
    # Paths
    source_path = r"c:\Open Source\leukiemea\ALL_IDB\ALL_IDB Dataset\L1\extracted_cells\cell_008.jpg"
    template_path = r"c:\Open Source\leukiemea\C-NMC_Dataset\fold_0\fold_0\all\UID_11_10_1_all.bmp"
    
    img_s = cv2.imread(source_path)
    img_t = cv2.imread(template_path)
    
    if img_s is None or img_t is None:
        print("Failed to load images")
        return

    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
    
    print(f"Source Shape: {img_s.shape}, Mean: {img_s.mean(axis=(0,1))}")
    print(f"Template Shape: {img_t.shape}, Mean: {img_t.mean(axis=(0,1))}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    # Images
    ax[0,0].imshow(img_s)
    ax[0,0].set_title("Source (ALL-IDB)")
    ax[0,1].imshow(img_t)
    ax[0,1].set_title("Template (C-NMC)")
    
    # Histograms
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist_s = cv2.calcHist([img_s], [i], None, [256], [0, 256])
        ax[1,0].plot(hist_s, color=col)
        
        hist_t = cv2.calcHist([img_t], [i], None, [256], [0, 256])
        ax[1,1].plot(hist_t, color=col)
        
    ax[1,0].set_title("Source Histogram")
    ax[1,1].set_title("Template Histogram")
    
    plt.tight_layout()
    plt.savefig("histogram_analysis.png")
    print("Analysis saved to histogram_analysis.png")

if __name__ == "__main__":
    analyze()
