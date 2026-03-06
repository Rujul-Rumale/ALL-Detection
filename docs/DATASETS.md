# Datasets Info

## What We're Using

### 1. C-NMC_Dataset ✅ **Main Dataset**
- **Source:** Cancer Image Archive (C-NMC Challenge 2019)
- **Why:** Binary ALL classification (exactly what we need)
- **Structure:**
  ```
  C-NMC_Dataset/
  ├── fold_0/fold_0/
  │   ├── all/    # Leukemia cells
  │   └── hem/    # Healthy cells
  ├── fold_1/fold_1/
  └── fold_2/fold_2/
  ```
- **Total:** ~10,600 pre-cropped cell images
- **Format:** BMP files (kinda large but whatever)
- **Labels:** Binary - 0=healthy (hem), 1=ALL

**Split for training:**
- fold_0 = training set (~7k images)
- fold_1 = validation set (~1.8k images) 
- fold_2 = test set (~1.8k images)

### 2. ALL_IDB Dataset
- **Purpose:** Testing segmentation on full blood smear images
- **Contains:** 
  - ALL_IDB1: Full microscopy images (not cropped)
  - Some have ALL cells, some dont
- **Use:** Test our K-means segmentation before classification
- **Note:** No per-cell labels, just for seeing if segmentation works

### 3. Healthy_WBC_Dataset
- **Purpose:** Multi-class WBC classification  
- **Classes:** Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil (5 types)
- **Structure:**
  ```
  Healthy_WBC_Dataset/
  ├── Train/      # ~10,000 images
  ├── Test-A/     # ~4,300 images
  └── Test-B/     # ~2,100 images
  ```
- **Status:** Not using for now, maybe later if we want to do multi-class stuff

---

## Current Approach

Using **C-NMC** for training the classifier:
- Already has labels (healthy vs ALL)
- Big dataset = better training
- Pre-cropped so no need to segment for training
- Standard benchmark dataset

For testing full pipeline:
- Use ALL-IDB1 images
- Run our K-means segmentation
- Extract cells
- Classify with trained model
- See if it works end-to-end
