# ALL Detection - Dataset Information

## Available Datasets

### 1. C-NMC_Dataset ✅ **USE THIS**
- **Source:** Cancer Image Archive (C-NMC Challenge 2019)
- **Purpose:** Binary ALL classification
- **Structure:**
  ```
  C-NMC_Dataset/
  ├── fold_0/
  │   ├── all/    # ALL blast cells
  │   └── hem/    # Healthy cells
  ├── fold_1/
  └── fold_2/
  ```
- **Total:** ~10,600 images (pre-split for 3-fold cross-validation)
- **Format:** BMP images
- **Labels:** Binary (all vs hem)

### 2. Healthy_WBC_Dataset
- **Purpose:** Multi-class WBC type classification
- **Classes:** Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil
- **Structure:**
  ```
  Healthy_WBC_Dataset/
  ├── Train/      # ~10,000 images
  ├── Test-A/     # ~4,300 images
  └── Test-B/     # ~2,100 images
  ```
- **Use case:** Future multi-class enhancement (not needed for initial ALL detection)

### 3. ALL_IDB Dataset (Original)
- **Purpose:** Reference/backup
- **Contains:** Full blood smear images (ALL-IDB1 style)
- **Status:** No per-cell labels, keep for documentation

---

## Recommendation

**Use C-NMC_Dataset** for training your MobileNetV3 classifier:
- Pre-labeled binary classification (perfect for your use case)
- Large dataset (better than ALL-IDB2's 260 images)
- Already cross-validation ready
- High-quality single-cell crops
