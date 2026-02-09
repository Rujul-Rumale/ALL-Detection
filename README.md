# ALL Detection Project

Automated leukemia detection from blood microscope images.

## Overview

This project detects Acute Lymphoblastic Leukemia (ALL) from blood cell images using computer vision + deep learning.

**How it works:**
1. Segment white blood cells using K-Means clustering on LAB color space
2. Classify each cell as healthy or ALL using MobileNetV3
3. Show results with Grad-CAM heatmaps (explainability)

## Datasets Used

- **C-NMC Dataset** - ~10,600 labeled cells for training (main dataset)
- **ALL-IDB1** - For testing segmentation on full blood smear images

See DATASETS.md for more info.

## Running the Code

### Test folder (MATLAB stuff - old demo)
```bash
cd test
# Open test.m in MATLAB and run it
```

### Python Implementation
```bash
pip install -r requirements.txt

# View dataset info and samples
python src/demo_dataset.py

# Test WBC segmentation (opens file picker)
python src/demo_segmentation.py
```

## Current Status

✅ WBC segmentation working (K-means on a*b* channels)  
✅ Dataset loader complete (3-fold CV ready)  
✅ Fixed over-segmentation issue with bilobed nuclei  
⏳ Model training script - next step  
⏳ TFLite export for Pi deployment  
⏳ Grad-CAM visualization  

## Project Structure

```
src/
├── segmentation/  - WBC segmentation (K-means)
├── classifier/    - Dataset loader, will add model here
├── utils/         - Visualization tools
└── xai/           - Grad-CAM (todo)
```

More details in PROJECT_STRUCTURE.md

## To-Do
- Train MobileNetV3 classifier
- Export to TFLite (for raspberry pi)
- Add Grad-CAM
- Build inference pipeline
- Make GUI maybe?
