# ALL Detection

Acute Lymphoblastic Leukemia (ALL) detection using digital signal processing and deep learning.

## Overview

This project implements an end-to-end system for ALL detection from blood smear microscopy images:

1. **DSP Pipeline:** Segments white blood cells using K-Means clustering on L*a*b* color space
2. **AI Classifier:** MobileNetV3-Small trained on 10K+ labeled cells
3. **Explainability:** Grad-CAM visualizations for prediction transparency

## Datasets

| Dataset | Purpose | Details |
|---------|---------|---------|
| **C-NMC** | Classifier training | ~10,600 pre-segmented cells (ALL vs healthy) |
| **ALL-IDB1** | DSP pipeline demo | Full blood smear images for segmentation |
| **Healthy-WBC** | Future enhancement | 5-class WBC type classification |

See [DATASETS.md](DATASETS.md) for details.

## Project Structure

```
ALL-Detection/
├── matlab_demo/          # MATLAB prototype (segmentation validated)
├── src/                  # Python implementation
│   ├── segmentation/     # DSP pipeline (ALL-IDB1 methods)
│   ├── classifier/       # MobileNetV3 training (C-NMC data)
│   ├── xai/              # Grad-CAM
│   └── gui/              # PyQt5 application
├── C-NMC_Dataset/        # Training data (pre-segmented cells)
├── ALL_IDB/              # Deployment test data (full images)
└── models/               # Trained .tflite models
```

## Quick Start

### MATLAB Demo
1. Run `matlab_demo/leukemia_dsp_demo.m`
2. Select a blood smear image from ALL-IDB1
3. View segmentation results

### Python (Coming Soon)
```bash
# Install dependencies
pip install -r requirements.txt

# Train classifier
python src/train.py --dataset C-NMC_Dataset/fold_0

# Run inference on new image
python src/infer.py --image path/to/blood_smear.jpg
```

## Method

### Training Phase
```
C-NMC pre-cropped cells → MobileNetV3 training → Trained model
```

### Deployment Phase
```
Raw microscope image
  ↓ DSP Segmentation (LAB + K-Means + Morphology)
  ↓ Cropped cells
  ↓ Classifier (trained on C-NMC)
  ↓ Results + Grad-CAM explanations
```

## License

MIT
