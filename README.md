# ALL Detection

Leukemia detection from blood images using DSP and deep learning.

## What it does

Detects white blood cells from microscope images and classifies them as healthy or ALL (leukemia).

**Main parts:**
- DSP segmentation (K-Means on LAB color space)
- MobileNetV3 classifier trained on C-NMC dataset
- Grad-CAM for explanations

## Datasets

We're using **C-NMC** for training (~10k labeled cells) and **ALL-IDB1** for testing the segmentation.

## Quick Start

### MATLAB version
Just run `matlab_demo/leukemia_dsp_demo.m` and pick an image. It'll show the segmentation and extract individual cells.

### Python (WIP)
Still working on this part. So far:
```bash
pip install -r requirements.txt
python src/demo_dataset.py      # check dataset
python src/demo_segmentation.py # test segmentation
```

## How it works

**Training:**
- Use pre-cropped C-NMC cells to train classifier
- MobileNetV3-Small model

**Deployment:**
- Raw image → DSP segmentation (LAB + K-Means) → crop cells → classifier → results

## TODO
- [ ] Finish model training
- [ ] Add Grad-CAM
- [ ] Make GUI for raspberry pi
- [ ] Test on real images

## License
MIT
