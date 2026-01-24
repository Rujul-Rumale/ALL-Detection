# Progress Notes

## Done so far

### MATLAB Demo
- Basic segmentation working (K-means on LAB)
- Added cell extraction - saves cropped cells to folder
- Shows 6-panel visualization

### Python Version
Started implementing python version:
- Segmentation module ported from MATLAB
- Dataset loader for C-NMC 
- Visualization utils (plots etc)
- Demo scripts to test things

### Datasets
Downloaded and organized:
- C-NMC dataset (~10k cells for training)
- ALL-IDB1 (full images for testing segmentation)
- Healthy-WBC (bonus dataset)

## What's left

- Train the actual model (MobileNetV3)
- Export to tflite for raspberry pi
- Grad-CAM implementation
- Build GUI
- Test everything end to end

## Issues/Notes

- Need to test on more images to check if segmentation is reliable
- Training might take a while, need to set up properly
- Pi deployment will need some optimization probably
