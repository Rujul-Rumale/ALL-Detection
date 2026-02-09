# Project Organization

Current folder structure:

```
ALL-Detection/
├── matlab_demo/         - MATLAB prototype
├── src/                 - Python implementation
│   ├── segmentation/    
│   ├── classifier/      
│   ├── xai/             
│   └── utils/           
├── docs/                - Papers and documentation
├── scripts/             - Helper scripts (currently empty)
├── models/              - Will store trained models here

```

**Datasets** (not tracked in git):
- `ALL_IDB/` - original dataset with full images
- `C-NMC_Dataset/` - training data with pre-cropped cells
- `Healthy_WBC_Dataset/` - bonus dataset for later
