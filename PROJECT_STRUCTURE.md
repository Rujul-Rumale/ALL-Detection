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
├── archive/             - Old downloads and temp files
├── models/              - Will store trained models here
└── notebooks/           - Jupyter notebooks (later)
```

**Datasets** (not tracked in git):
- `ALL_IDB/` - original dataset with full images
- `C-NMC_Dataset/` - training data with pre-cropped cells
- `Healthy_WBC_Dataset/` - bonus dataset for later

**Next steps:**
- Finish python training script
- Export model to .tflite
- Make raspberry pi GUI
