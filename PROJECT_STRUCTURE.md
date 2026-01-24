# Project Organization

## Current Structure

```
ALL-Detection/
├── .git/                    # Git repository
├── .gitignore               # Ignore datasets, archives, models
├── README.md                # Project overview
├── REQUIREMENTS.md          # MATLAB toolboxes & Python deps
├── DATASETS.md              # Dataset documentation
│
├── matlab_demo/             # MATLAB prototype
│   └── leukemia_dsp_demo.m
│
├── scripts/                 # Utility scripts
│   ├── analyze_dataset.py
│   ├── analyze_datasets_full.py
│   ├── prepare_cnmc_dataset.py
│   └── sample_dataset.py
│
├── docs/                    # Documentation & papers
│   └── Leukemia_Detection_using_Digital_Image_Processing_.pdf
│
├── archive/                 # Old files & downloads (not tracked)
│   ├── C-NMC.zip
│   ├── Heathy-WBC-IDB.zip
│   └── dataset_samples/
│
├── ALL_IDB/                 # Original ALL-IDB1 dataset (not tracked)
├── C-NMC_Dataset/           # Training dataset ~10,600 cells (not tracked)
└── Healthy_WBC_Dataset/     # Bonus multi-class dataset (not tracked)
```

## Next Steps

Ready to create `src/` folder structure for Phase 1 implementation:
- `src/segmentation/` - DSP pipeline
- `src/classifier/` - Model training
- `src/xai/` - Grad-CAM
- `src/gui/` - PyQt5 app

## Files Not Tracked by Git

- Datasets (too large)
- Archive folder (temporary downloads)
- Future: models/, logs/, checkpoints/
