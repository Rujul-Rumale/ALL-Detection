# ALL Detection

Acute Lymphoblastic Leukemia (ALL) detection using digital signal processing techniques.

## Overview

This project implements White Blood Cell (WBC) segmentation from blood smear images using K-Means clustering on the L*a*b* color space. The segmented cells can be used for leukemia detection and analysis.

## Project Structure

```
ALL-Detection/
├── matlab_demo/          # MATLAB implementation
│   └── leukemia_dsp_demo.m
├── python_impl/          # Python implementation (WIP)
├── ALL_IDB/              # Dataset (not tracked)
└── REQUIREMENTS.md       # Dependencies
```

## Quick Start

### MATLAB
1. Install required toolboxes (see REQUIREMENTS.md)
2. Run `matlab_demo/leukemia_dsp_demo.m`
3. Select a blood smear image when prompted

## Method

1. Load blood smear image
2. Convert RGB to L*a*b* color space
3. Apply K-Means clustering (k=3) on a*b* channels
4. Identify WBC cluster (darkest in L channel)
5. Clean mask with morphological operations
6. Extract and overlay cell boundaries

## Dataset

Uses ALL-IDB (Acute Lymphoblastic Leukemia Image Database):
https://homes.di.unimi.it/scotti/all/

## License

MIT
