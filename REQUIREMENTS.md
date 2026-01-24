# ALL Detection - Requirements

## MATLAB Requirements

**MATLAB Version:** R2020a or later recommended

### Required Toolboxes
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

### Functions Used
| Function | Toolbox |
|----------|---------|
| `rgb2lab` | Image Processing Toolbox |
| `imresize` | Image Processing Toolbox |
| `imopen`, `imfill` | Image Processing Toolbox |
| `bwareaopen` | Image Processing Toolbox |
| `bwboundaries` | Image Processing Toolbox |
| `strel` | Image Processing Toolbox |
| `kmeans` | Statistics and Machine Learning Toolbox |

### Check Installation
Run this in MATLAB to verify:
```matlab
ver('images')
ver('stats')
```

---

## Dataset

Download the ALL-IDB dataset and place it in the `ALL_IDB/` folder:
- Source: https://homes.di.unimi.it/scotti/all/

---

## Python Requirements (Future)

For the Python implementation (if added):
```
numpy
opencv-python
scikit-learn
matplotlib
```
