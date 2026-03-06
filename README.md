# ALL Detection System

Automated Acute Lymphoblastic Leukemia (ALL) detection from peripheral blood smear images, powered by deep learning and on-device AI.

## How It Works

1. **Cell Localization** — K-Means clustering in L\*a\*b\* color space identifies WBC nuclei, followed by Euclidean Distance Transform and marker-controlled Watershed to split clumped cells into individual centroids
2. **Precision Segmentation** — Centroids are passed as point prompts to the Segment Anything Model (SAM ViT-Base), which traces precise cell boundaries in a single batched forward pass
3. **Crop Standardization** — Cells are extracted into CNMC-matched 128×128 crops with scale normalization (~75% fill ratio), mask centroid centering, and background masking
4. **Classification** — Each cell is classified as ALL (blast) or HEM (healthy) by a MobileNetV3-Large TFLite model with 4-orientation test-time augmentation
5. **AI Summary** — An on-device LLM (Phi-3 via Ollama) generates a clinical-style interpretation of the results
6. **Visualization** — Results are displayed in a real-time dashboard with cell cards, SAM contour annotations, and confidence scores

## Quick Start

### Desktop (Windows/Linux/macOS)

```bash
pip install -r requirements.txt
cd src
python ui/classification_demo.py
```

### Raspberry Pi 5

```bash
chmod +x setup_pi.sh
./setup_pi.sh    # Installs everything automatically
./run.sh         # Launch the app
```

The setup script handles: Python dependencies, Ollama installation, model download, desktop shortcut.

## Project Structure

```
src/
├── ui/                          # Desktop GUI (CustomTkinter)
│   ├── classification_demo.py   # Main application (Watershed → SAM → Classify)
│   ├── app.py                   # Lightweight edge app (K-Means fallback)
│   └── theme.py                 # Design system
├── detection/                   # Core detection pipeline
│   ├── demo_pipeline.py         # Production pipeline (Watershed → SAM → TFLite)
│   ├── generate_cell_crops_sam.py  # Watershed centroid finder + SAM crop generator
│   ├── blast_detector_v5.py     # Lightweight edge-only detector (K-Means + L1 scoring)
│   ├── stage1_screening.py      # TFLite screening model wrapper
│   └── llm_utils.py             # AI summary generation (Ollama / Phi-3)
├── segmentation/                # WBC segmentation utilities
├── classifier/                  # Dataset loader + model definitions
└── utils/                       # Visualization, preprocessing utilities

models/                  # Trained model files (.tflite)
training_scripts/        # Model training scripts (PyTorch)
evaluation_scripts/      # ROC analysis, validation evaluation
benchmarks/              # Segmentation method comparison scripts
evaluation_outputs/      # Generated figures (ROC curves, confusion matrices)
docs/                    # Technical documentation + evaluation reports
data/                    # Reference images, processed crops
tests/                   # Development test scripts
```

## Datasets

| Dataset | Purpose | Details |
|---------|---------|---------|
| **C-NMC 2019** | Training & Validation | 12,528 pre-cropped cell images, strict patient-level 3-fold CV split |
| **ALL-IDB1** | End-to-end Testing | Full blood smear images for pipeline evaluation |
| **ALL-IDB2** | Segmentation Benchmark | 260 pre-cropped cells for cross-dataset testing |

See [DATASETS.md](DATASETS.md) for full details.

## Key Results

| Metric | Value |
|--------|-------|
| AUC (ROC) | 0.934 |
| Sensitivity @ t=0.50 | 95.3% (with 4x TTA) |
| Specificity @ t=0.50 | 80.2% |
| F1 Score | 0.890 |
| Model Size (TFLite) | 3.38 MB |
| Inference Latency | 22 ms/cell |

## Hardware Targets

- **Primary**: Raspberry Pi 5 (8GB) + 1080p display
- **Secondary**: Android devices (via companion app)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Cell Localization | K-Means (L\*a\*b\*) + Watershed + SAM (ViT-Base) |
| Classification | MobileNetV3-Large (TFLite) distilled via Online KD from EffB4 |
| Optimization Pipeline | Albumentations, Differential Learning Rates (AdamW), Weighted Sampler |
| AI Summary | Phi-3 Mini (3.8B) via Ollama |
| Desktop GUI | CustomTkinter (Python) |
| Terminal UI | Rich Dashboard (Real-time GPU monitoring) |
| Segmentation | OpenCV, scikit-image, HuggingFace Transformers |

## Requirements

See [requirements.txt](requirements.txt) for Python dependencies.

**System Requirements:**
- Python 3.10+
- Ollama (for AI summaries)
- 4GB+ RAM (8GB recommended for LLM + SAM)

## Citations & References

### Datasets

- **C-NMC 2019 (ISBI Challenge):** Gupta, A., & Gupta, R. (2019). *ALL Challenge dataset of ISBI 2019* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r
- **ALL-IDB:** Labati, R. D., Piuri, V., & Scotti, F. (2011). *ALL-IDB: The Acute Lymphoblastic Leukemia Image Database for Image Processing.* Proceedings of the IEEE International Conference on Image Processing (ICIP), pp. 2045–2048.

### Models & Architectures

- **MobileNetV3:** Howard, A., et al. (2019). *Searching for MobileNetV3.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324.
- **EfficientNet:** Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* Proceedings of the 36th International Conference on Machine Learning (ICML), pp. 6105–6114.
- **Segment Anything (SAM):** Kirillov, A., et al. (2023). *Segment Anything.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4015–4026.
- **Knowledge Distillation:** Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv preprint arXiv:1503.02531.

### Frameworks & Libraries

- **PyTorch:** Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS), 32.
- **timm:** Wightman, R. (2019). *PyTorch Image Models (timm).* GitHub. https://github.com/huggingface/pytorch-image-models
- **Albumentations:** Buslaev, A., et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* Information, 11(2), 125.
- **scikit-learn:** Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
- **OpenCV:** Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
- **Ollama / Phi-3:** Microsoft. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.* arXiv preprint arXiv:2404.14219.

### Methods

- **Watershed Segmentation:** Meyer, F. (1994). *Topographic distance and watershed lines.* Signal Processing, 38(1), 113–125.
- **Macenko Stain Normalization:** Macenko, M., et al. (2009). *A Method for Normalizing Histology Slides for Quantitative Analysis.* Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI), pp. 1107–1110.
- **Test-Time Augmentation (TTA):** Shanmugam, D., et al. (2021). *Better Aggregation in Test-Time Augmentation.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1214–1223.
