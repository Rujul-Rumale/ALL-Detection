# ALL Detection System

Automated Acute Lymphoblastic Leukemia (ALL) detection from peripheral blood smear images, powered by deep learning and on-device AI.

## How It Works

1. **Segmentation** — White blood cells are extracted from full blood smear images using K-Means clustering on the LAB color space
2. **Classification** — Each cell is analyzed by a MobileNetV3 model (TFLite) to determine if it is a healthy WBC or a blast cell (suspected ALL)
3. **AI Summary** — An on-device LLM (Phi-3 via Ollama) generates a clinical-style interpretation of the results
4. **Visualization** — Results are displayed in a real-time dashboard with cell cards, annotations, and confidence scores

## Quick Start

### Desktop (Windows/Linux/macOS)

```bash
pip install -r requirements.txt
cd src
python ui/app.py
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
├── ui/                  # Desktop GUI (CustomTkinter)
│   ├── app.py           # Main application
│   └── theme.py         # Design system
├── detection/           # Core detection pipeline
│   ├── blast_detector_v5.py  # Cell detection + classification
│   ├── llm_utils.py     # AI summary generation (Ollama)
│   └── stage1_screening.py   # TFLite screening model
├── segmentation/        # WBC segmentation (K-Means)
├── classifier/          # Dataset loader + model definitions
├── training/            # Model training scripts (reference)
├── api/                 # REST API for Android app (planned)
└── utils/               # Visualization, preprocessing utilities

android_app/             # Android companion app (Jetpack Compose)
models/                  # Trained model files (.tflite)
data/                    # Reference images, processed data
docs/                    # Research papers, technical docs
```

## Datasets

| Dataset | Purpose | Details |
|---------|---------|---------|
| **C-NMC 2019** | Training | ~10,600 pre-cropped cell images (ALL vs Healthy) |
| **ALL-IDB1** | Testing | Full blood smear images for end-to-end pipeline testing |

See [DATASETS.md](DATASETS.md) for full details.

## Hardware Targets

- **Primary**: Raspberry Pi 5 (8GB) + 1080p display
- **Secondary**: Android devices (via companion app)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection Model | MobileNetV3 (TFLite) |
| AI Summary | Phi-3 via Ollama |
| Desktop GUI | CustomTkinter (Python) |
| Android App | Jetpack Compose + Material 3 |
| Segmentation | OpenCV, scikit-image |

## Requirements

See [requirements.txt](requirements.txt) for Python dependencies.

**System Requirements:**
- Python 3.10+
- Ollama (for AI summaries)
- 4GB+ RAM (8GB recommended for LLM)
