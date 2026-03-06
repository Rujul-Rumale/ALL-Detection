# Project Summary: Automated Acute Lymphoblastic Leukemia (ALL) Detection System for Edge Devices

## 1. Project Overview & Objective
This project builds an automated, end-to-end computer vision and deep learning pipeline to detect Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. The primary objective is to develop a highly accurate, yet lightweight system capable of running inference on resource-constrained edge devices (specifically, a Raspberry Pi 5).

The system prioritizes **High Sensitivity (Recall)** to ensure that potential leukemia blasts are rarely missed, acting as a reliable first-line screening tool for medical professionals.

## 2. Core Technical Pipeline
The workflow is structured into four distinct phases: **Cell Extraction**, **Model Training & Optimization**, **Model Evaluation**, and **Edge Deployment**.

### Phase 1: Robust Cell Extraction (K-Means → Watershed → SAM)
Individual white blood cells (WBCs) must be isolated from complex whole-slide microscopic images. This is critical because background noise (Red Blood Cells, staining artifacts) confuses standard classifiers.

We employ a hybrid traditional + deep learning segmentation pipeline:

1. **K-Means Nucleus Identification**: The image is converted to L\*a\*b\* color space. K-Means clustering ($K=3$) on the a\* and b\* chrominance channels separates WBC nuclei (darkest cluster in L\* channel) from RBCs and background.

2. **Clump Splitting (EDT + Watershed)**: The binary nucleus mask undergoes morphological cleanup, then Euclidean Distance Transform (EDT) followed by `peak_local_max` to identify individual nuclei peaks within clumped cell clusters. These peaks seed a marker-controlled Watershed algorithm that labels individual cell regions.

3. **Debris Classification**: Each Watershed region is filtered by area (50–8000 px), circularity (≥0.40), solidity (≥0.50), and mean L\* intensity (≤65). Regions failing these criteria are classified as debris and tracked separately.

4. **SAM Precision Boundary Tracing**: Cell and debris centroids are passed as **point prompts** to HuggingFace's Segment Anything Model (SAM, `facebook/sam-vit-base`). SAM draws precise, smooth boundaries around each cell in a single batched forward pass — avoiding the computational overhead of SAM's automatic "segment everything" mode.

5. **CNMC-Matched Crop Extraction**: Each cell is extracted into a standardized 128×128 crop with four normalization steps: largest connected component isolation, bad mask rejection, mask centroid centering, and scale normalization (cell fills ~75% of frame). Background is masked to black.

### Phase 2: Model Architecture & PyTorch Training
With thousands of isolated cell crops, we trained a classifier to distinguish between `ALL` (Leukemic blasts) and `HEM` (Healthy cells).

- **Dataset**: The expert-annotated **C-NMC 2019 dataset** (~10,600 images), split by patient ID to prevent data leakage.
- **Architecture**: **MobileNetV3-Large** (ImageNet V2 pretrained), with a custom classification head: Linear(960→256) → HardSwish → Dropout(0.4) → Linear(256→2).
- **Training Framework**: Native PyTorch with local GPU acceleration (NVIDIA RTX 3050), ~45 seconds per epoch.
- **Training Strategy**: 2-Phase progressive fine-tuning:
  1. **Phase 1 (Epochs 1–3)**: Frozen backbone; train only the custom classification head.
  2. **Phase 2 (Epochs 4–30)**: Full model unfreeze with reduced learning rate ($\text{lr} = 10^{-4}$), AdamW optimizer with cosine annealing schedule, early stopping (patience 8, monitoring F1).
- **Class Imbalance Handling**: Inverse-frequency class weighting ($w_c = N / 2N_c$) in CrossEntropyLoss with label smoothing ($\epsilon = 0.1$). This forces the model to penalize missing an ALL cell more heavily than misclassifying a healthy cell.
- **Data Augmentation**: Aggressive pipeline including random crops, flips (H/V), rotation (0–180°), affine transforms, color jitter, Gaussian blur, and random erasing.

### Phase 3: Model Evaluation & ROC Analysis
We systematically evaluated the models on the 3,553-image C-NMC validation set with full ROC threshold analysis.

**MobileNetV3-Large v2 with Test-Time Augmentation (4 orientations):**

| Threshold | Sensitivity | Specificity | F1 Score | Notes |
|-----------|------------|-------------|----------|-------|
| t=0.50 (default) | **91.7%** | 60.9% | 0.877 | Best for screening |
| t=0.572 (Youden optimal) | 82.6% | 71.9% | 0.847 | Balanced operating point |
| **AUC** | **0.84** | — | — | Area Under ROC Curve |

**Test-Time Augmentation (TTA)**: At inference, each cell is evaluated across 4 orientations (original, H-flip, V-flip, both flips). Softmax probabilities are averaged, improving robustness to orientation variation.

**Segmentation Benchmark**: We evaluated 8 segmentation methods on the ALL-IDB2 dataset (clumped samples). Our hybrid K-Means → Watershed → SAM approach produced the best accuracy-feasibility tradeoff. SAM automatic mode (277 sub-masks, 75s), Cellpose (memory overflow), and YOLO (failed generalization) were all rejected for production use.

### Phase 4: Export to Edge Format (TFLite)
The PyTorch model was exported through a multi-stage conversion pipeline:

$$\text{PyTorch (.pth)} \xrightarrow{\text{ONNX}} \text{ONNX (.onnx)} \xrightarrow{\text{onnx-tf}} \text{TF SavedModel} \xrightarrow{\text{TFLite}} \text{TFLite (.tflite)}$$

- **Final Model Size**: `mobilenetv3_large_v2.tflite` — **3.38 MB**
- **Inference Latency**: 22 ms per cell on Raspberry Pi 5 (CPU)
- **Conclusion**: We achieved strong screening sensitivity (91.7% @ t=0.50, AUC 0.84) while keeping the model file size vastly below the 25 MB edge-computing threshold. The system integrates an on-device LLM (Phi-3 via Ollama) for clinical-style interpretation, completing the assistive diagnostic workflow.
