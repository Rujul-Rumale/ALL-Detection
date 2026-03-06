Technical Specification: Embedded AI Leukemia Detection System

Version: 1.1.0 (Post-Engineering Review)
Status: Implementation Phase
Target Hardware: Raspberry Pi 5 (Primary), Mobile/Edge Devices (Secondary)
Primary Metric: Recall (Sensitivity) @ >95%
Secondary Metrics: Precision, AUC (Reporting required)

1. Executive Summary

This project aims to build a low-cost, real-time automated diagnostic tool for Acute Lymphoblastic Leukemia (ALL). The system processes single-cell images extracted from blood smears, classifies them as "Malignant" (Lymphoblast) or "Normal" (Benign), and runs entirely offline on edge hardware.

The core challenge is balancing medical-grade sensitivity (missing cancer is unacceptable) with edge-device constraints (low latency, limited thermal headroom).

2. System Architecture

2.1 Hardware Stack

Compute: Raspberry Pi 5 (Broadcom BCM2712, Quad-core Cortex-A76).

Accelerator (Optional): Hailo-8L (via Pi AI Kit) or Google Coral TPU. Note: Baseline implementation targets CPU.

Camera/Input: Static image feed from digital microscope or slide scanner.

2.2 Software Stack

Training Host: Linux/Windows GPU Workstation.

Python 3.10+

TensorFlow 2.14+ / Keras

keras_cv (for Transformer models)

Edge Runtime:

OS: Raspberry Pi OS (Bookworm 64-bit)

Inference Engine: tflite-runtime (Version 2.14+)

Image Processing: OpenCV (headless version recommended)

3. Data Engineering Strategy

3.1 Datasets & Domain Shift Mitigation

We utilize a dual-source strategy to prevent overfitting to specific camera equipment.

Role

Dataset

Characteristics

Action Required

Training

C-NMC 2019

~15k images, stain-normalized, pre-segmented.

Use as Ground Truth distribution.

Validation

ALL-IDB 1/2

High-res full slides (IDB1) & crops (IDB2). Different stain color.

CRITICAL: Must apply Macenko Normalization to match C-NMC.

3.2 Preprocessing Pipeline (Strict Protocol)

Every image entering the model (Training OR Inference) must pass through this exact sequence:

ROI Extraction: Crop single cell from slide.

Square Padding:

Input: Rectangular crop (e.g., $40 \times 60$).

Operation: Pad the shorter dimension to match the longer dimension using cv2.BORDER_REFLECT or constant black.

Output: $60 \times 60$ square. Reason: Prevents aspect ratio distortion during resize.

Resize:

Target: $224 \times 224$ (MobileNet) or $256 \times 256$ (MobileViT).

Method: Bilinear or Bicubic Interpolation.

Stain Normalization (Macenko Method):

Reference Constraint: Compute stain matrix ONCE from reference_img.jpg and serialize it (e.g., stain_vectors.npz). Load this static file at runtime. DO NOT recompute reference statistics dynamically per batch.

Logic: Decompose RGB into Stain Vectors (Hematoxylin & Eosin). Normalize concentration to match the loaded C-NMC reference.

Fallback: If Macenko fails (singular matrix), use Histogram Matching.

Safety: Log usage of fallback. Abort training if fallback rate > 5%.

Tensor Normalization:

Convert to FLOAT32.

Scale: pixel_value / 255.0 (Range $0.0 \rightarrow 1.0$).

Constraint: Do not use ImageNet mean subtraction [0.485, ...].

3.3 Data Splitting Protocol

Patient-Level Split: When generating Train/Val/Test sets, splits must be done by Subject ID / Patient ID, not by image index.

Constraint: No single patient's cells may appear in both Training and Validation sets.

4. Model Engineering

4.1 Architecture Candidates

The repository supports selectable backbones via config.

Candidate A (Baseline): MobileNetV2

Alpha: 1.0

Input: $224 \times 224 \times 3$

Pros: Extremely stable, native TFLite support, fast on Pi 5 (~40ms).

Candidate B (High Speed): MobileNetV3-Small

Config: Minimalistic (HardSwish).

Pros: Fastest possible inference (<15ms). Best for video feeds.

Candidate C (High Accuracy): MobileViT-XS (Transformer)

Input: $256 \times 256 \times 3$

Pros: Captures global context better than CNNs.

Cons: Heavier; requires careful quantization.

Warning: Dynamic Range Quantization may degrade accuracy. Use Float16 or full-integer quantization if Recall drops below 90%.

4.2 Training Hyperparameters

Loss Function: BinaryCrossentropy

Optimizer: Adam (Learning Rate: 1e-4, Decay: 1e-6).

Batch Size: 32 (Host), 1 (Edge Inference).

Checkpointing:

Monitor: 'val_recall'

Mode: 'max' (Save only if Recall improves).

Class Imbalance Handling:

Calculate weights: $W_i = \frac{N_{total}}{2 \times N_i}$

Apply class_weight dictionary during model.fit().

4.3 Augmentation Policy

To be applied only during training:

Geometric: Random Rotation ($0-180^{\circ}$), Horizontal/Vertical Flip.

Photometric: Random Contrast (0.2), Random Brightness (0.2). Simulates poor lighting conditions.

5. Development Roadmap & Directory Structure

5.1 Repository Structure

/
├── data/
│   ├── raw/               # Original downloads
│   ├── processed/         # Normalized & Cropped images
│   └── reference_img.jpg  # C-NMC reference for Macenko
├── src/
│   ├── training/
│   │   ├── train.py       # Main training loop
│   │   ├── models.py      # Architecture definitions
│   │   └── generator.py   # Custom DataGenerator (w/ Macenko & Patient Split)
│   ├── deployment/
│   │   ├── convert.py     # TFLite Quantization script
│   │   └── inference.py   # Raspberry Pi runtime script
│   └── utils/
│       ├── preprocessing.py # Macenko (with .npz load) & Padding logic
│       └── visualization.py # Confusion matrix & GradCAM
├── models/
│   ├── production/        # Final .tflite files
│   └── checkpoints/       # Intermediate .h5/.keras files
└── requirements.txt


5.2 Implementation Phases

Phase 1: Data Pipeline (Days 1-2)

Task 1.1: Write src/utils/preprocessing.py. Implement Macenko normalization.

Subtask: Create generate_reference.py to produce stain_vectors.npz from reference_img.jpg.

Task 1.2: Create a script to bulk-process ALL-IDB images into the C-NMC format structure.

Verification: Visual inspection of normalized ALL-IDB images against C-NMC samples. Colors must look identical.

Phase 2: Training (Days 3-4)

Task 2.1: Implement models.py with switchable backbones.

Task 2.2: Train MobileNetV2 on C-NMC (monitor val_recall).

Task 2.3: Validate on ALL-IDB (processed).

Target Metric: Validation Recall > 0.90.

Phase 3: Quantization & Edge Port (Days 5-6)

Task 3.1: Convert to TFLite using Dynamic Range Quantization.

converter.optimizations = [tf.lite.Optimize.DEFAULT]

Task 3.2: Deploy to Raspberry Pi 5.

Task 3.3: Benchmark inference speed (ms/image) and CPU temperature.

6. Testing & Quality Assurance

6.1 Statistical Validation

Metrics Reported: Recall, Precision, AUC, F1-Score.

False Negative Rate (FNR): Critical metric. Formula: $FN / (FN + TP)$. Must be $< 0.05$.

Threshold Tuning: Default sigmoid threshold is 0.5. If Recall is low, lower threshold to 0.3-0.4.

6.2 Edge Hardware Validation

Thermal Stress Test: Run inference loop for 15 minutes on Pi 5. Ensure no thermal throttling (CPU > 80°C).

Memory Leak Test: Ensure RAM usage remains stable over 1000 inferences.

6.3 "Sanity" Checks

The Rotation Test: Inference on an image rotated $90^{\circ}$ must yield the same class probability ($\pm 5\%$).

The Blank Test: A black image should predict class 0 (Normal) or low confidence.

The Stain Perturbation Test: Artificially shift H&E intensity by $\pm 10\%$. Model prediction should remain stable (variance < 10%).

7. Action Plan for Agent/Team

Setup: Clone repo and install dependencies (tensorflow, opencv-python, scikit-learn).

Data: Download C-NMC dataset to data/raw.

Code: Run Phase 1 tasks (Preprocessing logic & Reference Vector generation).

Train: Execute src/training/train.py --model mobilenetv2.

Export: Run src/deployment/convert.py.

Deploy: Copy .tflite and inference.py to Pi.