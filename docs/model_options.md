# ALL-IDB Model Options

## Option 1: Peter Moss ALL-Arduino-Nano-33-BLE-Sense-Classifier ✓ VERIFIED

**Status:** Works 100% on raw ALL-IDB images

### Source
- Repo: `https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier`
- Local: `ALL-Arduino-Nano-33-BLE-Sense-Classifier/`

### What It Does
Classifies **full microscope images** (not segmented cells) as leukemia or healthy.

### Model Files
| File | Purpose |
|------|---------|
| `model/all_nano_33_ble_sense.h5` | Keras weights |
| `model/all_nano_33_ble_sense.tflite` | TFLite for edge |
| `model/all_nano_33_ble_sense.json` | Architecture |

### How to Use
```python
import cv2
import tensorflow as tf

# Load model
with open("model/all_nano_33_ble_sense.json") as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights("model/all_nano_33_ble_sense.h5")

# Preprocess: resize to 100x100, normalize
img = cv2.imread(image_path)
img = cv2.resize(img, (100, 100))
img = img.astype('float32') / 255.0

# Predict
pred = model.predict(img[None, ...])
# pred[0][0] > pred[0][1] → ALL (leukemia)
# pred[0][1] > pred[0][0] → HEM (healthy)
```

### Key Points
- Input: 100x100 RGB images
- Output: 2 classes (ALL/HEM)
- Labels: `_0` in filename = ALL, `_1` = HEM
- **No cell segmentation needed**

### Performance
- Reported: 93.3% accuracy, 96.7% AUC
- Our test: **100%** on L1/L2 raw images

---

## Option 2: all-classifiers-2019 (Multiple Models)

**Status:** Not tested yet — has more model options

### Source
- Repo: `https://github.com/AMLResearchProject/all-classifiers-2019`
- Local: `all-classifiers-2019/`

### Available Models
| Path | Dataset | Format |
|------|---------|--------|
| `Projects/Keras/AllCNN/Paper_1/ALL_IDB1/Non_Augmented/Model/weights.h5` | ALL-IDB1 | Keras |
| `Projects/Keras/AllCNN/Paper_1/ALL_IDB2/Non_Augmented/Model/weights.h5` | ALL-IDB2 | Keras |
| `Projects/Keras/QuantisedCode/weights.tflite` | - | TFLite |
| `Projects/Keras/QuantisedCode/quant_weights.tflite` | - | TFLite Quantized |
| `Projects/NCS1/Model/ALLGraph.pb` | - | TensorFlow PB |

### Key Points
- Multiple datasets: ALL-IDB1 (full images) and ALL-IDB2 (cropped cells)
- Has Jupyter notebooks for training/testing
- Reference paper: "Acute Leukemia Classification Using CNN"
- Quantized TFLite available for edge

---

## Option 3: all-detection-system-2019 (TBD)

**Status:** Not cloned yet

### Source
- Repo: `https://github.com/AMLResearchProject/all-detection-system-2019`

---

## Comparison Summary

| Option | Input Type | Edge Ready | Tested |
|--------|------------|------------|--------|
| Option 1 (Arduino) | Full image 100x100 | ✓ TFLite | ✓ 100% |
| Option 2 (classifiers) | Full/Cell varies | ✓ TFLite | - |
| Option 3 (detection) | TBD | TBD | - |

## Recommendation

For **Edge-AI with microscope**:
- **Option 1** is simplest — just resize and classify full images
- No complex segmentation pipeline needed
