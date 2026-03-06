"""
Train MobileNetV3-Small on C-NMC Dataset for ALL vs HEM Classification.
Uses fold_0 + fold_1 for training, fold_2 for validation.
Exports final model to TFLite for Raspberry Pi 5 deployment.

V3: 3-phase training with full layer unfreezing.
  Phase 1: Frozen base, train head only (5 epochs)
  Phase 2: Unfreeze top 30 layers (10 epochs)
  Phase 3: Unfreeze ALL layers with very low LR (15 epochs)
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import shutil
import numpy as np

# ============ CONFIG ============
CNMC_DIR = r"c:\Open Source\leukiemea\C-NMC"
OUTPUT_DIR = r"c:\Open Source\leukiemea\models"
STAGING_DIR = r"c:\Open Source\leukiemea\cnmc_staging"
IMG_SIZE = 224  # MobileNetV3 default
BATCH_SIZE = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15   # cumulative epoch number
PHASE3_EPOCHS = 30   # cumulative epoch number
LEARNING_RATE = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ STAGE DATA ============
def stage_data():
    """Create train/val split from C-NMC folds."""
    train_dir = os.path.join(STAGING_DIR, "train")
    val_dir = os.path.join(STAGING_DIR, "val")

    # Only re-stage if not already done
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_count = sum(len(os.listdir(os.path.join(train_dir, c))) for c in ["all", "hem"])
        if train_count > 7000:
            print("  Staging directory already exists, skipping copy.")
            return train_dir, val_dir

    # Clean previous staging
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    for split_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(split_dir, "all"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "hem"), exist_ok=True)

    # fold_0 + fold_1 → train, fold_2 → val
    train_folds = ["fold_0", "fold_1"]
    val_folds = ["fold_2"]

    copied = {"train_all": 0, "train_hem": 0, "val_all": 0, "val_hem": 0}

    for fold in train_folds:
        for cls in ["all", "hem"]:
            src = os.path.join(CNMC_DIR, fold, fold, cls)
            dst = os.path.join(train_dir, cls)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                copied[f"train_{cls}"] += 1

    for fold in val_folds:
        for cls in ["all", "hem"]:
            src = os.path.join(CNMC_DIR, fold, fold, cls)
            dst = os.path.join(val_dir, cls)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                copied[f"val_{cls}"] += 1

    print(f"  Train: {copied['train_all']} ALL + {copied['train_hem']} HEM = {copied['train_all']+copied['train_hem']}")
    print(f"  Val:   {copied['val_all']} ALL + {copied['val_hem']} HEM = {copied['val_all']+copied['val_hem']}")
    return train_dir, val_dir


# ============ MODEL ============
def build_model():
    """MobileNetV3-Small with custom classification head."""
    base = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,  # Handles normalization from [0,255] internally
    )
    # Freeze base for initial training
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary: ALL vs HEM

    model = Model(inputs, outputs)
    return model, base


# ============ TRAINING ============
def train():
    print("=" * 60)
    print("  MobileNetV3-Small — C-NMC Training Pipeline (V2)")
    print("=" * 60)

    # Stage data
    print("\n[1/4] Staging C-NMC dataset...")
    train_dir, val_dir = stage_data()

    # Data generators
    # KEY FIX: NO rescale=1./255 — MobileNetV3 include_preprocessing=True
    # expects raw [0, 255] pixel values and normalizes internally.
    print("\n[2/4] Setting up data generators...")
    train_datagen = ImageDataGenerator(
        # NO rescale! MobileNetV3 handles normalization internally.
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        zoom_range=0.1,
        fill_mode="reflect",
    )
    val_datagen = ImageDataGenerator()  # No rescale, no augmentation

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["hem", "all"],  # 0=healthy, 1=blast
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["hem", "all"],  # 0=healthy, 1=blast
        shuffle=False,
    )

    print(f"  Classes: {train_gen.class_indices}")

    # Sanity check: verify pixel range
    sample_batch, sample_labels = next(train_gen)
    print(f"  Pixel range check: min={sample_batch.min():.1f}, max={sample_batch.max():.1f} (should be 0-255)")
    print(f"  Label distribution in batch: {np.bincount(sample_labels.astype(int))}")
    train_gen.reset()

    # Build model
    print("\n[3/4] Building MobileNetV3-Small...")
    model, base = build_model()

    # Milder class weights (push accuracy higher while keeping decent recall)
    n_all = train_gen.classes.sum()
    n_hem = len(train_gen.classes) - n_all
    total = n_all + n_hem
    # Sqrt-balanced weights: less aggressive than full inverse-frequency
    class_weight = {0: (total / (2 * n_hem)) ** 0.5, 1: (total / (2 * n_all)) ** 0.5}
    print(f"  Class weights: hem(0)={class_weight[0]:.2f}, all(1)={class_weight[1]:.2f}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )

    print(f"  Total params: {model.count_params():,}")
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable params: {trainable:,}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc_best.h5"),
            monitor="val_recall", mode="max", save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor="val_recall", mode="max", patience=8, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
    ]

    # Phase 1: Train head only (base frozen) — 5 epochs
    print("\n[4/4] Training Phase 1: Head only (base frozen)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    val_acc_1 = history1.history.get("val_accuracy", [0])[-1]
    print(f"\n  Phase 1 final val_accuracy: {val_acc_1:.4f}")
    if val_acc_1 < 0.55:
        print("  WARNING: Accuracy still near random. Check data pipeline!")

    # Phase 2: Unfreeze top 30 layers — 10 more epochs
    print("\nPhase 2: Fine-tuning top 30 layers of MobileNetV3...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        initial_epoch=PHASE1_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Phase 3: UNFREEZE ALL LAYERS — 15 more epochs with very low LR
    print("\nPhase 3: Full fine-tuning — ALL layers unfrozen (LR=1e-5)...")
    base.trainable = True  # Everything is now trainable

    trainable_p3 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  All {trainable_p3:,} params are now trainable.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE3_EPOCHS,
        initial_epoch=PHASE2_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # ============ EXPORT TO TFLITE ============
    print("\nExporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "mobilenetv3_cnmc.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")

    # Final evaluation
    print("\nFinal Evaluation on Validation Set:")
    results = model.evaluate(val_gen)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    print("\nDone! Model ready for Watershed -> Crop -> MobileNetV3 TFLite pipeline.")


if __name__ == "__main__":
    train()
