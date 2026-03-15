

import sys
import os

# Setup GPU paths for Windows Native + TF 2.10
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add src root to path if needed for direct execution
try:
    from src.utils import gpu_setup
except ImportError:
    # If running from src root
    try:
        from utils import gpu_setup
    except ImportError:
        pass

import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Import our modules
from models import build_mobilenetv2
from generator import create_generators

def train(args):
    # 1. Setup Data Generators
    base_dir = args.data_dir
    if not os.path.exists(base_dir):
        print(f"Error: Data directory {base_dir} does not exist.")
        return

    train_gen, val_gen = create_generators(
        base_dir, 
        target_size=(224, 224), 
        batch_size=args.batch_size
    )

    # 2. Build Model
    print("Building MobileNetV2...")
    model = build_mobilenetv2(input_shape=(224, 224, 3))
    
    # 3. Setup Callbacks
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Save best model based on validation recall (or loss if recall is unstable)
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "mobilenetv2_best.keras"),
            monitor='val_recall', # Primary metric
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Stop if no improvement
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Log metrics to CSV
        CSVLogger("training_log.csv")
    ]

    # 4. Train
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # 5. Save Final Model
    final_model_path = "models/production/mobilenetv2_final.keras"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Leukemia Detection Model")
    parser.add_argument("--data_dir", type=str, default=r"C-NMC_Dataset\fold_0\fold_0", help="Path to dataset fold")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    train(args)
