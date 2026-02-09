
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates training and validation data generators for C-NMC dataset.
    
    Args:
        data_dir: Path to the dataset root (e.g., 'C-NMC_Dataset/fold_0/fold_0').
                  Expects subdirectories 'all' and 'hem'.
        target_size: Tuple (height, width) for resizing.
        batch_size: Batch size for training.
        
    Returns:
        train_generator, val_generator
    """
    
    # Define augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
        validation_split=0.2  # Use 20% of data for validation if not using separate folds
    )

    # For validation, only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print(f"Loading training data from {data_dir}...")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    print(f"Loading validation data from {data_dir}...")
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

if __name__ == "__main__":
    # Test the generator
    base_dir = r"C-NMC_Dataset\fold_0\fold_0"
    if os.path.exists(base_dir):
        train_gen, val_gen = create_generators(base_dir)
        print(f"Classes: {train_gen.class_indices}")
    else:
        print(f"Directory {base_dir} not found. Please check path.")
