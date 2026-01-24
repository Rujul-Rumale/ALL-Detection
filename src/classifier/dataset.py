"""
C-NMC Dataset Loader
"""

import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import Sequence
import tensorflow as tf


class CNMCDataset:
    # loads C-NMC dataset for ALL classification
    CLASS_NAMES = ['hem', 'all']  # healthy vs ALL
    
    def __init__(self, root_dir, fold='fold_0'):
        self.root_dir = Path(root_dir)
        self.fold = fold
        self.fold_path = self.root_dir / fold / fold
        
        if not self.fold_path.exists():
            raise ValueError(f"Can't find fold at: {self.fold_path}")
        
        self.image_paths, self.labels = self._load_dataset()
    
    def _load_dataset(self):
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = self.fold_path / class_name
            
            if not class_dir.exists():
                print(f"Warning: missing {class_name} folder")
                continue
            
            # C-NMC uses BMP format
            for img_path in class_dir.glob('*.bmp'):
                image_paths.append(img_path)
                labels.append(class_idx)
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, label
    
    def get_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.CLASS_NAMES[i]: count for i, count in zip(unique, counts)}


class CNMCDataGenerator(Sequence):
    # Keras data generator with augmentation
    
    def __init__(self, dataset, batch_size=32, img_size=(128, 128),
                 augment=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        
        batch_images = []
        batch_labels = []
        
        for i in batch_indexes:
            image, label = self.dataset[i]
            
            image = cv2.resize(image, self.img_size)
            
            if self.augment:
                image = self._augment(image)
            
            image = image.astype(np.float32) / 255.0
            
            batch_images.append(image)
            batch_labels.append(label)
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _augment(self, image):
        # random flips and rotations
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # brightness adjustment
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image


def create_train_val_test_split(root_dir, train_fold='fold_0',
                                  val_fold='fold_1', test_fold='fold_2'):
    # create datasets from the 3 folds
    train_ds = CNMCDataset(root_dir, fold=train_fold)
    val_ds = CNMCDataset(root_dir, fold=val_fold)
    test_ds = CNMCDataset(root_dir, fold=test_fold)
    
    print("Dataset loaded:")
    print(f"  Train: {len(train_ds)} images - {train_ds.get_class_distribution()}")
    print(f"  Val:   {len(val_ds)} images - {val_ds.get_class_distribution()}")
    print(f"  Test:  {len(test_ds)} images - {test_ds.get_class_distribution()}")
    
    return train_ds, val_ds, test_ds
