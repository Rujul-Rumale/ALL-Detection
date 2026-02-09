
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.manifold import TSNE
import random

def run_tsne_analysis(sample_size=200):
    weights_path = 'imagenet' # Start with standard features
    
    # 1. Load Model (Features only, no top)
    model = MobileNetV2(weights=weights_path, include_top=False, input_shape=(224, 224, 3), pooling='avg')
    print("MobileNetV2 loaded for feature extraction.")

    # 2. Prepare Data
    base_dir = r"C-NMC_Dataset\fold_0\fold_0"
    all_files = glob.glob(os.path.join(base_dir, "all", "*.bmp"))
    hem_files = glob.glob(os.path.join(base_dir, "hem", "*.bmp"))
    
    # Stratified sample
    n = sample_size // 2
    files = random.sample(all_files, n) + random.sample(hem_files, n)
    labels = ([1] * n) + ([0] * n) # 1=ALL, 0=HEM
    
    print(f"Extracting features from {len(files)} images...")
    
    features = []
    valid_labels = []
    
    for i, img_path in enumerate(files):
        try:
            # Load and Preprocess
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract Feature
            feat = model.predict(x, verbose=0)
            features.append(feat.flatten())
            valid_labels.append(labels[i])
            
            if i % 20 == 0:
                print(f"Processed {i}/{len(files)}")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    features = np.array(features)
    valid_labels = np.array(valid_labels)

    # 3. Run t-SNE
    print("Running t-SNE... (this captures the 'hidden' similarity)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)

    # 4. Plot
    plt.figure(figsize=(10, 8))
    
    # Plot HEM (Healthy)
    mask_hem = (valid_labels == 0)
    plt.scatter(tsne_results[mask_hem, 0], tsne_results[mask_hem, 1], 
                c='green', label='Healthy (HEM)', alpha=0.6)
    
    # Plot ALL (Cancer)
    mask_all = (valid_labels == 1)
    plt.scatter(tsne_results[mask_all, 0], tsne_results[mask_all, 1], 
                c='red', label='Leukemia (ALL)', alpha=0.6)
    
    plt.title("t-SNE Projection of C-NMC Dataset (MobileNetV2 Features)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "dataset_tsne_check.png"
    plt.savefig(output_path)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    run_tsne_analysis()
