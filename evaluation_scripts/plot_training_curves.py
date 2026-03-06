import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_PATH = r"c:\Open Source\leukiemea\models\training_log_pytorch.csv"
OUTPUT_DIR = r"c:\Open Source\leukiemea\evaluation_outputs"

if not os.path.exists(LOG_PATH):
    print("Log not found.")
    exit(1)

df = pd.read_csv(LOG_PATH)
print("DataFrame Shape:", df.shape)

epochs = df['epoch'].values
train_acc = df['train_acc'].values * 100
val_acc = df['val_acc'].values * 100
train_loss = df['train_loss'].values
val_loss = df['val_loss'].values
val_rec = df['val_recall'].values * 100

plt.figure(figsize=(12, 10))

# Plot Accuracy & Recall
plt.subplot(2, 1, 1)
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy')
plt.plot(epochs, val_rec, 'r--', label='Validation Sensitivity (Recall)')
plt.axvline(x=5, color='gray', linestyle=':', label='Phase 2 Start')
plt.axvline(x=15, color='orange', linestyle=':', label='Phase 3 Start')
plt.title('MobileNetV3-Large: Training Accuracy & Sensitivity over 50 Epochs')
plt.ylabel('Percentage (%)')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)

# Plot Loss
plt.subplot(2, 1, 2)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'g-', label='Validation Loss')
plt.axvline(x=5, color='gray', linestyle=':', label='Phase 2 Start')
plt.axvline(x=15, color='orange', linestyle=':', label='Phase 3 Start')
plt.title('MobileNetV3-Large: Training & Validation Loss')
plt.ylabel('Loss (Cross-Entropy)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved metric curves to: {plot_path}")
