import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

VAL_DIR = r"c:\Open Source\leukiemea\cnmc_staging\val"
IMG_SIZE = 224
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    {
        "name": "MobileNetV3-Small (Weighted)",
        "path": r"c:\Open Source\leukiemea\models\mobilenetv3_cnmc_best.pth",
        "arch": "mobilenet_v3_small"
    },
    {
        "name": "MobileNetV3-Small (Unweighted)",
        "path": r"c:\Open Source\leukiemea\models\mobilenetv3_cnmc_unweighted.pth",
        "arch": "mobilenet_v3_small"
    },
    {
        "name": "MobileNetV3-Large (Weighted)",
        "path": r"c:\Open Source\leukiemea\models\mobilenetv3_large_cnmc_best.pth",
        "arch": "mobilenet_v3_large"
    }
]

def load_model(arch, path):
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_large(weights=None)
        
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint['class_to_idx']

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(results_df, save_path):
    metrics = ['Accuracy', 'Sensitivity (ALL)', 'Specificity (HEM)']
    
    # Clean string percentages to floats for plotting
    plot_df = results_df.copy()
    for col in metrics:
        plot_df[col] = plot_df[col].str.rstrip('%').astype(float)
        
    x = np.arange(len(plot_df['Model Name']))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    acc_bars = ax.bar(x - width, plot_df['Accuracy'], width, label='Accuracy', color='#4c72b0')
    sens_bars = ax.bar(x, plot_df['Sensitivity (ALL)'], width, label='Sensitivity (ALL)', color='#55a868')
    spec_bars = ax.bar(x + width, plot_df['Specificity (HEM)'], width, label='Specificity (HEM)', color='#c44e52')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Model Name'], rotation=0, ha='center')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 115)

    # Label bars
    for bars in [acc_bars, sens_bars, spec_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_models():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    img_paths = [sample[0] for sample in val_dataset.samples]

    summary_results = []
    
    os.makedirs(r"c:\Open Source\leukiemea\evaluation_outputs", exist_ok=True)

    for m in MODELS:
        print(f"\nEvaluating {m['name']}...")
        model, class_to_idx = load_model(m['arch'], m['path'])
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        all_preds = []
        all_labels = []
        all_confidences = []
        
        t0 = time.time()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confs.cpu().numpy())
        t1 = time.time()
        
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        
        all_idx = class_to_idx['all']
        hem_idx = class_to_idx['hem']
        tn, fp, fn, tp = cm.ravel() if all_idx == 1 else cm[::-1, ::-1].ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        total_params = sum(p.numel() for p in model.parameters())
        
        summary_results.append({
            "Model Name": m["name"],
            "Accuracy": f"{acc*100:.2f}%",
            "Sensitivity (ALL)": f"{sensitivity*100:.2f}%",
            "Specificity (HEM)": f"{specificity*100:.2f}%",
            "Total Params": total_params,
            "Inference Time Eval Phase (s)": round(t1 - t0, 2)
        })
        
        safe_name = m['name'].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        
        # Plot and save CM
        # Re-arrange confusion matrix so ALL is always the first class conceptually
        if all_idx == 1:
            cm = cm[::-1, ::-1]
        
        cm_path = f"c:\\Open Source\\leukiemea\\evaluation_outputs\\cm_{safe_name}.png"
        plot_confusion_matrix(cm, ['ALL', 'HEM'], f"{m['name']} Confusion Matrix", cm_path)
        
        # Save individual predictions
        predictions_data = []
        for i in range(len(all_labels)):
            true_lbl_name = idx_to_class[all_labels[i]]
            pred_lbl_name = idx_to_class[all_preds[i]]
            predictions_data.append({
                "File Path": img_paths[i],
                "True Class": true_lbl_name,
                "Predicted Class": pred_lbl_name,
                "Correct": true_lbl_name == pred_lbl_name,
                "Confidence": f"{all_confidences[i]:.4f}"
            })
            
        df = pd.DataFrame(predictions_data)
        csv_path = f"c:\\Open Source\\leukiemea\\evaluation_outputs\\predictions_{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed predictions to {csv_path}")
        print(f"Saved confusion matrix to {cm_path}")

    # Process overall summary
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = r"c:\Open Source\leukiemea\evaluation_outputs\summary_all_models.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Plot comparisons
    comp_path = r"c:\Open Source\leukiemea\evaluation_outputs\model_comparison.png"
    plot_model_comparison(summary_df, comp_path)
    
    print(f"\nSaved summary CSV to {summary_csv_path}")
    print(f"Saved comparison graph to {comp_path}")
    print("\nFINAL RESULTS:")
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    evaluate_models()
