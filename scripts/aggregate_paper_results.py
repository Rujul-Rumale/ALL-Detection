import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths to the metrics files for the 6 final folds
METRICS_FILES = {
    "EfficientNet-B0": [
        r"c:\Open Source\leukiemea\outputs\local_effb0_final1\local_effb0_final1_fold1_20260316_220343_metrics.csv",
        r"c:\Open Source\leukiemea\outputs\local_effb0_final2\local_effb0_final2_fold2_20260317_010239_metrics.csv",
        r"c:\Open Source\leukiemea\outputs\local_effb0_final3\local_effb0_final3_fold3_20260317_035939_metrics.csv",
    ],
    "MobileNetV3-Large": [
        r"c:\Open Source\leukiemea\outputs\local_mnv3l_final1\local_mnv3l_final1_fold1_20260315_152609_metrics.csv",
        r"c:\Open Source\leukiemea\outputs\local_mnv3l_final2\local_mnv3l_final2_fold2_20260315_173338_metrics.csv",
        r"c:\Open Source\leukiemea\outputs\local_mnv3l_final3\local_mnv3l_final3_fold3_20260315_194014_metrics.csv",
    ]
}

def aggregate_metrics(name, files):
    fold_results = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found.")
            continue
        df = pd.read_csv(f)
        # Find the best epoch based on AUC
        best_row = df.loc[df['auc'].idxmax()]
        fold_results.append({
            'auc': best_row['auc'],
            'acc': best_row['val_acc'],
            'f1': best_row['f1'],
            'sens': best_row['sensitivity'],
            'spec': best_row['specificity']
        })
    
    if not fold_results:
        return None
    
    df_results = pd.DataFrame(fold_results)
    means = df_results.mean()
    stds = df_results.std()
    
    return means, stds

def main():
    print("ALL Leukemia Cell Detection - 3-Fold Cross-Validation Results")
    print("=" * 70)
    
    final_table = []
    
    for model_name, files in METRICS_FILES.items():
        res = aggregate_metrics(model_name, files)
        if res:
            means, stds = res
            print(f"\n{model_name}:")
            for metric in ['acc', 'auc', 'f1', 'sens', 'spec']:
                print(f"  {metric.upper():<5}: {means[metric]:.4f} ± {stds[metric]:.4f}")
            
            final_table.append({
                'Model': model_name,
                'Accuracy': f"{means['acc']:.4f} ± {stds['acc']:.4f}",
                'AUC': f"{means['auc']:.4f} ± {stds['auc']:.4f}",
                'F1 Score': f"{means['f1']:.4f} ± {stds['f1']:.4f}",
                'Sensitivity': f"{means['sens']:.4f} ± {stds['sens']:.4f}",
                'Specificity': f"{means['spec']:.4f} ± {stds['spec']:.4f}"
            })

    # Save to CSV for the paper
    summary_df = pd.DataFrame(final_table)
    summary_df.to_csv("paper_results_metrics_summary.csv", index=False)
    print("\nSummary table saved to paper_results_metrics_summary.csv")

    # Generate LaTeX code
    latex_code = summary_df.to_latex(index=False, caption="Classification performance (3-fold CV)", label="tab:results")
    with open("paper_results_table_latex.txt", "w") as f:
        f.write(latex_code)
    print("LaTeX table code saved to paper_results_table_latex.txt")

if __name__ == "__main__":
    main()
