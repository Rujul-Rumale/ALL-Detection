import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_scripts.train import get_model, get_loaders
from src.utils.training_metrics import sweep_all_positive_thresholds

def evaluate_fold(model_type, fold, run_dir, res=320, batch_size=32):
    print(f"\n--- Evaluating {model_type.upper()} Fold {fold} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import types
    args = types.SimpleNamespace(
        model = model_type,
        fold = fold,
        splits_json = "cv_splits/cv_splits_3fold.json",
        res = res,
        batch_size = batch_size,
        num_workers = 0
    )
    
    # Needs cd to main dir for splits_json
    os.chdir(r"C:\Open Source\leukiemea")

    _, val_loader, _, _, _ = get_loaders(args)
    model, _, _, _, _ = get_model(args)
    
    pth_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
    if not pth_files:
        print(f"No .pth file found in {run_dir}")
        return None
    
    ckpt_path = os.path.join(run_dir, pth_files[0])
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    all_targets = []
    all_scores_all = []

    print("Starting inference...", flush=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(val_loader)}", flush=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            variants = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]

            avg_probs = torch.zeros(images.size(0), 2, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                for v in variants:
                    logits = model(v)
                    avg_probs += torch.softmax(logits.float(), dim=1)

            avg_probs /= len(variants)
            
            all_targets.extend(labels.cpu().numpy())
            # Probability of ALL is at index 0
            all_scores_all.extend(avg_probs[:, 0].cpu().numpy())

    all_targets = np.array(all_targets)
    all_scores_all = np.array(all_scores_all)

    best_metrics = sweep_all_positive_thresholds(all_targets, all_scores_all)
    
    print(f"Opt Thresh: {best_metrics['threshold']:.2f}")
    print(f"AUC:  {best_metrics['auc']:.4f}")
    print(f"Sens: {best_metrics['sens']:.4f} (ALL accurately detected)")
    print(f"Spec: {best_metrics['spec']:.4f} (HEM accurately detected)")
    print(f"Acc:  {best_metrics['acc']:.4f}")
    print(f"F1:   {best_metrics['f1']:.4f}")
    print(f"TP(ALL): {best_metrics['tp']}, FN(Missed ALL): {best_metrics['fn']}, TN(HEM): {best_metrics['tn']}, FP(False ALL): {best_metrics['fp']}")
    
    return best_metrics

if __name__ == "__main__":
    runs = [
        ("mnv3l", 1, r"C:\Open Source\leukiemea\outputs\local_mnv3l_final1"),
        ("mnv3l", 2, r"C:\Open Source\leukiemea\outputs\local_mnv3l_final2"),
        ("mnv3l", 3, r"C:\Open Source\leukiemea\outputs\local_mnv3l_final3"),
        ("effb0", 1, r"C:\Open Source\leukiemea\outputs\local_effb0_final1"),
        ("effb0", 2, r"C:\Open Source\leukiemea\outputs\local_effb0_final2"),
        ("effb0", 3, r"C:\Open Source\leukiemea\outputs\local_effb0_final3"),
    ]
    
    results = {'mnv3l': [], 'effb0': []}

    for model_type, fold, run_dir in runs:
        res = evaluate_fold(model_type, fold, run_dir)
        if res:
            results[model_type].append(res)
            
    print("\n================== FINAL SUMMARIES ==================")
    for model_type in ['mnv3l', 'effb0']:
        if len(results[model_type]) == 3:
            auc = np.mean([r['auc'] for r in results[model_type]])
            sens = np.mean([r['sens'] for r in results[model_type]])
            spec = np.mean([r['spec'] for r in results[model_type]])
            acc = np.mean([r['acc'] for r in results[model_type]])
            f1 = np.mean([r['f1'] for r in results[model_type]])
            
            auc_sd = np.std([r['auc'] for r in results[model_type]])
            sens_sd = np.std([r['sens'] for r in results[model_type]])
            spec_sd = np.std([r['spec'] for r in results[model_type]])
            acc_sd = np.std([r['acc'] for r in results[model_type]])
            f1_sd = np.std([r['f1'] for r in results[model_type]])
            
            print(f"--- {model_type.upper()} (ALL as Positive) ---")
            print(f"AUC:  {auc:.4f} ± {auc_sd:.4f}")
            print(f"Sens: {sens:.4f} ± {sens_sd:.4f}")
            print(f"Spec: {spec:.4f} ± {spec_sd:.4f}")
            print(f"Acc:  {acc:.4f} ± {acc_sd:.4f}")
            print(f"F1:   {f1:.4f} ± {f1_sd:.4f}")
            print("--------------------------------------------------")
