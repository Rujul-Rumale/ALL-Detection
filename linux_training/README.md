# Linux Training Setup

Training environment for ALL Leukemia Edge Classifier on Fedora KDE.

## First-time setup
```bash
# From project root
bash linux_training/install_deps.sh
```

## Verify environment
```bash
source venv/bin/activate
python linux_training/verify_env.py
```

## Run training

Fold 1:
```bash
source venv/bin/activate
python training_scripts/train_original_cpu_baseline.py \
  --model mnv3l --fold 1 --run_name mnv3l_v3 \
  --epochs 150 --patience 25 \
  --batch_size 32 --num_workers 4 --no_live
```

Fold 2:
```bash
python training_scripts/train_original_cpu_baseline.py \
  --model mnv3l --fold 2 --run_name mnv3l_v3 \
  --epochs 150 --patience 25 \
  --batch_size 32 --num_workers 4 --no_live
```

Fold 3:
```bash
python training_scripts/train_original_cpu_baseline.py \
  --model mnv3l --fold 3 --run_name mnv3l_v3 \
  --epochs 150 --patience 25 \
  --batch_size 32 --num_workers 4 --no_live
```

## Resume a run
```bash
python training_scripts/train_original_cpu_baseline.py \
  --model mnv3l --fold 1 --run_name mnv3l_v3 \
  --epochs 150 --patience 25 \
  --batch_size 32 --num_workers 4 --no_live \
  --resume outputs/mnv3l_v3/mnv3l_v3_fold1_YYYYMMDD_HHMMSS_best.pth
```

## Notes

- Outputs saved to `outputs/<run_name>/`
- Best checkpoint: `*_best.pth`
- Metrics CSV: `*_metrics.csv`
- System log: `*_system.log`
- RAM exhaustion fix: always clear previous training processes before starting a new run
- CUDA PyTorch wheel targets CUDA 12.1 — if your driver supports a different version adjust the index URL in `install_deps.sh`
