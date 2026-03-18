# Fedora Training Setup

Training environment for the current ALL Leukemia Edge Classifier workflow on Fedora.

## First-time setup
```bash
# From project root
bash linux_training/install_deps.sh
```

## Verify environment
```bash
source venv/bin/activate
python3 linux_training/verify_env.py
```

## Prepare dataset splits on Fedora

The tracked `cv_splits/cv_splits_3fold.json` was generated on Windows and may
contain Windows absolute paths. After copying `C-NMC_Dataset/` into the repo
root, rebuild the split file on Fedora:

```bash
source venv/bin/activate
python3 training_scripts/build_cv_splits.py
```

## Run training with the current trainer

Fold 1:
```bash
source venv/bin/activate
python3 training_scripts/train.py \
  --model mnv3l --fold 1 --run_name fedora_mnv3l_f1 \
  --epochs 150 --patience 25 \
  --batch_size 48 --num_workers 4 --no_live
```

Fold 2:
```bash
python3 training_scripts/train.py \
  --model mnv3l --fold 2 --run_name fedora_mnv3l_f2 \
  --epochs 150 --patience 25 \
  --batch_size 48 --num_workers 4 --no_live
```

Fold 3:
```bash
python3 training_scripts/train.py \
  --model mnv3l --fold 3 --run_name fedora_mnv3l_f3 \
  --epochs 150 --patience 25 \
  --batch_size 48 --num_workers 4 --no_live
```

## Resume a run
```bash
python3 training_scripts/train.py \
  --model mnv3l --fold 1 --run_name fedora_mnv3l_f1 \
  --epochs 150 --patience 25 \
  --batch_size 48 --num_workers 4 --no_live \
  --resume outputs/fedora_mnv3l_f1/fedora_mnv3l_f1_fold1_YYYYMMDD_HHMMSS_best.pth
```

## Notes

- Outputs are saved to `outputs/<run_name>/`
- Best checkpoint: `*_best.pth`
- Metrics CSV: `*_metrics.csv`
- System log: `*_system.log`
- `train.py` uses `nvidia-smi` for monitoring when available, but it still runs without it.
- The install script prefers `python3.11`, then `python3.10`, then `python3`.
- CUDA PyTorch wheels target CUDA 12.1. If your Fedora NVIDIA stack needs a different wheel, adjust the index URL in `install_deps.sh`.
- If you want the desktop demo UI on Fedora, make sure `python3-tkinter` is installed.
- If you want ONNX -> TFLite export, install the extra export dependency manually:
  `pip install onnx-tf`
