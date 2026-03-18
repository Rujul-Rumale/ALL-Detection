#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  ALL Leukemia Classifier - Fedora Setup"
echo "=========================================="

# Navigate to project root (one level up from linux_training/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Prefer Python 3.11/3.10 for smoother package compatibility on Linux.
PYTHON_BIN=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No usable Python 3 interpreter found."
    echo "Install one with: sudo dnf install python3.11 python3.11-devel"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python interpreter: $PYTHON_BIN"
echo "Python version: $PYTHON_VERSION"

echo ""
echo "Installing Fedora system packages..."
sudo dnf install -y \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    python3-tkinter \
    mesa-libGL \
    mesa-libGL-devel \
    git

if [ ! -d "venv" ]; then
    echo "Creating virtualenv at venv/..."
    "$PYTHON_BIN" -m venv venv
else
    echo "Virtualenv already exists at venv/ - skipping creation"
fi

source venv/bin/activate
echo "Virtualenv activated: $(which python)"

pip install --upgrade pip setuptools wheel --quiet

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "NVIDIA GPU detected - installing CUDA 12.1 PyTorch wheels..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo ""
    echo "No NVIDIA GPU detected - installing CPU PyTorch wheels..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo ""
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "Verifying torch runtime..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available - training will run on CPU only')
"

echo ""
echo "=========================================="
echo "  Setup complete."
echo ""
echo "  To activate the environment in future:"
echo "    source venv/bin/activate"
echo ""
echo "  To verify the full environment:"
echo "    python linux_training/verify_env.py"
echo ""
echo "  After copying C-NMC_Dataset/ into the repo, regenerate Linux-safe splits:"
echo "    python training_scripts/build_cv_splits.py"
echo ""
echo "  To start training fold 1 with the current trainer:"
echo "    python training_scripts/train.py \\"
echo "      --model mnv3l --fold 1 --run_name fedora_mnv3l_f1 \\"
echo "      --epochs 150 --patience 25 \\"
echo "      --batch_size 48 --num_workers 4 --no_live"
echo ""
echo "  Optional export dependency (only if you need TFLite export):"
echo "    pip install onnx-tf"
echo "=========================================="
