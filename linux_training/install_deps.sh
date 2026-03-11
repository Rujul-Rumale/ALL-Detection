#!/bin/bash
set -e

echo "=========================================="
echo "  ALL Leukemia Classifier — Linux Setup"
echo "=========================================="

# Navigate to project root (one level up from linux_training/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install with: sudo dnf install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# ── Virtualenv setup ──────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtualenv at venv/..."
    python3 -m venv venv
else
    echo "Virtualenv already exists at venv/ — skipping creation"
fi

source venv/bin/activate
echo "Virtualenv activated: $(which python)"

# ── Pip upgrade ───────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── System packages (Fedora) ──────────────────────────────────────────────────
echo ""
echo "Checking system dependencies..."
# libGL is required by OpenCV on Linux
if ! rpm -q mesa-libGL &> /dev/null; then
    echo "Installing mesa-libGL (required by OpenCV)..."
    sudo dnf install -y mesa-libGL mesa-libGL-devel
else
    echo "mesa-libGL already installed"
fi

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── Core ML packages ──────────────────────────────────────────────────────────
echo ""
echo "Installing core ML packages..."
pip install \
    timm>=0.9.0 \
    albumentations>=1.3.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    Pillow>=10.0.0 \
    --quiet

# ── Monitoring & display ──────────────────────────────────────────────────────
echo ""
echo "Installing monitoring packages..."
pip install \
    psutil>=5.9.0 \
    rich>=13.0.0 \
    --quiet

# ── Verify CUDA is accessible ─────────────────────────────────────────────────
echo ""
echo "Verifying CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available — training will run on CPU only')
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
echo "  To start training fold 1:"
echo "    python training_scripts/train_original_cpu_baseline.py \\"
echo "      --model mnv3l --fold 1 --run_name mnv3l_v3 \\"
echo "      --epochs 150 --patience 25 \\"
echo "      --batch_size 32 --num_workers 4 --no_live"
echo "=========================================="
