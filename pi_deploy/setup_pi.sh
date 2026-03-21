#!/usr/bin/env bash
# =============================================================================
# setup_pi.sh
# ===========
# Bootstraps a fresh Raspberry Pi 5 (Raspberry Pi OS 64-bit) for leukemia
# classifier deployment. Idempotent — safe to run multiple times.
#
# Usage:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh
# =============================================================================

set -euo pipefail

VENV_DIR="$HOME/leukemia_venv"
LOG_FILE="$HOME/leukemia_setup.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== Leukemia Classifier Pi 5 Setup ==="
log "Raspberry Pi OS 64-bit required. Log: $LOG_FILE"

# =============================================================================
# STEP 1 — System packages
# =============================================================================
log "Step 1/5: Installing system packages ..."
sudo apt-get update -y >> "$LOG_FILE" 2>&1
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    git \
    wget >> "$LOG_FILE" 2>&1
log "  System packages installed."

# =============================================================================
# STEP 2 — Python virtual environment
# =============================================================================
log "Step 2/5: Creating virtual environment at $VENV_DIR ..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log "  Virtual environment created."
else
    log "  Virtual environment already exists — skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "  Virtual environment activated."

# Upgrade pip inside venv (idempotent)
pip install --upgrade pip --quiet >> "$LOG_FILE" 2>&1

# =============================================================================
# STEP 3 — Python dependencies
# =============================================================================
log "Step 3/5: Installing Python packages ..."

# numpy — pinned for ABI compatibility with tflite-runtime on Pi
pip install "numpy==1.24.4" --quiet >> "$LOG_FILE" 2>&1
log "  numpy==1.24.4 done."

# opencv headless — full opencv is not needed; headless avoids GTK deps
pip install "opencv-python-headless==4.8.1.78" --quiet >> "$LOG_FILE" 2>&1
log "  opencv-python-headless==4.8.1.78 done."

# tflite-runtime — piwheels builds for aarch64 Pi OS, faster than pip
# Falls back to pip if piwheels is not configured as an extra index.
pip install \
    --extra-index-url https://www.piwheels.org/simple \
    tflite-runtime --quiet >> "$LOG_FILE" 2>&1
log "  tflite-runtime done."

# Pillow — image loading for pre-processing
pip install "Pillow" --quiet >> "$LOG_FILE" 2>&1
log "  Pillow done."

# psutil — memory / CPU monitoring in benchmark script
pip install "psutil" --quiet >> "$LOG_FILE" 2>&1
log "  psutil done."

# PyTorch CPU-only ARM wheel (2.1.0 for Pi 5 aarch64)
# Official CPU wheel — GPU builds are not available for Pi 5.
pip install "torch==2.1.0" --index-url https://download.pytorch.org/whl/cpu \
    --quiet >> "$LOG_FILE" 2>&1
log "  torch==2.1.0 (CPU) done."

# Transformers — needed for facebook/sam-vit-base in demo_pipeline.py
pip install "transformers==4.38.0" --quiet >> "$LOG_FILE" 2>&1
log "  transformers==4.38.0 done."

# sentencepiece — required by many tokenizers inside transformers
pip install "sentencepiece" --quiet >> "$LOG_FILE" 2>&1
log "  sentencepiece done."

log "  All Python packages installed."

# =============================================================================
# STEP 4 — Clone / rsync project
# =============================================================================
log "Step 4/5: Project transfer ..."

# ──────────────────────────────────────────────────────────────────────────────
# TODO: Fill in one of the two options below before running this script.
# ──────────────────────────────────────────────────────────────────────────────
REPO_URL=""      # e.g. "https://github.com/YOURUSER/leukiemea.git"
PROJECT_DIR="$HOME/leukiemea"

if [ -n "$REPO_URL" ]; then
    if [ ! -d "$PROJECT_DIR/.git" ]; then
        log "  Cloning $REPO_URL → $PROJECT_DIR ..."
        git clone "$REPO_URL" "$PROJECT_DIR" >> "$LOG_FILE" 2>&1
        log "  Clone complete."
    else
        log "  Repository already present — pulling latest ..."
        git -C "$PROJECT_DIR" pull --ff-only >> "$LOG_FILE" 2>&1
        log "  Pull complete."
    fi
else
    log "  [SKIP] REPO_URL not set."
    log "  Manually rsync the project:"
    log "    rsync -avz --exclude='outputs/' --exclude='*.pth' \\"
    log "          --exclude='*.onnx' --exclude='__pycache__/' \\"
    log "          /path/to/leukiemea/ pi@<PI_IP>:~/leukiemea/"
fi

# =============================================================================
# STEP 5 — Done
# =============================================================================
log "Step 5/5: Setup complete."
echo ""
echo "=================================================="
echo "  Setup complete."
echo "  Activate with:  source $VENV_DIR/bin/activate"
echo "  Benchmark with: python3 $PROJECT_DIR/pi_deploy/pi_benchmark.py \\"
echo "       --model $PROJECT_DIR/models/tflite_final/mnv3l_fold1_best.tflite"
echo "=================================================="
