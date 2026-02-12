#!/bin/bash
# ============================================================
#  ALL Detection System — Raspberry Pi Setup Script
#  Target: Raspberry Pi 5 (Bookworm 64-bit)
#  Usage:  chmod +x setup_pi.sh && ./setup_pi.sh
# ============================================================

set -e  # Exit on any error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo "=========================================="
echo "  ALL Detection System — Pi Setup"
echo "=========================================="
echo ""

# ─── 1. System packages ─────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq

# libopenblas-dev is more reliable on Pi OS Bookworm than libatlas-base-dev
PACKAGES="python3-pip python3-venv python3-dev python3-tk python3-pil python3-pil.imagetk libopenblas-dev libhdf5-dev fonts-roboto curl gfortran"

if ! sudo apt-get install -y -qq $PACKAGES; then
    warn "Some individual packages failed. Attempting to install core dependencies only..."
    sudo apt-get install -y python3-pip python3-venv python3-tk libopenblas-dev curl
fi

# ─── 2. Python virtual environment ──────────────────────────
VENV_DIR="$HOME/all-detection-venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" --system-site-packages
else
    info "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
info "Activated venv: $(python3 --version)"

# ─── 3. Python dependencies ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

info "Installing Python packages..."
pip install --upgrade pip -q
pip install -q \
    customtkinter>=5.2.0 \
    Pillow>=9.0.0 \
    numpy>=1.21.0 \
    opencv-python-headless>=4.5.0 \
    scikit-learn>=1.0.0 \
    scikit-image>=0.18.0 \
    scipy>=1.7.0 \
    tflite-runtime \
    ollama>=0.1.0 \
    tqdm>=4.62.0 \
    pyyaml>=6.0

info "Python packages installed ✓"

# ─── 4. Ollama ── ────────────────────────────────────────────
if command -v ollama &> /dev/null; then
    info "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown')"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    info "Ollama installed ✓"
fi

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    sleep 3
fi

# ─── 5. Pull AI model ───────────────────────────────────────
info "Pulling phi3 model (this may take a few minutes on first run)..."
ollama pull phi3
info "phi3 model ready ✓"

# ─── 6. Create launcher script ──────────────────────────────
LAUNCHER="$SCRIPT_DIR/run.sh"
cat > "$LAUNCHER" << 'EOF'
#!/bin/bash
# Launch ALL Detection System
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/all-detection-venv/bin/activate"
cd "$SCRIPT_DIR/src"
python3 ui/app.py "$@"
EOF
chmod +x "$LAUNCHER"
info "Created launcher: $LAUNCHER"

# ─── 7. Desktop shortcut (optional) ─────────────────────────
DESKTOP_FILE="$HOME/Desktop/ALL-Detection.desktop"
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=ALL Detection System
Comment=Acute Lymphoblastic Leukemia Detection
Exec=$LAUNCHER
Icon=utilities-terminal
Terminal=false
Type=Application
Categories=Science;MedicalSoftware;
EOF
chmod +x "$DESKTOP_FILE"
info "Desktop shortcut created ✓"

# ─── Done ────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo -e "  ${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "  To run:  ./run.sh"
echo "  Or double-click 'ALL Detection' on the Desktop"
echo ""
