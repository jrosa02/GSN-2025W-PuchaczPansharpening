#!/usr/bin/env bash
set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as an error

# -------------------------------
# Activate virtual environment
# -------------------------------
VENV_DIR="./.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    # Bash activation
    source "$VENV_DIR/bin/activate"
else
    echo "Error: .venv not found. Please create it first:"
    echo "       python -m venv $VENV_DIR"
    exit 1
fi

# -------------------------------
# Detect installer
# -------------------------------
if command -v uv >/dev/null 2>&1; then
    INSTALLER="uv pip"
    echo "Using 'uv' as installer"
else
    INSTALLER="pip"
    echo "'uv' not found, falling back to pip"
fi

# -------------------------------
# Detect NVIDIA GPU
# -------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PRESENT=true
    echo "NVIDIA GPU detected, installing standard torch"
else
    GPU_PRESENT=false
    echo "No NVIDIA GPU detected, installing CPU-only torch"
fi

# -------------------------------
# Install PyTorch
# -------------------------------
if [ "$GPU_PRESENT" = true ]; then
    $INSTALLER install torch torchvision torchaudio --upgrade
else
    $INSTALLER install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --upgrade
fi

# -------------------------------
# Install other packages
# -------------------------------
$INSTALLER install pytorch_lightning --upgrade
$INSTALLER install torchsummary tensorboardX tensorboard --upgrade
$INSTALLER install kaggle --upgrade
$INSTALLER install matplotlib --upgrade
$INSTALLER install pystac-client planetary-computer --upgrade
$INSTALLER install tifffile --upgrade
$INSTALLER install opencv-python --upgrade
$INSTALLER install rasterio --upgrade
$INSTALLER install zarr --upgrade
$INSTALLER install pytest --upgrade

echo "All packages installed successfully!"
