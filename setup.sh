#!/bin/bash
# Setup script for SAM3 LoRA project with pyvenv and pip

set -e  # Exit on error

echo "==========================================="
echo "SAM3 LoRA Setup Script"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect Python installation
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Check Python version
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python 3.8+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"

echo ""
echo "==========================================="
echo "Step 1: Creating Virtual Environment"
echo "==========================================="

# Remove existing venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at: $VENV_DIR"
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

echo ""
echo "==========================================="
echo "Step 2: Activating Virtual Environment"
echo "==========================================="

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo ""
echo "==========================================="
echo "Step 3: Upgrading pip, setuptools, wheel"
echo "==========================================="

pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip, setuptools, wheel upgraded${NC}"

echo ""
echo "==========================================="
echo "Step 4: Installing Dependencies"
echo "==========================================="

# Check if requirements.txt exists
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
fi

# Install development dependencies if available
if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
    echo "Installing development requirements..."
    pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    echo -e "${GREEN}✓ Development requirements installed${NC}"
fi

echo ""
echo "==========================================="
echo "Step 5: Installing Package in Development Mode"
echo "==========================================="

if [ -f "$PROJECT_ROOT/setup.py" ]; then
    echo "Installing package in development mode..."
    pip install -e "$PROJECT_ROOT"
    echo -e "${GREEN}✓ Package installed in development mode${NC}"
fi

echo ""
echo "==========================================="
echo "Step 6: Verifying Installation"
echo "==========================================="

# Verify torch installation
echo "Checking PyTorch installation..."
$PYTHON_CMD -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"

# Verify transformers installation
echo "Checking Transformers installation..."
$PYTHON_CMD -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo -e "${YELLOW}  source venv/bin/activate${NC}"
echo ""
echo "To deactivate the virtual environment, run:"
echo -e "${YELLOW}  deactivate${NC}"
echo ""
echo "To start training:"
echo -e "${YELLOW}  python train_enhanced.py --config configs/lora_config_example.yaml${NC}"
echo ""
echo "To run inference:"
echo -e "${YELLOW}  python inference_enhanced.py --config configs/full_lora_config.yaml --image path/to/image.jpg --weights path/to/weights.pt${NC}"
echo ""
