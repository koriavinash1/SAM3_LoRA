#!/bin/bash
# Setup script for SAM3 LoRA project with poetry

set -e  # Exit on error

echo "==========================================="
echo "SAM3 LoRA Setup Script (with Poetry)"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry is not installed${NC}"
    echo ""
    echo "Please install Poetry from https://python-poetry.org/docs/#installation"
    echo ""
    echo "Quick install:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

POETRY_VERSION=$(poetry --version)
echo -e "${GREEN}✓ Found ${POETRY_VERSION}${NC}"

echo ""
echo "==========================================="
echo "Step 1: Installing Dependencies with Poetry"
echo "==========================================="

poetry install

echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""
echo "==========================================="
echo "Step 2: Verifying Installation"
echo "==========================================="

poetry run python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
poetry run python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "To activate the poetry environment, run:"
echo -e "${YELLOW}  poetry shell${NC}"
echo ""
echo "Or to run commands in the poetry environment:"
echo -e "${YELLOW}  poetry run python train_enhanced.py --config configs/lora_config_example.yaml${NC}"
echo ""
