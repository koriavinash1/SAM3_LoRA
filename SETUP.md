# Setup Guide for SAM3 LoRA Project

This guide provides detailed instructions for setting up the SAM3 LoRA project on various systems.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Setup Methods](#setup-methods)
4. [Virtual Environment Setup](#virtual-environment-setup)
5. [Dependency Installation](#dependency-installation)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended for training)
- **Disk Space**: 50 GB (for model checkpoints and datasets)
- **GPU** (optional but recommended):
  - NVIDIA GPU with CUDA Compute Capability 6.1 or higher
  - CUDA 11.8 or higher
  - cuDNN 8.0 or higher

### Supported Platforms

- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.15+, Intel and Apple Silicon)
- Windows 10/11 (with WSL2 recommended)

## Quick Start

### For Linux/macOS

```bash
# Clone the repository
git clone https://github.com/yourusername/sam3_lora.git
cd sam3_lora

# Run setup script with pyvenv
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### For Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/yourusername/sam3_lora.git
cd sam3_lora

# Run setup with pyvenv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Setup Methods

### Method 1: Using pyvenv (Recommended for Beginners)

pyvenv is Python's built-in virtual environment manager, ideal for simple setups.

**Advantages:**
- No external dependencies
- Built into Python 3.3+
- Simple and straightforward

**Steps:**

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate Virtual Environment:**
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows (CMD):
     ```batch
     venv\Scripts\activate.bat
     ```
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Package in Development Mode:**
   ```bash
   pip install -e .
   ```

### Method 2: Using Poetry (Recommended for Advanced Users)

Poetry provides reproducible builds and lockfile support.

**Advantages:**
- Reproducible dependency versions
- Elegant dependency management
- Script management via `pyproject.toml`
- Version conflict resolution

**Prerequisites:**
Install Poetry from https://python-poetry.org/docs/#installation

**Steps:**

1. **Navigate to Project Directory:**
   ```bash
   cd sam3_lora
   ```

2. **Install Dependencies:**
   ```bash
   poetry install
   ```

3. **Activate Virtual Environment:**
   ```bash
   poetry shell
   ```

4. **Or Run Commands Directly:**
   ```bash
   poetry run python train_enhanced.py --config configs/lora_config_example.yaml
   ```

### Method 3: Using UV (Fastest Alternative)

uv is an extremely fast Python package installer written in Rust.

**Installation:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Setup:**
```bash
cd sam3_lora
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e .
```

## Virtual Environment Setup

### Creating Virtual Environment

```bash
# Using pyvenv
python3 -m venv venv

# Using Poetry
poetry env use python3.10  # Specify Python version if needed
poetry install

# Using uv
uv venv
```

### Activating Virtual Environment

**Linux/macOS (bash/zsh):**
```bash
source venv/bin/activate
```

**Linux/macOS (fish):**
```fish
source venv/bin/activate.fish
```

**Windows (CMD):**
```batch
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**With Poetry:**
```bash
poetry shell
```

### Deactivating Virtual Environment

```bash
deactivate
```

## Dependency Installation

### Core Dependencies

The project requires several key dependencies:

| Package | Purpose | Min Version |
|---------|---------|-------------|
| torch | Deep learning framework | 2.7.0 |
| transformers | HuggingFace model library | 4.48.0 |
| torchvision | Computer vision utilities | 0.19.0 |
| Pillow | Image processing | 10.0.0 |
| opencv-python | Advanced image processing | 4.8.0 |
| PyYAML | Configuration management | 6.0 |
| tqdm | Progress bars | 4.65.0 |
| matplotlib | Visualization | 3.7.0 |
| numpy | Numerical computing | 1.24.0 |
| pandas | Data analysis | 1.5.0 |

### Installation Methods

**Install from requirements.txt:**
```bash
pip install -r requirements.txt
```

**Install from pyproject.toml (Poetry):**
```bash
poetry install
```

**Install specific packages:**
```bash
pip install torch>=2.7.0 transformers>=4.48.0
```

### CUDA/GPU Support

For GPU training, ensure CUDA compatibility:

**Check NVIDIA CUDA Toolkit:**
```bash
nvidia-smi
```

**PyTorch GPU Installation:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify Installation:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Verification

### Check Python Version

```bash
python --version  # Should be 3.8 or higher
```

### Verify Core Dependencies

```python
import torch
import transformers
import numpy as np
import opencv as cv2

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Run Verification Script

```bash
python -c "
import torch
import transformers
from transformers import Sam3Model, Sam3Processor

print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ All dependencies verified!')
"
```

### Test Import of SAM3 Models

```bash
python -c "
from transformers import Sam3Model, Sam3Processor
model = Sam3Model.from_pretrained('facebook/sam3-base')
print('✓ SAM3 model loaded successfully')
"
```

## Troubleshooting

### Issue: "Python command not found"

**Solution:**
- Linux/macOS: Use `python3` instead of `python`
- Windows: Add Python to PATH or use full path

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install --upgrade pip
pip install torch torchvision
```

### Issue: "CUDA/GPU not detected"

**Solution:**
1. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. Check CUDA compatibility:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

### Issue: "Permission denied" for setup.sh

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

### Issue: "Virtual environment already exists"

**Solution:**
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### Issue: Package installation fails with "Collecting error"

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try with specific index
pip install -i https://pypi.org/simple/ package_name

# Clear cache
pip cache purge
pip install -r requirements.txt
```

### Issue: ImportError with SAM3 module

**Solution:**
1. Ensure SAM3 path is correctly set in `sam3/__init__.py`
2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Check PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Issue: Out of Memory during model loading

**Solution:**
- Reduce batch size in config
- Use CPU for inference: `--cpu` flag
- Use smaller model variant if available

### Getting Help

- Check documentation in `README.md`
- Review training guide in `TRAINING.md`
- Check inference guide in `INFERENCE.md`
- Review profiling guide in `PROFILING.md`

### Creating Issues

When reporting issues, include:
```bash
python -c "import torch; print(torch.cuda.is_available())"
pip list
python --version
uname -a  # Linux/macOS
systeminfo  # Windows
```

## Next Steps

After successful setup:

1. **Review Project Structure:**
   - Read `README.md` for project overview
   - Check `src/` directory for code organization

2. **Start Training:**
   - Review `TRAINING.md` for detailed training instructions
   - Run: `python train_enhanced.py --config configs/lora_config_example.yaml`

3. **Run Inference:**
   - Review `INFERENCE.md` for inference instructions
   - Run: `python inference_enhanced.py --config configs/full_lora_config.yaml --image path/to/image.jpg --weights path/to/weights.pt`

4. **Explore Profiling:**
   - Review `PROFILING.md` for profiling information
   - Enable profiling in training/inference scripts

## Additional Resources

- **PyTorch Documentation:** https://pytorch.org/docs/
- **Transformers Documentation:** https://huggingface.co/docs/transformers/
- **SAM3 Documentation:** https://github.com/facebookresearch/sam3
- **Poetry Documentation:** https://python-poetry.org/docs/
- **uv Documentation:** https://docs.astral.sh/uv/
