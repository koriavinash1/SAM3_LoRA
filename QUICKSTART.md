# Quick Start Guide for SAM3 LoRA

Get started with SAM3 LoRA in 5 minutes!

## 1. Setup (2 minutes)

### Clone Repository
```bash
git clone https://github.com/yourusername/sam3_lora.git
cd sam3_lora
```

### Run Setup Script
```bash
# Linux/macOS
chmod +x setup.sh
./setup.sh

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 2. Prepare Your Data (1 minute)

Organize your data in COCO format:

```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

## 3. Create Configuration (1 minute)

Create `configs/my_config.yaml`:

```yaml
model:
  checkpoint: "facebook/sam3-base"

lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["qkv", "proj"]

data:
  train_dataset_path: "data/train"
  val_dataset_path: "data/val"
  image_size: 1008

training:
  epochs: 5
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"
  num_workers: 4
```

## 4. Train Model (1 minute setup, runs in background)

```bash
python train_enhanced.py \
    --config configs/my_config.yaml \
    --log-dir logs/training \
    --checkpoint-dir checkpoints/training
```

**Monitor training:**
```bash
# In another terminal
tail -f logs/training/training.log
```

### Expected Output
```
============================================================
SAM3 LoRA Enhanced Training
============================================================
Using device: cuda
...
Epoch 1/5: 100%|████████| 125/125 [5:23<00:00, 2.59s/batch]
loss: 0.2345, lr: 0.000100
Epoch 1 - Average Loss: 0.2345
```

## 5. Run Inference (Take 1 minute test)

### Single Image Inference

```bash
python inference_enhanced.py \
    --weights checkpoints/training/best_checkpoint.pt \
    --image path/to/test_image.jpg \
    --prompt "crack" \
    --output results/prediction.png
```

### Multiple Images (Batch)

```bash
python batch_inference.py \
    --weights checkpoints/training/best_checkpoint.pt \
    --image-dir data/test_images \
    --prompt "crack" "damage" \
    --output-dir results/batch_results
```

## 6. View Results

```bash
# View inference image
open results/prediction.png  # macOS
xdg-open results/prediction.png  # Linux
start results/prediction.png  # Windows

# View results JSON
cat results/batch_results/batch_results.json | python -m json.tool
```

---

## Key Commands Reference

### Training
```bash
# Basic training
python train_enhanced.py --config CONFIG_FILE

# Training with profiling
python train_enhanced.py \
    --config CONFIG_FILE \
    --profile-report logs/profile.json \
    --summary-json logs/summary.json

# Resume training
python train_enhanced.py \
    --config CONFIG_FILE \
    --resume checkpoints/training/last_checkpoint.pt

# CPU training (for testing)
python train_enhanced.py --config CONFIG_FILE --cpu
```

### Inference
```bash
# Single image, single prompt
python inference_enhanced.py \
    --weights weights.pt \
    --image image.jpg \
    --prompt "crack"

# Single image, multiple prompts
python inference_enhanced.py \
    --weights weights.pt \
    --image image.jpg \
    --prompt "crack" "damage" "defect"

# Multiple images (batch)
python batch_inference.py \
    --weights weights.pt \
    --image-dir images_dir \
    --prompt "crack"

# With profiling
python inference_enhanced.py \
    --weights weights.pt \
    --image image.jpg \
    --prompt "crack" \
    --profile-report profile.json \
    --results-json results.json
```

---

## Common Issues

### Q: CUDA out of memory
**A:** Reduce batch size in config:
```yaml
training:
  batch_size: 4  # from 8
```

### Q: No detections in inference
**A:** Try lower threshold:
```bash
python inference_enhanced.py \
    --weights weights.pt \
    --image image.jpg \
    --prompt "crack" \
    --threshold 0.3  # from 0.5
```

### Q: Slow data loading
**A:** Increase workers:
```yaml
training:
  num_workers: 8  # from 4
```

### Q: Training loss not decreasing
**A:** Try higher learning rate:
```yaml
training:
  learning_rate: 5e-4  # from 1e-4
```

---

## Next Steps

1. **Read Full Documentation:**
   - [SETUP.md](SETUP.md) - Detailed setup instructions
   - [TRAINING.md](TRAINING.md) - Training guide
   - [INFERENCE.md](INFERENCE.md) - Inference guide
   - [PROFILING.md](PROFILING.md) - Performance profiling

2. **Create Your First Model:**
   - Prepare your dataset (COCO format)
   - Copy and modify a config file
   - Run training with `train_enhanced.py`
   - Evaluate on validation data

3. **Deploy and Share:**
   - Export LoRA weights (10-50MB)
   - Build API with Flask/FastAPI
   - Share with team/community

---

## Performance Tips

### For Faster Training
- Increase batch size (if VRAM allows)
- Use fewer validation samples
- Enable mixed precision (FP16)
- Use multiple GPUs

### For Faster Inference
- Use smaller LoRA rank (8 instead of 16)
- Reduce image resolution
- Enable GPU inference
- Batch multiple images

### For Lower Memory Usage
- Reduce batch size
- Reduce image resolution
- Reduce LoRA rank
- Enable gradient checkpointing

---

## Configuration Examples

Pre-made configurations are available in `configs/` directory:

```bash
# List available configs
ls configs/

# Use a config
python train_enhanced.py --config configs/basic_lora_config.yaml

# Create new config from template
cp configs/basic_lora_config.yaml configs/my_custom_config.yaml
```

---

## Getting Help

**Before asking for help, try:**

1. Check logs: `tail -f logs/training/training.log`
2. Enable debug: `--debug` flag
3. Review relevant documentation
4. Check GitHub issues

**When creating issue, include:**
- Your environment (Python/PyTorch/CUDA versions)
- Full error message
- Steps to reproduce
- Your configuration file

---

## Related Resources

- **SAM3:** https://github.com/facebookresearch/sam3
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Transformers:** https://huggingface.co/docs/transformers/
- **PyTorch:** https://pytorch.org/docs/

---

**Congratulations! You're ready to train and use SAM3 LoRA models! 🎉**
