# Training Guide for SAM3 LoRA

This guide provides comprehensive instructions for training LoRA-adapted SAM3 models with the enhanced training script.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Training Script](#training-script)
4. [Running Training](#running-training)
5. [Monitoring Training](#monitoring-training)
6. [Checkpoints and Resuming](#checkpoints-and-resuming)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Overview

The enhanced training script (`train_enhanced.py`) provides:

- **Comprehensive Logging:** All training metrics logged to file and console
- **Profiling:** Track timing and memory usage during training
- **Checkpointing:** Automatic checkpoint saving for best and last models
- **Validation:** Periodic validation with loss tracking
- **Learning Rate Scheduling:** Multiple scheduler options (cosine, linear, inverse sqrt)
- **Flexible Configuration:** YAML-based configuration system

## Configuration

### Configuration File Structure

Create a YAML configuration file for training. Example: `configs/lora_training_config.yaml`

```yaml
# Model configuration
model:
  checkpoint: "facebook/sam3-base"  # Base model
  
# LoRA configuration
lora:
  rank: 16                          # LoRA rank
  alpha: 32                         # LoRA scaling factor
  dropout: 0.1                      # Dropout in LoRA layers
  target_modules:                   # Modules to apply LoRA
    - "qkv"
    - "proj"
    - "mlp"

# Data configuration
data:
  train_dataset_path: "data/train"  # Training data directory
  val_dataset_path: "data/val"      # Validation data directory
  image_size: 1008                  # Input image size
  num_samples: 1000                 # Number of training samples

# Training configuration
training:
  epochs: 10                        # Number of epochs
  batch_size: 8                     # Batch size
  learning_rate: 1e-4               # Learning rate
  weight_decay: 0.01                # Weight decay
  optimizer: "adamw"                # Optimizer: adamw, adam, sgd
  scheduler: "cosine"               # Scheduler: cosine, linear, inverse_sqrt
  warmup_steps: 500                 # Warmup steps (for inverse_sqrt)
  num_workers: 4                    # Data loading workers
  gradient_clip: 1.0                # Gradient clipping
```

### Configuration Options

#### Model Configuration

| Option | Type | Description |
|--------|------|-------------|
| `checkpoint` | string | HuggingFace model ID or local path |

#### LoRA Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rank` | int | 8 | Rank of LoRA matrices |
| `alpha` | int | 16 | Scaling factor for LoRA |
| `dropout` | float | 0.1 | Dropout probability |
| `target_modules` | list | ["qkv"] | Modules to apply LoRA to |
| `use_rslora` | bool | false | Use Rank-Stabilized LoRA |

#### Data Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `train_dataset_path` | string | - | Path to training data |
| `val_dataset_path` | string | - | Path to validation data |
| `image_size` | int | 1008 | Model input resolution |
| `num_samples` | int | - | Number of training samples |

#### Training Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | 10 | Number of training epochs |
| `batch_size` | int | 8 | Batch size |
| `learning_rate` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 0.01 | Weight decay |
| `optimizer` | string | "adamw" | Optimizer type |
| `scheduler` | string | "cosine" | LR scheduler type |
| `warmup_steps` | int | 500 | Warmup steps |
| `num_workers` | int | 4 | Data loading workers |
| `gradient_clip` | float | 1.0 | Gradient clipping value |

## Training Script

### Script Features

The `train_enhanced.py` script includes:

1. **Setup Logging:** Configurable logging with file and console output
2. **Configuration Loading:** YAML configuration support
3. **Model Building:** SAM3 model with LoRA injection
4. **Data Loading:** Efficient batch loading with workers
5. **Training Loop:** Epoch-based training with metrics
6. **Validation:** Periodic validation evaluation
7. **Checkpointing:** Save best and last checkpoints
8. **Profiling:** Track timing and memory usage
9. **Reporting:** JSON summary of training results

### Command Line Arguments

```bash
python train_enhanced.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML configuration file |

#### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-dir` | `logs/training` | Directory for log files |
| `--checkpoint-dir` | `checkpoints/training` | Directory for model checkpoints |
| `--profile-report` | - | Path to save profiling report (JSON) |
| `--summary-json` | - | Path to save training summary (JSON) |

#### Training Arguments

| Argument | Flag | Description |
|----------|------|-------------|
| `--cpu` | - | Use CPU instead of GPU |
| `--resume` | - | Path to checkpoint to resume from |
| `--debug` | - | Enable debug logging |

## Running Training

### Basic Training

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml
```

### Training with Profiling

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --log-dir logs/training_run1 \
    --profile-report logs/training_run1/profile.json \
    --summary-json logs/training_run1/summary.json
```

### Training with Debug Logging

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --debug
```

### Training on CPU (for testing)

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --cpu
```

### Resume Training from Checkpoint

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --resume checkpoints/training/last_checkpoint.pt
```

## Monitoring Training

### Log Files

Training logs are saved to `logs/training/training.log` by default:

```
2024-01-15 10:30:45,123 - __main__ - INFO - Loading configuration from: configs/lora_training_config.yaml
2024-01-15 10:30:47,234 - __main__ - INFO - Building Model
2024-01-15 10:30:52,456 - __main__ - INFO - Model parameters: 95,000,000 (trainable: 50,000)
2024-01-15 10:31:02,789 - __main__ - INFO - Loading Data
```

### Console Output

The training script displays:

1. **Epoch Header:**
   ```
   ============================================================
   Epoch 1/10
   ============================================================
   ```

2. **Progress Bar:**
   ```
   Epoch 1: 100%|████████| 125/125 [5:23<00:00, 2.59s/batch]
   loss: 0.2345, lr: 0.000100
   ```

3. **Epoch Summary:**
   ```
   Epoch 1 - Average Loss: 0.2345
   Epoch 1 - Validation Loss: 0.2156
   ```

### Real-time Metrics

Monitor key metrics during training:

- **Loss:** Training and validation loss
- **Learning Rate:** Current learning rate
- **Batch Time:** Time per batch
- **Memory:** GPU/CPU memory usage

### Checkpointing

The script saves:

1. **Last Checkpoint:** `checkpoints/training/last_checkpoint.pt`
   - Updated every epoch
   - Contains latest model state

2. **Best Checkpoint:** `checkpoints/training/best_checkpoint.pt`
   - Saved when validation loss improves
   - Use this for inference

## Checkpoints and Resuming

### Checkpoint Structure

Each checkpoint contains:

```python
{
    "epoch": 5,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "metrics": {
        "total_steps": 500,
        "final_loss": 0.1234,
        "best_loss": 0.1134,
        "best_epoch": 3,
        ...
    }
}
```

### Saving Checkpoints

Automatic checkpointing:
- Every epoch (last checkpoint)
- When validation loss improves (best checkpoint)

Custom checkpoint saving:
```python
from train_enhanced import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    metrics=metrics,
    checkpoint_path="checkpoints/custom_checkpoint.pt",
)
```

### Loading Checkpoints

```python
import torch

checkpoint = torch.load("checkpoints/training/best_checkpoint.pt")
epoch = checkpoint["epoch"]
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

### Resuming Training

Resume from a checkpoint:

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --resume checkpoints/training/last_checkpoint.pt
```

## Performance Optimization

### Training Speed

Optimize training speed:

1. **Increase Batch Size:** Higher batch sizes are faster but use more memory
   ```yaml
   training:
     batch_size: 16  # Increase if GPU memory allows
   ```

2. **Reduce Number of Workers:** If CPU is bottleneck
   ```yaml
   training:
     num_workers: 2
   ```

3. **Mixed Precision Training:**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   # Use autocast for forward pass
   ```

4. **Gradient Accumulation:** For larger effective batch size
   ```python
   for i, batch in enumerate(loader):
       outputs = model(batch)
       loss = outputs.loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
   ```

### Memory Optimization

Reduce memory usage:

1. **Smaller LoRA Rank:**
   ```yaml
   lora:
     rank: 8  # Default 16
   ```

2. **Lower Resolution:**
   ```yaml
   data:
     image_size: 512  # Default 1008
   ```

3. **Smaller Batch Size:**
   ```yaml
   training:
     batch_size: 4
   ```

4. **Gradient Checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

### Multi-GPU Training

For multi-GPU training, use `torch.nn.DataParallel`:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size
2. Reduce image resolution
3. Reduce LoRA rank
4. Enable gradient checkpointing
5. Use CPU for testing: `--cpu`

### Issue: "No module named 'sam3'"

**Solution:**
```bash
# Ensure SAM3 is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Issue: "Data loading is too slow"

**Solutions:**
1. Increase `num_workers`:
   ```yaml
   training:
     num_workers: 8
   ```

2. Reduce dataset size for testing:
   ```yaml
   data:
     num_samples: 100
   ```

3. Use faster storage (SSD)

### Issue: "Loss not decreasing"

**Solutions:**
1. Increase learning rate:
   ```yaml
   training:
     learning_rate: 5e-4
   ```

2. Check data quality and labels
3. Verify data loading (check first batch)
4. Try different optimizer:
   ```yaml
   training:
     optimizer: "adam"
   ```

### Issue: "Training is too slow"

**Solutions:**
1. Increase batch size if memory allows
2. Reduce number of validation steps
3. Use mixed precision training
4. Profile to identify bottleneck: `--profile-report profile.json`

### Issue: "Validation loss is NaN"

**Solutions:**
1. Check for invalid data in dataset
2. Reduce learning rate
3. Enable gradient clipping (enabled by default)
4. Check for numerical stability in loss calculation

## Best Practices

1. **Start Small:** Use small dataset/config to test first
2. **Monitor Metrics:** Watch loss curves for anomalies
3. **Save Checkpoints:** Always have multiple checkpoints
4. **Version Control:** Track configs and results
5. **Document Runs:** Keep notes on hyperparameters tried
6. **Profile Early:** Identify bottlenecks early
7. **Validate Frequently:** Check validation metrics
8. **Use Tensorboard:** Visualize training (optional enhancement)

## Advanced Topics

### Custom Loss Functions

Implement custom loss in training loop:

```python
def custom_loss(outputs, batch):
    # Your loss implementation
    return loss

loss = custom_loss(outputs, batch)
loss.backward()
```

### Learning Rate Warmup

Custom warmup with inverse sqrt scheduler:

```yaml
training:
  scheduler: "inverse_sqrt"
  warmup_steps: 1000
```

### Distributed Training

For multi-node distributed training:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_enhanced.py \
    --config configs/lora_training_config.yaml
```

## Next Steps

After training:

1. **Evaluate Model:** Use inference script on test data
2. **Export Weights:** Save LoRA weights for deployment
3. **Profile Inference:** Check inference performance
4. **Create Visualizations:** Generate result plots

See `INFERENCE.md` for inference instructions.
