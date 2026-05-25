# Profiling Guide for SAM3 LoRA

This guide provides comprehensive information about profiling training and inference with the enhanced scripts.

## Table of Contents

1. [Overview](#overview)
2. [Profiling Utilities](#profiling-utilities)
3. [Training Profiling](#training-profiling)
4. [Inference Profiling](#inference-profiling)
5. [Interpreting Results](#interpreting-results)
6. [Performance Analysis](#performance-analysis)
7. [Optimization Recommendations](#optimization-recommendations)
8. [Advanced Profiling](#advanced-profiling)

## Overview

Profiling helps identify performance bottlenecks and optimize:

1. **Timing:** Measure execution time of different stages
2. **Memory:** Track GPU/CPU memory usage
3. **Throughput:** Calculate samples per second
4. **Efficiency:** Identify inefficient code paths

### Key Metrics

- **Latency:** Time to process a single sample
- **Throughput:** Samples processed per unit time
- **Memory Peak:** Maximum memory used during execution
- **Memory Efficiency:** Memory used per sample

## Profiling Utilities

### Profiler Class

The `Profiler` class from `src.utils.profiling_utils` provides:

```python
from src.utils import Profiler

# Create profiler
profiler = Profiler(name="my_profiling_task")

# Time code blocks
with profiler.context("operation_name"):
    # Your code here
    pass

# Measure memory
profiler.measure_memory("measurement_name")

# Get summary
summary = profiler.get_summary()

# Print summary
profiler.print_summary()

# Save to JSON
profiler.save_summary("profile.json")
```

### Profiler Methods

#### start_timer(name: str)
Start timing a named operation.

```python
profiler.start_timer("data_loading")
# ... operation ...
profiler.stop_timer()
```

#### stop_timer()
Stop the current timer and record measurement.

#### context(name: str)
Context manager for timing code blocks.

```python
with profiler.context("inference"):
    model.forward(inputs)
```

#### measure_memory(name: str)
Measure current memory usage.

```python
profiler.measure_memory("before_training")
# ... training ...
profiler.measure_memory("after_training")
```

#### get_summary()
Get profiling summary as dictionary.

```python
summary = profiler.get_summary()
print(summary["timers"])
print(summary["memory"])
```

#### print_summary()
Print formatted profiling results.

```python
profiler.print_summary()
```

Output example:
```
============================================================
Profiling Summary: full_inference
============================================================

⏱️  Timing Statistics:
Measurement                Count    Mean        Std         Min         Max
------
model_loading             1        5.234       0.000       5.234       5.234
preprocessing             1        0.123       0.000       0.123       0.123
inference                 10       0.245       0.012       0.234       0.267

💾 Memory Statistics:

before_inference:
  Peak Memory: 2048.50 MB
  Current Memory: 2048.50 MB
  Allocated Memory: 2100.00 MB
============================================================
```

#### save_summary(output_path: str)
Save profiling summary to JSON file.

```python
profiler.save_summary("results/profile.json")
```

## Training Profiling

### Enabling Profiling in Training

```bash
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --profile-report logs/training/profile.json \
    --summary-json logs/training/summary.json
```

### Profiling Output

The training script tracks:

1. **Model Building Time**
   ```json
   "model_building": {
     "count": 1,
     "total_time": 8.234,
     "mean_time": 8.234
   }
   ```

2. **Data Loading Time**
   ```json
   "data_loading": {
     "count": 1,
     "total_time": 12.567,
     "mean_time": 12.567
   }
   ```

3. **Per-Batch Timing**
   ```json
   "batch_0": {
     "count": 1,
     "total_time": 0.456,
     "mean_time": 0.456
   }
   ```

4. **Epoch Timing**
   - Total epoch time
   - Per-batch average
   - Training and validation

### Training Metrics

The training script also logs:

```json
{
  "total_steps": 1250,
  "final_loss": 0.1234,
  "best_loss": 0.1134,
  "best_epoch": 3,
  "avg_loss": 0.2345,
  "avg_lr": 0.0001,
  "avg_step_time": 0.456
}
```

### Example Training Profile

```bash
# Run training with profiling
python train_enhanced.py \
    --config configs/lora_training_config.yaml \
    --profile-report profile.json \
    --summary-json summary.json

# View profile
cat profile.json | python -m json.tool
```

Sample output:
```json
{
  "name": "full_training",
  "timers": {
    "model_building": {
      "count": 1,
      "total_time": 8.234,
      "mean_time": 8.234,
      "std_time": 0.0,
      "min_time": 8.234,
      "max_time": 8.234
    },
    "data_loading": {
      "count": 1,
      "total_time": 12.567,
      "mean_time": 12.567,
      "std_time": 0.0,
      "min_time": 12.567,
      "max_time": 12.567
    },
    "training_loop": {
      "count": 1,
      "total_time": 2345.678,
      "mean_time": 2345.678,
      "std_time": 0.0,
      "min_time": 2345.678,
      "max_time": 2345.678
    }
  },
  "memory": {
    "after_training": {
      "peak_memory_mb": 2048.5,
      "current_memory_mb": 512.3,
      "allocated_memory_mb": 2100.0
    }
  }
}
```

## Inference Profiling

### Enabling Profiling in Inference

```bash
python inference_enhanced.py \
    --weights path/to/weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --profile-report logs/inference/profile.json \
    --results-json logs/inference/results.json
```

### Profiling Output

The inference script tracks:

1. **Model Loading**
   - Base model loading
   - LoRA application
   - LoRA weights loading
   - Move to device

2. **Preprocessing**
   - Image loading
   - Image preprocessing

3. **Inference**
   - Per-prompt inference time
   - Total inference time

4. **Visualization**
   - Visualization generation time

5. **Memory**
   - Before inference
   - After inference

### Example Inference Profile

```bash
# Run inference with profiling
python inference_enhanced.py \
    --weights checkpoints/best_weights.pt \
    --image data/test_image.jpg \
    --prompt "crack" "damage" \
    --profile-report inference_profile.json \
    --results-json inference_results.json

# View profile
python -c "
import json
with open('inference_profile.json') as f:
    profile = json.load(f)
    print('Inference Profiling Results:')
    for timer_name, stats in profile['timers'].items():
        print(f'{timer_name}: {stats[\"mean_time\"]:.4f}s')
"
```

## Interpreting Results

### Timing Statistics

Each timing measurement includes:

| Metric | Meaning |
|--------|---------|
| `count` | Number of measurements |
| `total_time` | Sum of all measurements |
| `mean_time` | Average time |
| `std_time` | Standard deviation |
| `min_time` | Minimum time |
| `max_time` | Maximum time |

### Memory Statistics

| Metric | Meaning |
|--------|---------|
| `peak_memory_mb` | Peak GPU memory used |
| `current_memory_mb` | Current memory allocated |
| `allocated_memory_mb` | Total allocated memory |

### Example Analysis

```python
import json

with open('profile.json') as f:
    profile = json.load(f)

# Analyze inference timing
print("=== Inference Timing Analysis ===")
total_time = profile['timers']['full_inference']['total_time']
for name, timer in profile['timers'].items():
    if name != 'full_inference':
        percentage = (timer['total_time'] / total_time) * 100
        print(f"{name}: {timer['mean_time']:.4f}s ({percentage:.1f}%)")

# Analyze memory
print("\n=== Memory Analysis ===")
for name, mem in profile['memory'].items():
    print(f"{name}:")
    print(f"  Peak: {mem['peak_memory_mb']:.1f} MB")
    print(f"  Current: {mem['current_memory_mb']:.1f} MB")
```

## Performance Analysis

### Throughput Calculation

```python
import json

with open('profile.json') as f:
    profile = json.load(f)

# Calculate throughput (images per second)
inference_time = profile['timers']['full_inference']['mean_time']
throughput = 1.0 / inference_time
print(f"Throughput: {throughput:.2f} images/second")
```

### Latency Breakdown

```python
# Analyze latency components
timers = profile['timers']

components = {
    'Model Loading': timers.get('load_base_model', {}).get('mean_time', 0),
    'LoRA Setup': timers.get('apply_lora', {}).get('mean_time', 0),
    'Preprocessing': timers.get('load_image', {}).get('mean_time', 0),
    'Inference': timers.get('full_inference', {}).get('mean_time', 0),
    'Visualization': timers.get('create_visualization', {}).get('mean_time', 0),
}

total = sum(components.values())
for name, time in components.items():
    percentage = (time / total) * 100
    print(f"{name}: {time:.4f}s ({percentage:.1f}%)")
```

### Memory Efficiency

```python
# Calculate memory efficiency
memory = profile['memory']['after_inference']
peak_memory = memory['peak_memory_mb']

print(f"Peak Memory Usage: {peak_memory:.1f} MB")
print(f"Memory per Image: {peak_memory:.1f} MB")
print(f"GPU Available: Check with nvidia-smi")
```

## Optimization Recommendations

### Based on Profiling Results

#### If Model Loading is Slow (>5s)
1. Use smaller model variant
2. Enable model caching
3. Consider quantization

#### If Inference is Slow (>2s)
1. Reduce image resolution
2. Use smaller LoRA rank
3. Enable half-precision (FP16)

#### If Memory Usage is High (>4GB)
1. Reduce batch size
2. Reduce image resolution
3. Use gradient checkpointing

#### If Data Loading is Slow
1. Use multiple workers:
   ```yaml
   training:
     num_workers: 8
   ```
2. Use faster storage (SSD)
3. Preprocess and cache data

## Advanced Profiling

### PyTorch Profiler Integration

For deeper profiling, use PyTorch's native profiler:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Custom Profiling

Add profiling to custom code:

```python
from src.utils import Profiler

profiler = Profiler(name="custom_inference")

with profiler.context("preprocessing"):
    image = preprocess(image)

with profiler.context("model_forward"):
    outputs = model(image)

profiler.print_summary()
profiler.save_summary("custom_profile.json")
```

### Visualization

Create plots from profiling data:

```python
import json
import matplotlib.pyplot as plt

with open('profile.json') as f:
    profile = json.load(f)

# Extract timing data
names = list(profile['timers'].keys())
times = [profile['timers'][n]['mean_time'] for n in names]

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(names, times)
plt.xlabel('Operation')
plt.ylabel('Time (seconds)')
plt.title('Inference Timing Breakdown')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timing_breakdown.png')
```

## Best Practices

1. **Profile Early:** Don't wait until deployment to profile
2. **Baseline First:** Establish baseline before optimization
3. **Measure Multiple Times:** Account for variance
4. **Profile in Real Conditions:** Use actual data and hardware
5. **Document Results:** Keep profiling results for comparison
6. **Iterative Optimization:** Profile, optimize, repeat
7. **Profile All Stages:** Don't optimize just one part

## Performance Targets

### Recommended Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model Loading | <5s | One-time cost |
| Inference Latency | <1s | Per image |
| Throughput | >1 img/s | Depends on resolution |
| Memory Peak | <4GB | For single GPU |
| Training Time | <24h | For 10 epochs |

## Benchmarking

### Benchmark Script

```python
#!/usr/bin/env python3
"""Benchmark inference performance."""

import torch
import time
from pathlib import Path
from inference_enhanced import load_model_with_lora, preprocess_image, run_inference

# Configuration
weights_path = "checkpoints/best_weights.pt"
image_path = "data/test_image.jpg"
num_runs = 5

# Load model
model, processor, _ = load_model_with_lora(
    base_model_name="facebook/sam3-base",
    lora_weights_path=weights_path,
)

# Load image
image, _, _ = preprocess_image(image_path, processor)

# Warmup
for _ in range(2):
    run_inference(model, processor, image, ["crack"])

# Benchmark
times = []
for i in range(num_runs):
    start = time.time()
    results = run_inference(model, processor, image, ["crack"])
    elapsed = time.time() - start
    times.append(elapsed)

# Results
print(f"Average inference time: {sum(times)/len(times):.4f}s")
print(f"Min/Max: {min(times):.4f}s / {max(times):.4f}s")
print(f"Throughput: {1.0 / (sum(times)/len(times)):.2f} images/second")
```

## Next Steps

1. **Establish Baseline:** Profile current setup
2. **Identify Bottlenecks:** Find slowest operations
3. **Optimize:** Implement optimization strategies
4. **Verify Improvements:** Re-profile and compare
5. **Document:** Record results and configuration

See `TRAINING.md` and `INFERENCE.md` for specific guidance.
