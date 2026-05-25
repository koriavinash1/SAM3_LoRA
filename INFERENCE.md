# Inference Guide for SAM3 LoRA

This guide provides comprehensive instructions for running inference with LoRA-adapted SAM3 models using the enhanced inference script.

## Table of Contents

1. [Overview](#overview)
2. [Inference Script](#inference-script)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Output Formats](#output-formats)
6. [Configuration](#configuration)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Overview

The enhanced inference script (`inference_enhanced.py`) provides:

- **Comprehensive Logging:** Detailed logs of inference steps
- **Profiling:** Timing and memory measurements
- **Multiple Prompts:** Support for multiple text prompts per image
- **Flexible Output:** Visualization and JSON results
- **GPU/CPU Support:** Automatic device detection
- **Batch Processing:** Handle multiple images efficiently

## Inference Script

### Features

1. **Model Loading:** Load base SAM3 model with LoRA weights
2. **Image Preprocessing:** Automatic image loading and processing
3. **Inference:** Run segmentation with text prompts
4. **Profiling:** Track inference timing and memory
5. **Visualization:** Create visualization with segmentation results
6. **Reporting:** Generate JSON results and profiling reports

### Command Line Arguments

```bash
python inference_enhanced.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--weights` | Path to LoRA weights file |
| `--image` | Path to input image |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `facebook/sam3-base` | Base SAM3 model |
| `--config` | - | Path to LoRA config YAML |
| `--prompt` | `["object"]` | Text prompt(s) |
| `--output` | `output.png` | Output visualization path |
| `--results-json` | - | Save results as JSON |
| `--threshold` | `0.5` | Confidence threshold |
| `--cpu` | - | Use CPU instead of GPU |
| `--profile-report` | - | Save profiling report |
| `--log-dir` | `logs/inference` | Log directory |
| `--debug` | - | Enable debug logging |

## Basic Usage

### Single Prompt Inference

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --output output.png
```

### Multiple Prompts

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" "defect" "damage" \
    --output output.png
```

### With Configuration File

```bash
python inference_enhanced.py \
    --config configs/lora_config.yaml \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack"
```

## Advanced Usage

### With Profiling Report

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --profile-report inference_profile.json \
    --results-json inference_results.json
```

### With Custom Threshold

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --threshold 0.3  # More detections
```

### CPU Inference (for testing)

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --cpu
```

### With Debug Logging

```bash
python inference_enhanced.py \
    --weights path/to/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --debug \
    --log-dir logs/debug_inference
```

### Batch Inference (Multiple Images)

Create a script to process multiple images:

```bash
#!/bin/bash

weights="path/to/best_lora_weights.pt"
output_dir="results/"

for image in data/test_images/*.jpg; do
    filename=$(basename "$image" .jpg)
    python inference_enhanced.py \
        --weights "$weights" \
        --image "$image" \
        --prompt "crack" \
        --output "$output_dir/${filename}_output.png" \
        --results-json "$output_dir/${filename}_results.json"
done
```

## Output Formats

### Visualization Output

The inference script generates:

1. **PNG Visualization:**
   - Input image with detected objects
   - Segmentation masks (semi-transparent overlay)
   - Bounding boxes with confidence scores
   - Text summary of results

Example output structure:
```
output.png
├── Input image
├── Detected objects with colors
├── Mask overlays
└── Text summary (bottom left)
```

### JSON Results

When using `--results-json`, results are saved in JSON format:

```json
{
  "prompts": {
    "crack": {
      "n_detections": 3,
      "max_confidence": 0.75,
      "masks_shape": [1, 3, 1008, 1008]
    },
    "damage": {
      "n_detections": 1,
      "max_confidence": 0.62,
      "masks_shape": [1, 1, 1008, 1008]
    }
  },
  "metadata": {
    "device": "cuda",
    "threshold": 0.5,
    "image_size": [1024, 768]
  }
}
```

### Profiling Report

When using `--profile-report`, timing and memory statistics are saved:

```json
{
  "name": "full_inference",
  "timers": {
    "load_base_model": {
      "count": 1,
      "total_time": 5.23,
      "mean_time": 5.23,
      "std_time": 0.0,
      "min_time": 5.23,
      "max_time": 5.23
    },
    "full_inference": {
      "count": 1,
      "total_time": 2.45,
      "mean_time": 2.45,
      ...
    }
  },
  "memory": {
    "before_inference": {
      "peak_memory_mb": 2048.5,
      "current_memory_mb": 2048.5,
      "allocated_memory_mb": 2100.0
    },
    "after_inference": {
      "peak_memory_mb": 2048.5,
      "current_memory_mb": 512.3,
      "allocated_memory_mb": 2100.0
    }
  }
}
```

## Configuration

### Model Configuration

Specify SAM3 model variant:

```bash
# Use SAM3 Large
python inference_enhanced.py \
    --model facebook/sam3-large \
    --weights path/to/weights.pt \
    --image image.jpg

# Use SAM3 Huge
python inference_enhanced.py \
    --model facebook/sam3-huge \
    --weights path/to/weights.pt \
    --image image.jpg
```

### LoRA Configuration File

If using `--config`, the YAML file structure:

```yaml
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "qkv"
    - "proj"
```

## Performance Optimization

### Inference Speed

Optimize inference speed:

1. **Use GPU:** Ensure CUDA is available
   ```bash
   python inference_enhanced.py ... --weights weights.pt
   ```

2. **Batch Multiple Images:** Process multiple images together
3. **Use Smaller Models:** Consider `facebook/sam3-base` over larger variants

### Memory Optimization

Reduce memory usage:

1. **Use CPU for Inference:** Good for development/testing
   ```bash
   python inference_enhanced.py ... --cpu
   ```

2. **Process Images Sequentially:** Don't hold multiple images in memory

3. **Reduce Image Resolution:** If not critical for accuracy

### GPU Memory Profile

Typical memory usage (for SAM3-base):

| Step | Memory (MB) |
|------|------------|
| Model Loading | 1500-2000 |
| Image Preprocessing | 50-100 |
| Inference (single image) | 2000-2500 |
| Visualization | 100-200 |

## Real-world Examples

### Crack Detection

```bash
python inference_enhanced.py \
    --weights checkpoints/crack_detector_weights.pt \
    --image data/concrete_sample.jpg \
    --prompt "crack" "fracture" \
    --threshold 0.4 \
    --output results/crack_detection.png
```

### Multi-class Defect Detection

```bash
python inference_enhanced.py \
    --weights checkpoints/defect_detector_weights.pt \
    --image data/product_image.jpg \
    --prompt "scratch" "dent" "discoloration" "deformation" \
    --threshold 0.5 \
    --profile-report results/profile.json \
    --results-json results/detections.json
```

### Medical Image Segmentation

```bash
python inference_enhanced.py \
    --weights checkpoints/medical_seg_weights.pt \
    --image data/xray.jpg \
    --prompt "tumor" "lesion" \
    --threshold 0.6 \
    --output results/segmentation.png
```

## Logging Output

The inference script produces detailed logs:

```
============================================================
SAM3 LoRA Enhanced Inference
============================================================
Using device: cuda
CUDA device: NVIDIA A100
CUDA capability: (8, 0)

--- Loading Model ---
Loading base model: facebook/sam3-base
✓ Base model loaded
Loading LoRA config from: configs/lora_config.yaml
Applying LoRA to model...
✓ LoRA applied to model
Loading LoRA weights from: checkpoints/best_weights.pt
✓ LoRA weights loaded
Moving model to device: cuda
✓ Model on device
Model parameters: 95,000,000 (trainable: 50,000)

--- Preprocessing ---
Loading image from: data/test_image.jpg
  Image size: (1024, 768)
Preprocessing image...
  Image shape for model: (1024, 768)

--- Running Inference ---
Running inference with 2 prompt(s)
  Prompts: ['crack', 'damage']
  Threshold: 0.5
Processing prompt 1/2: 'crack'
  ✓ Found 2 detections (max confidence: 0.75)
Processing prompt 2/2: 'damage'
  ✓ Found 1 detections (max confidence: 0.62)

--- Visualization ---
Creating visualization...
✓ Visualization saved to output.png
✓ Results saved to results.json

============================================================
Profiling Summary: full_inference
============================================================

⏱️  Timing Statistics:
Measurement                Count    Mean        Std         Min         Max
------
load_base_model           1        5.2340      0.0000      5.2340      5.2340
apply_lora                1        0.1234      0.0000      0.1234      0.1234
load_lora_weights         1        0.0456      0.0000      0.0456      0.0456
full_inference            1        2.4567      0.0000      2.4567      2.4567

💾 Memory Statistics:

before_inference:
  Peak Memory: 2048.50 MB
  Current Memory: 2048.50 MB
  Allocated Memory: 2100.00 MB

============================================================
Inference completed successfully
============================================================
```

## Troubleshooting

### Issue: "Model loading is slow"

**Solutions:**
1. First load caches the model - subsequent runs are faster
2. Use smaller model variant (base instead of huge)
3. Check disk I/O with `--debug` logging

### Issue: "No detections found"

**Solutions:**
1. Lower threshold:
   ```bash
   --threshold 0.3
   ```

2. Try different prompts:
   ```bash
   --prompt "crack" "fracture" "breakage"
   ```

3. Check image quality and relevance to training data

4. Verify LoRA weights are correct:
   ```bash
   ls -lh path/to/weights.pt
   ```

### Issue: "CUDA out of memory"

**Solutions:**
1. Use CPU:
   ```bash
   --cpu
   ```

2. Process smaller images
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: "Output image is blank"

**Solutions:**
1. Check input image path exists
2. Verify image is readable:
   ```python
   from PIL import Image
   img = Image.open("path/to/image.jpg")
   img.show()
   ```

3. Check visualization code in `inference_enhanced.py`

### Issue: "JSON results are incomplete"

**Solutions:**
1. Check model outputs with debug logging:
   ```bash
   --debug
   ```

2. Verify LoRA weights are properly loaded
3. Check for errors in model forward pass

## Advanced Customization

### Custom Visualization

Modify `visualize_results()` function in `inference_enhanced.py`:

```python
def visualize_results_custom(image, results, output_path):
    # Your custom visualization code
    pass
```

### Custom Result Processing

Extend result processing in `run_inference()`:

```python
# Add post-processing logic
if masks is not None:
    # Apply morphological operations, clustering, etc.
    pass
```

## Next Steps

1. **Batch Processing:** Create batch inference script
2. **API Deployment:** Wrap in Flask/FastAPI
3. **Model Optimization:** Export for faster inference
4. **Performance Analysis:** Use profiling reports

See `PROFILING.md` for detailed profiling information.
