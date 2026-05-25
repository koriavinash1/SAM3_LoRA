# Integration Guide: Adding Enhanced Training to Existing Trainer

This guide explains how to integrate the new enhanced training features into your existing SAM3 trainer with minimal changes.

## Quick Start

### 1. Add Enhanced Configuration to Your Config File

In your training YAML config, add:

```yaml
enhanced_training:
  class_imbalance:
    enabled: true
    strategy: "weighted_sampling"
    analyze_statistics: true
  
  per_iteration_validation:
    enabled: true
    frequency: 50
    num_batches: 5
  
  visualization:
    enabled: true
    frequency: 100
    save_dir: "./logs/visualizations"
  
  enhanced_logging:
    enabled: true
    log_every_n_iterations: 50
    class_wise_metrics: true
```

### 2. Instantiate Trainer Enhancer in Your Training Script

In your main training script (e.g., `train.py` or `train.py`):

```python
from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

# After creating your trainer
trainer = Trainer(cfg)

# Create enhancer
enhancer = create_trainer_enhancer(
    trainer=trainer,
    enhanced_config=cfg.get("enhanced_training", {})
)
```

### 3. Update Your Training Loop

Modify your training loop to use enhancer features:

```python
# In train_epoch or wherever you do per-iteration operations

for data_iter, batch in enumerate(train_loader):
    # Your existing training code
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # Add enhanced features
    iteration = epoch * len(train_loader) + data_iter
    
    # Per-iteration validation
    val_metrics = enhancer.validate_per_iteration(iteration)
    
    # Update aggregated metrics
    if data_iter % 10 == 0:  # Every 10 iterations
        metrics = {"train_loss": loss.item()}
        enhancer.update_metrics(metrics, iteration)
        
        # Track per-class metrics
        for class_id in range(num_classes):
            enhancer.update_class_metrics(class_id, loss.item())
    
    # Visualize predictions
    enhancer.visualize_predictions(
        batch.images, 
        model_output, 
        batch.targets,
        iteration
    )
```

## Integration Points

### Class Imbalance Handling

**Before:**
```python
# Standard COCO loading
loader = COCO_FROM_JSON(annotation_file)
```

**After:**
```python
# With class imbalance support
loader = COCO_FROM_JSON(
    annotation_file,
    target_split="train",  # Filter to train split
    class_imbalance_strategy="weighted_sampling"
)

# Get class weights for loss
class_weights = loader.getClassWeights()

# Use in your loss function (if supported)
# weighted_loss = criterion(output, target, weight=class_weights)
```

### Loss Function Integration

If your loss function supports class weighting:

```python
# Get weights from enhancer
weights = enhancer.get_class_weights()

if weights:
    # Convert to tensor
    from sam3.train.utils.class_imbalance import create_class_weights_tensor
    weight_tensor = create_class_weights_tensor(
        weights, 
        num_classes=len(weights),
        device=device
    )
    
    # Use in loss (example for CrossEntropyLoss)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
```

### Per-Iteration Validation

**Minimal Change:**
```python
# In training loop, add one line
val_metrics = enhancer.validate_per_iteration(iteration)
```

**With Logging:**
```python
val_metrics = enhancer.validate_per_iteration(iteration)
if val_metrics:
    logger.log_dict(val_metrics, iteration)
```

### Visualization

**Minimal Change:**
```python
# In training loop
enhancer.visualize_predictions(
    images=batch.images,
    predictions=model_output,
    ground_truth=batch.targets,
    iteration=iteration
)
```

### Enhanced Logging

**Replace per-image logging:**

Before:
```python
# Logging every image
for i, image_id in enumerate(image_ids):
    logger.log(f"loss/image_{image_id}", loss[i], step)
```

After:
```python
# Aggregate metrics
metrics = {f"loss_{i}": loss[i].item() for i in range(len(loss))}
enhancer.update_metrics(metrics, iteration)
# Automatic logging every N iterations
```

## Minimal Integration Example

For the absolute minimum integration, add these 3 changes:

```python
# 1. Import at top of file
from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

# 2. After trainer initialization
enhancer = create_trainer_enhancer(trainer, cfg.get("enhanced_training", {}))

# 3. In training loop
val_metrics = enhancer.validate_per_iteration(iteration)
```

## Feature-by-Feature Integration

### Just Class Imbalance Analysis

```python
from sam3.train.utils.class_imbalance import ClassImbalanceAnalyzer

# Analyze once at start
analyzer = ClassImbalanceAnalyzer("coco.json")
stats = analyzer.get_statistics_summary()
weights = analyzer.get_class_weights(strategy="balanced")

# Use weights in loss function
```

### Just Per-Iteration Validation

```python
from sam3.train.utils.per_iteration_validation import PerIterationValidator

validator = PerIterationValidator(
    val_loader=val_loader,
    model=model,
    loss_fn=loss_fn,
    device=device,
    frequency=50
)

# In training loop
metrics = validator.validate(iteration)
```

### Just Visualization

```python
from sam3.train.utils.visualization import SegmentationVisualizer

visualizer = SegmentationVisualizer(image_size=(1024, 1024))

# Visualize predictions
vis_img = visualizer.visualize_instance_masks(
    image=image_array,
    masks=pred_masks,
    class_ids=class_ids
)
```

### Just Improved Logging

```python
from sam3.train.utils.per_iteration_validation import IterationMetricsAggregator

aggregator = IterationMetricsAggregator(log_frequency=50)

# In training loop
metrics = {"loss": loss.item()}
should_log = aggregator.update(metrics, iteration)
if should_log:
    agg_metrics = aggregator.get_aggregated_metrics()
    logger.log_dict(agg_metrics, iteration)
```

## Debugging Integration Issues

### Module Not Found

If you get `ModuleNotFoundError` when importing:

```python
# Make sure sam3 is in your Python path
import sys
sys.path.insert(0, '/path/to/SAM3_LoRA')
```

### Attribute Error on Trainer

If enhancer complains about missing trainer attributes:

```python
# Ensure trainer has these attributes:
# - model
# - device
# - logger (optional)
# - logging_conf.log_dir (optional)
# - val_dataset (optional)
```

### Class Weights Not Applied

If class imbalance doesn't seem to work:

1. Verify `enabled: true` in config
2. Check that loss function supports weights
3. Verify weights are passed to loss function:
   ```python
   weights = enhancer.get_class_weights()
   if weights is not None:
       print(f"Class weights: {weights}")
   ```

### Memory Issues

If training is slow or OOM:

1. Reduce `frequency` in visualization/validation config
2. Reduce `num_batches` in per-iteration validation
3. Disable features you don't need

## Complete Integration Example

See `examples/enhanced_training_example.py` for a complete working example.

## Configuration Reference

See `configs/enhanced_training_template.yaml` for all available configuration options.

## Backward Compatibility

All enhancements are fully backward compatible:
- Standard COCO format still works (new fields optional)
- Features can be disabled via config
- No changes to existing trainer interface
- No breaking changes to data loading

## Support and Troubleshooting

For issues or questions:

1. Check `ENHANCED_TRAINING_GUIDE.md` for detailed documentation
2. Review example scripts in `examples/`
3. Check configuration template in `configs/enhanced_training_template.yaml`
4. Enable logging to see detailed diagnostic messages:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
