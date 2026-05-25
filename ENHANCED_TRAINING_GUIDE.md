# Enhanced SAM3 Training with Class Imbalance Handling

This document describes the new features added to support training with heavily class-imbalanced datasets using the enhanced COCO schema.

## Overview

The following enhancements have been added to the SAM3 training framework:

1. **Enhanced COCO Schema Support** - Support for the new COCO format with `attributes`, `metadata`, and `split` fields
2. **Class Imbalance Handling** - Tools and utilities for detecting and handling class imbalanced datasets
3. **Per-Iteration Validation** - Run validation checks during training to monitor convergence
4. **Visualization with Class Labels** - Generate visualizations showing predictions, ground truth, and class information
5. **Improved Logging** - Aggregate metrics over iterations instead of logging per-image

## New Modules

### 1. Class Imbalance Utilities (`sam3/train/utils/class_imbalance.py`)

Provides tools for analyzing and handling class imbalance:

**Key Classes:**
- `ClassImbalanceAnalyzer`: Analyzes dataset class distribution
  - Computes class frequencies and imbalance ratios
  - Generates class weights for different strategies
  - Provides split-aware statistics (train/val/test)

**Usage:**

```python
from sam3.train.utils.class_imbalance import ClassImbalanceAnalyzer

# Analyze dataset
analyzer = ClassImbalanceAnalyzer("path/to/coco.json")

# Get statistics
stats = analyzer.get_statistics_summary()
print(f"Imbalance ratio: {stats['imbalance_ratio']}")

# Get class weights for loss
weights = analyzer.get_class_weights(strategy="balanced")

# Get split statistics
split_stats = analyzer.get_split_statistics()
```

### 2. Visualization Utilities (`sam3/train/utils/visualization.py`)

Provides segmentation visualization capabilities:

**Key Classes:**
- `SegmentationVisualizer`: Creates visualizations with masks, bboxes, and labels
  - `visualize_instance_masks()`: Overlay instance masks on image
  - `visualize_bboxes()`: Draw bounding boxes with class labels
  - `visualize_combined()`: Create comprehensive visualization with GT and predictions
  - `create_grid_visualization()`: Create grid of multiple samples

**Usage:**

```python
from sam3.train.utils.visualization import SegmentationVisualizer

visualizer = SegmentationVisualizer(image_size=(1024, 1024))

# Visualize predictions
vis_image = visualizer.visualize_instance_masks(
    image=image_array,
    masks=predicted_masks,
    class_ids=class_ids,
    class_names=class_name_dict,
    scores=confidence_scores
)

# Create combined visualization
fig = visualizer.visualize_combined(
    image=image_array,
    gt_masks=gt_masks,
    gt_class_ids=gt_class_ids,
    pred_masks=pred_masks,
    pred_class_ids=pred_class_ids,
    class_names=class_name_dict
)
```

### 3. Per-Iteration Validation (`sam3/train/utils/per_iteration_validation.py`)

Enables validation during training:

**Key Classes:**
- `PerIterationValidator`: Runs validation every N iterations
- `IterationMetricsAggregator`: Aggregates metrics over multiple iterations
- `ClassWiseMetricsTracker`: Tracks metrics separately per class

**Usage:**

```python
from sam3.train.utils.per_iteration_validation import (
    PerIterationValidator,
    IterationMetricsAggregator
)

# Setup per-iteration validation
validator = PerIterationValidator(
    val_loader=val_loader,
    model=model,
    loss_fn=loss_fn,
    device=device,
    frequency=50,  # Validate every 50 iterations
    num_batches=5  # Use 5 batches per validation
)

# Setup metrics aggregation
aggregator = IterationMetricsAggregator(log_frequency=50)

# During training loop:
metrics = {"loss": loss_value, "accuracy": acc_value}
should_log = aggregator.update(metrics, current_iteration)
if should_log:
    aggregated_metrics = aggregator.get_aggregated_metrics()
    # Log aggregated_metrics
```

### 4. Enhanced COCO Schema Support

The COCO JSON loader now supports the extended schema:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image.jpg",
      "width": 1024,
      "height": 768,
      "split": "train",
      "seq_id": "sequence_001",
      "frame_id": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 5,
      "bbox": [10, 20, 100, 80],
      "area": 8000,
      "segmentation": {...},
      "iscrowd": 0,
      "attributes": {
        "label_l1": "category",
        "label_l2": "subcategory",
        "bbox_xyxy": [10, 20, 110, 100],
        "centroid": [60, 60]
      },
      "metadata": {
        "col1": "value1",
        "col2": "value2"
      }
    }
  ]
}
```

**Updated `COCO_FROM_JSON` class features:**

```python
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON

# Load with split filtering
loader = COCO_FROM_JSON(
    annotation_file="coco.json",
    target_split="train",  # Filter to train split only
    class_imbalance_strategy="weighted_sampling",
    include_negatives=True
)

# Get class weights
weights = loader.getClassWeights()

# Access preserved attributes
datapoints = loader.loadQueriesAndAnnotationsFromDatapoint(idx)
# Annotations now include 'attributes' and 'metadata' fields
```

### 5. Enhanced Configuration (`sam3/train/utils/enhanced_config.py`)

Configuration classes for new features:

```python
from sam3.train.utils.enhanced_config import (
    ClassImbalanceConfig,
    PerIterationValidationConfig,
    VisualizationConfig,
    EnhancedLoggingConfig
)

# Configure class imbalance handling
imbalance_cfg = ClassImbalanceConfig(
    enabled=True,
    strategy="weighted_sampling",
    weight_power=1.0,
    analyze_statistics=True
)

# Configure per-iteration validation
val_cfg = PerIterationValidationConfig(
    enabled=True,
    frequency=50,
    num_batches=5
)

# Configure visualization
vis_cfg = VisualizationConfig(
    enabled=True,
    frequency=100,
    num_samples=4,
    save_dir="./visualizations"
)

# Configure enhanced logging
log_cfg = EnhancedLoggingConfig(
    log_every_n_iterations=50,
    class_wise_metrics=True
)
```

### 6. Trainer Integration (`sam3/train/utils/trainer_enhancer.py`)

Helper class to integrate all enhancements with existing trainer:

```python
from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

# Create enhancer
enhancer = create_trainer_enhancer(
    trainer=trainer,
    enhanced_config={
        "class_imbalance": {
            "enabled": True,
            "strategy": "weighted_sampling",
            "annotation_file": "coco.json"
        },
        "per_iteration_validation": {
            "enabled": True,
            "frequency": 50,
            "num_batches": 5
        },
        "visualization": {
            "enabled": True,
            "frequency": 100,
            "save_dir": "./visualizations"
        },
        "enhanced_logging": {
            "enabled": True,
            "log_every_n_iterations": 50,
            "class_wise_metrics": True
        }
    }
)

# During training
# Validate per-iteration
metrics = enhancer.validate_per_iteration(iteration)

# Update metrics with aggregation
enhancer.update_metrics({"loss": loss_value}, iteration)

# Track per-class metrics
enhancer.update_class_metrics(class_id=2, loss=loss_value)

# Get class weights for loss computation
weights = enhancer.get_class_weights()
```

## Configuration in YAML

Add the following to your training config YAML:

```yaml
enhanced_training:
  class_imbalance:
    enabled: true
    strategy: "weighted_sampling"  # or "focal_loss", "class_weighting"
    weight_power: 1.0
    analyze_statistics: true

  per_iteration_validation:
    enabled: true
    frequency: 50
    num_batches: 5
    log_metrics: true

  visualization:
    enabled: true
    frequency: 100
    num_samples: 4
    save_dir: "./logs/visualizations"
    overlay_alpha: 0.6
    draw_bboxes: true
    draw_centroids: true

  enhanced_logging:
    log_every_n_iterations: 50
    class_wise_metrics: true
    aggregation_strategy: "mean"

  split_aware_loading: true
  preserve_attributes: true
```

## Best Practices for Class Imbalanced Datasets

1. **Enable Statistics Analysis**: Always enable `analyze_statistics=True` to understand your data distribution

2. **Choose Appropriate Weighting Strategy**:
   - `inverse_frequency`: Simple 1/frequency weighting
   - `balanced`: Weights instances equally per class
   - `effective_number`: Handles extreme imbalance better

3. **Use Per-Iteration Validation**: Monitor convergence and catch overfitting early
   - Set frequency to 5-10% of total iterations
   - Use small num_batches (3-5) to avoid overhead

4. **Monitor Class-Wise Metrics**: Track losses separately per class to identify underperforming classes

5. **Visualize Predictions**: Regularly inspect model predictions, especially for minority classes

## Example Training Script

```python
import torch
from sam3.train.trainer import Trainer
from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

# Create trainer (existing code)
trainer = Trainer(config)

# Enhance trainer with new features
enhancer = create_trainer_enhancer(
    trainer=trainer,
    enhanced_config=config.get("enhanced_training", {})
)

# Training loop with enhancements
for epoch in range(num_epochs):
    for iteration, batch in enumerate(train_loader):
        # Forward pass
        loss = model(batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        current_iteration = epoch * len(train_loader) + iteration
        
        # Per-iteration validation
        val_metrics = enhancer.validate_per_iteration(current_iteration)
        
        # Update aggregated metrics
        enhancer.update_metrics({"loss": loss.item()}, current_iteration)
        
        # Visualize predictions periodically
        enhancer.visualize_predictions(batch.images, predictions, targets, current_iteration)
```

## Backward Compatibility

All enhancements are optional and backward compatible:
- Standard COCO format still works (new fields are optional)
- Features can be disabled individually
- No changes to existing trainer interface

## Troubleshooting

### High Memory Usage During Validation
- Reduce `num_batches` in per-iteration validation config
- Increase validation `frequency` to run less often

### Slow Training Due to Visualization
- Reduce `frequency` in visualization config
- Reduce `num_samples` for visualization

### Class Imbalance Not Being Addressed
- Verify `enabled: true` in config
- Check that class weights are properly passed to loss function
- Analyze statistics with `ClassImbalanceAnalyzer` to verify detection

## References

For more information on handling class imbalance in deep learning:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection"
- Class Weighting: Effective Techniques in Class Imbalanced Learning
- COCO Format: https://cocodataset.org/#format-data
