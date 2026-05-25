# Quick Reference: Enhanced SAM3 Training

## Quick Start (5 minutes)

### 1. Analyze Class Imbalance
```python
from sam3.train.utils.class_imbalance import ClassImbalanceAnalyzer

analyzer = ClassImbalanceAnalyzer("coco.json")
stats = analyzer.get_statistics_summary()
print(f"Imbalance ratio: {stats['imbalance_ratio']}")
```

### 2. Enable Enhanced Training
```yaml
# In your config.yaml
enhanced_training:
  class_imbalance:
    enabled: true
  per_iteration_validation:
    enabled: true
  visualization:
    enabled: true
  enhanced_logging:
    enabled: true
```

### 3. Integrate with Trainer
```python
from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

enhancer = create_trainer_enhancer(trainer, cfg.enhanced_training)

# In training loop
enhancer.validate_per_iteration(iteration)
enhancer.update_metrics({"loss": loss.item()}, iteration)
```

## Command Reference

### Analyze Dataset
```bash
python examples/enhanced_training_example.py --coco-json data/coco.json --analyze-only
```

### View Configuration Options
```bash
cat configs/enhanced_training_template.yaml
```

## Module Reference

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `class_imbalance.py` | Analyze class distribution | `ClassImbalanceAnalyzer` |
| `visualization.py` | Visualize predictions | `SegmentationVisualizer` |
| `per_iteration_validation.py` | Validate during training | `PerIterationValidator` |
| `enhanced_config.py` | Configuration classes | `EnhancedTrainerConfig` |
| `trainer_enhancer.py` | Integration layer | `TrainerEnhancer` |

## Common Tasks

### Get Class Weights
```python
weights = analyzer.get_class_weights(strategy="balanced")
# Use in loss: criterion = torch.nn.CrossEntropyLoss(weight=weights)
```

### Visualize Predictions
```python
visualizer = SegmentationVisualizer(image_size=(1024, 1024))
vis_img = visualizer.visualize_instance_masks(
    image, masks, class_ids, class_names=class_dict
)
```

### Track Class-Wise Metrics
```python
tracker = ClassWiseMetricsTracker(num_classes=10)
tracker.update(class_id=2, loss=0.5)
stats = tracker.get_class_wise_stats()
```

### Aggregate Metrics
```python
aggregator = IterationMetricsAggregator(log_frequency=50)
should_log = aggregator.update({"loss": 0.5}, iteration)
if should_log:
    agg_metrics = aggregator.get_aggregated_metrics()
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High memory usage | Reduce `num_batches` in validation config |
| Slow training | Increase `frequency` in validation/visualization |
| Imports failing | Ensure torch, numpy, matplotlib are installed |
| Class weights not applied | Check loss function supports weighting parameter |
| No visualizations saved | Check `save_dir` exists and is writable |

## Configuration Snippets

### Minimal (Class Imbalance Only)
```yaml
enhanced_training:
  class_imbalance:
    enabled: true
    analyze_statistics: true
```

### Monitoring (Per-Iteration Validation)
```yaml
enhanced_training:
  per_iteration_validation:
    enabled: true
    frequency: 50
```

### Visualization
```yaml
enhanced_training:
  visualization:
    enabled: true
    frequency: 100
    num_samples: 4
    save_dir: "./viz"
```

### Full Features
```yaml
enhanced_training:
  class_imbalance:
    enabled: true
  per_iteration_validation:
    enabled: true
  visualization:
    enabled: true
  enhanced_logging:
    enabled: true
```

## API Quick Reference

### ClassImbalanceAnalyzer
```python
analyzer = ClassImbalanceAnalyzer(json_path)
analyzer.get_class_frequencies()          # Dict[class_id, frequency]
analyzer.get_class_weights(strategy)      # Dict[class_id, weight]
analyzer.get_imbalance_ratio()            # float
analyzer.get_statistics_summary()         # Dict with full stats
analyzer.get_split_statistics()           # Dict[split, stats]
```

### SegmentationVisualizer
```python
viz = SegmentationVisualizer(image_size=(h, w))
viz.visualize_instance_masks(image, masks, class_ids, ...)
viz.visualize_bboxes(image, bboxes, class_ids, ...)
viz.visualize_combined(image, gt_masks, pred_masks, ...)
viz.create_grid_visualization(images, masks, class_ids, ...)
```

### PerIterationValidator
```python
validator = PerIterationValidator(val_loader, model, loss_fn, device)
validator.should_validate(iteration)      # bool
validator.validate(iteration)             # Dict[metric_name, value]
```

### IterationMetricsAggregator
```python
agg = IterationMetricsAggregator(log_frequency=50)
agg.update(metrics, iteration)            # bool (should_log)
agg.get_aggregated_metrics()              # Dict with stats
agg.reset()
```

### ClassWiseMetricsTracker
```python
tracker = ClassWiseMetricsTracker(num_classes)
tracker.update(class_id, loss, weight=1.0)
tracker.get_class_wise_stats()            # Dict[class_id, stats]
tracker.reset()
```

### TrainerEnhancer
```python
enhancer = TrainerEnhancer(trainer, **configs)
enhancer.validate_per_iteration(iteration)
enhancer.update_metrics(metrics, iteration)
enhancer.update_class_metrics(class_id, loss, weight)
enhancer.visualize_predictions(images, preds, gt, iteration)
enhancer.get_class_weights()              # Dict or None
```

## File Structure
```
sam3/train/utils/
├── class_imbalance.py          # Class imbalance analysis
├── visualization.py            # Visualization utilities
├── per_iteration_validation.py # Validation during training
├── enhanced_config.py          # Configuration dataclasses
└── trainer_enhancer.py         # Integration layer

configs/
└── enhanced_training_template.yaml  # Configuration template

examples/
└── enhanced_training_example.py     # Working example

Documentation:
├── ENHANCED_TRAINING_GUIDE.md       # Comprehensive guide
├── INTEGRATION_GUIDE.md              # Integration instructions
└── CHANGES_SUMMARY.md               # Complete change list
```

## Common Patterns

### Setup and Enhance Trainer
```python
trainer = Trainer(config)
enhancer = create_trainer_enhancer(trainer, config.enhanced_training)
```

### Training Loop Integration
```python
for epoch in range(epochs):
    for iter, batch in enumerate(loader):
        loss = train_step(batch)
        iteration = epoch * len(loader) + iter
        
        # Add enhancements
        enhancer.validate_per_iteration(iteration)
        enhancer.update_metrics({"loss": loss.item()}, iteration)
```

### Class Imbalance Analysis
```python
analyzer = ClassImbalanceAnalyzer("coco.json")
weights = analyzer.get_class_weights()
for class_id, weight in weights.items():
    print(f"Class {class_id}: weight={weight}")
```

### Visualization Generation
```python
viz = SegmentationVisualizer((1024, 1024))
for batch in val_loader:
    images = batch['images']
    for i, img in enumerate(images):
        result = model(img)
        vis_img = viz.visualize_instance_masks(
            img, result['masks'], result['class_ids']
        )
        save_image(vis_img, f"output_{i}.png")
```

## Performance Tips

1. **Disable unused features** in config to avoid overhead
2. **Increase validation frequency** if validation is slow
3. **Use fewer validation batches** for faster iteration
4. **Run visualization less frequently** for faster training
5. **Enable class weights** for more effective training on imbalanced data

## Getting Help

- Read `ENHANCED_TRAINING_GUIDE.md` for detailed documentation
- Check `examples/enhanced_training_example.py` for working code
- Review `INTEGRATION_GUIDE.md` for integration help
- See `configs/enhanced_training_template.yaml` for all options
