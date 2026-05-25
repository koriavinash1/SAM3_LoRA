# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Configuration classes for enhanced training with class imbalance and per-iteration validation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassImbalanceConfig:
    """Configuration for handling class imbalance."""
    
    enabled: bool = False
    """Enable class imbalance handling"""
    
    strategy: str = "weighted_sampling"
    """Strategy for handling imbalance: 'weighted_sampling', 'focal_loss', 'class_weighting'"""
    
    weight_power: float = 1.0
    """Power for computing weights (higher = stronger imbalance correction)"""
    
    analyze_statistics: bool = True
    """Analyze and log class distribution statistics"""


@dataclass
class PerIterationValidationConfig:
    """Configuration for per-iteration validation during training."""
    
    enabled: bool = False
    """Enable validation after every N iterations"""
    
    frequency: int = 50
    """Run validation every N iterations"""
    
    num_batches: int = 5
    """Number of validation batches to use per validation run"""
    
    log_metrics: bool = True
    """Log validation metrics"""


@dataclass
class VisualizationConfig:
    """Configuration for visualization during training."""
    
    enabled: bool = False
    """Enable visualization"""
    
    frequency: int = 100
    """Generate visualization every N iterations"""
    
    num_samples: int = 4
    """Number of samples to visualize per batch"""
    
    save_dir: Optional[str] = None
    """Directory to save visualization images"""
    
    overlay_alpha: float = 0.6
    """Alpha blending factor for mask overlay"""
    
    draw_bboxes: bool = True
    """Draw bounding boxes on visualization"""
    
    draw_centroids: bool = True
    """Draw centroids on visualization"""


@dataclass
class EnhancedLoggingConfig:
    """Configuration for enhanced logging with aggregation."""
    
    log_every_n_iterations: int = 50
    """Aggregate and log metrics every N iterations instead of per-image"""
    
    class_wise_metrics: bool = True
    """Track and log metrics separately per class"""
    
    aggregation_strategy: str = "mean"
    """How to aggregate metrics: 'mean', 'median', 'sum'"""


@dataclass
class EnhancedTrainerConfig:
    """Enhanced trainer configuration with all new features."""
    
    class_imbalance: ClassImbalanceConfig = field(default_factory=ClassImbalanceConfig)
    """Class imbalance handling config"""
    
    per_iteration_validation: PerIterationValidationConfig = field(
        default_factory=PerIterationValidationConfig
    )
    """Per-iteration validation config"""
    
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    """Visualization config"""
    
    enhanced_logging: EnhancedLoggingConfig = field(default_factory=EnhancedLoggingConfig)
    """Enhanced logging config"""
    
    split_aware_loading: bool = False
    """Enable split-aware data loading (train/val/test)"""
    
    preserve_attributes: bool = True
    """Preserve annotation attributes and metadata from new COCO schema"""
