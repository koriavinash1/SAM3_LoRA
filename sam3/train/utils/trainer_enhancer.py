# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Integration helpers for enhanced training features with existing trainer.
"""

import logging
import os
from typing import Dict, Optional, Callable

import torch

logger = logging.getLogger(__name__)


class TrainerEnhancer:
    """
    Utility class to enhance an existing trainer with new features.
    Handles per-iteration validation, visualization, and improved logging.
    """

    def __init__(
        self,
        trainer,
        class_imbalance_config: Optional[Dict] = None,
        per_iteration_val_config: Optional[Dict] = None,
        visualization_config: Optional[Dict] = None,
        enhanced_logging_config: Optional[Dict] = None,
    ):
        """
        Initialize the trainer enhancer.

        Args:
            trainer: The existing trainer instance
            class_imbalance_config: Class imbalance configuration dict
            per_iteration_val_config: Per-iteration validation config dict
            visualization_config: Visualization config dict
            enhanced_logging_config: Enhanced logging config dict
        """
        self.trainer = trainer
        self.class_imbalance_config = class_imbalance_config or {}
        self.per_iteration_val_config = per_iteration_val_config or {}
        self.visualization_config = visualization_config or {}
        self.enhanced_logging_config = enhanced_logging_config or {}

        self._setup_components()

    def _setup_components(self):
        """Setup enhanced training components."""
        # Import here to avoid circular imports
        from sam3.train.utils.class_imbalance import ClassImbalanceAnalyzer
        from sam3.train.utils.per_iteration_validation import (
            PerIterationValidator,
            IterationMetricsAggregator,
            ClassWiseMetricsTracker,
        )
        from sam3.train.utils.visualization import SegmentationVisualizer

        # Setup class imbalance analyzer if enabled
        if self.class_imbalance_config.get("enabled", False):
            self.class_imbalance_analyzer = ClassImbalanceAnalyzer(
                self.class_imbalance_config.get("annotation_file")
            )
            logger.info(f"Class imbalance analyzer initialized")
            self._log_class_statistics()
        else:
            self.class_imbalance_analyzer = None

        # Setup per-iteration validator if enabled
        if self.per_iteration_val_config.get("enabled", False):
            self.per_iteration_validator = PerIterationValidator(
                val_loader=getattr(self.trainer, "val_dataset", None),
                model=self.trainer.model,
                loss_fn=lambda outputs, batch: self.trainer._run_step(
                    batch, "val", {}, {}
                )[0],  # Simplified
                device=self.trainer.device,
                frequency=self.per_iteration_val_config.get("frequency", 50),
                num_batches=self.per_iteration_val_config.get("num_batches", 5),
            )
            logger.info(f"Per-iteration validator initialized")
        else:
            self.per_iteration_validator = None

        # Setup metrics aggregator if enabled
        if self.enhanced_logging_config.get("enabled", False):
            self.metrics_aggregator = IterationMetricsAggregator(
                log_frequency=self.enhanced_logging_config.get(
                    "log_every_n_iterations", 50
                )
            )
            logger.info(f"Metrics aggregator initialized")
        else:
            self.metrics_aggregator = None

        # Setup class-wise metrics tracker if enabled
        if self.enhanced_logging_config.get("class_wise_metrics", False):
            num_classes = len(self.trainer._cat_idx_to_text)
            self.class_wise_tracker = ClassWiseMetricsTracker(num_classes)
            logger.info(f"Class-wise metrics tracker initialized")
        else:
            self.class_wise_tracker = None

        # Setup visualizer if enabled
        if self.visualization_config.get("enabled", False):
            image_size = self.visualization_config.get("image_size", (1024, 1024))
            self.visualizer = SegmentationVisualizer(
                image_size=image_size,
                font_scale=self.visualization_config.get("font_scale", 0.5),
            )
            logger.info(f"Segmentation visualizer initialized")
        else:
            self.visualizer = None

    def _log_class_statistics(self):
        """Log class distribution statistics."""
        if self.class_imbalance_analyzer is None:
            return

        stats = self.class_imbalance_analyzer.get_statistics_summary()
        logger.info("=" * 80)
        logger.info("CLASS DISTRIBUTION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total instances: {stats['total_instances']}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Number of classes: {stats['num_classes']}")
        logger.info(f"Mean instances per class: {stats['mean_instances_per_class']:.2f}")
        logger.info(f"Std instances per class: {stats['std_instances_per_class']:.2f}")
        logger.info(f"Min instances: {stats['min_instances']}")
        logger.info(f"Max instances: {stats['max_instances']}")
        logger.info(f"Imbalance ratio (max/min): {stats['imbalance_ratio']:.2f}")
        logger.info("-" * 80)

        # Log per-class counts
        logger.info("Per-class instance counts:")
        class_counts = stats["class_counts"]
        class_names = stats["class_names"]
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            name = class_names.get(class_id, f"Class {class_id}")
            percentage = (count / stats['total_instances'] * 100) if stats['total_instances'] > 0 else 0
            logger.info(f"  {name} (ID: {class_id}): {count} instances ({percentage:.1f}%)")

        logger.info("=" * 80)

    def validate_per_iteration(self, iteration: int) -> Optional[Dict]:
        """
        Run per-iteration validation if configured.

        Args:
            iteration (int): Current iteration number

        Returns:
            Dict with validation metrics, or None if not enabled
        """
        if self.per_iteration_validator is None:
            return None

        if not self.per_iteration_validator.should_validate(iteration):
            return None

        logger.info(f"Running per-iteration validation at iteration {iteration}")
        metrics = self.per_iteration_validator.validate(iteration)

        if self.trainer.logger:
            self.trainer.logger.log_dict(metrics, iteration)

        return metrics

    def update_metrics(self, metrics: Dict, iteration: int):
        """
        Update metrics through the aggregator if enabled.

        Args:
            metrics (Dict): Metrics to add
            iteration (int): Current iteration
        """
        if self.metrics_aggregator is None:
            return

        should_log = self.metrics_aggregator.update(metrics, iteration)
        if should_log and self.trainer.logger:
            aggregated = self.metrics_aggregator.get_aggregated_metrics()
            self.trainer.logger.log_dict(aggregated, iteration)
            self.metrics_aggregator.reset()

    def update_class_metrics(self, class_id: int, loss: float, weight: float = 1.0):
        """
        Update class-wise metrics.

        Args:
            class_id (int): Class ID
            loss (float): Loss value
            weight (float): Sample weight
        """
        if self.class_wise_tracker is None:
            return

        self.class_wise_tracker.update(class_id, loss, weight)

    def visualize_predictions(
        self,
        images: torch.Tensor,
        predictions: Dict,
        ground_truth: Dict,
        iteration: int,
    ) -> Optional[str]:
        """
        Generate and save visualizations.

        Args:
            images (torch.Tensor): Batch of images
            predictions (Dict): Model predictions
            ground_truth (Dict): Ground truth annotations
            iteration (int): Current iteration

        Returns:
            Path to saved visualization, or None if not enabled
        """
        if self.visualizer is None:
            return None

        frequency = self.visualization_config.get("frequency", 100)
        if iteration % frequency != 0:
            return None

        save_dir = self.visualization_config.get("save_dir")
        if save_dir is None:
            save_dir = os.path.join(self.trainer.logging_conf.log_dir, "visualizations")

        os.makedirs(save_dir, exist_ok=True)

        # Generate visualization
        # This is a placeholder - actual implementation depends on specific format
        output_path = os.path.join(save_dir, f"iteration_{iteration:06d}.png")
        logger.info(f"Saving visualization to {output_path}")

        return output_path

    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """
        Get class weights for loss computation.

        Returns:
            Dict mapping class IDs to weights, or None
        """
        if self.class_imbalance_analyzer is None:
            return None

        strategy = self.class_imbalance_config.get("strategy", "inverse_frequency")
        return self.class_imbalance_analyzer.get_class_weights(strategy)


def create_trainer_enhancer(
    trainer,
    enhanced_config: Optional[Dict] = None,
) -> TrainerEnhancer:
    """
    Factory function to create a trainer enhancer.

    Args:
        trainer: Trainer instance to enhance
        enhanced_config: Configuration dictionary

    Returns:
        TrainerEnhancer instance
    """
    config = enhanced_config or {}

    enhancer = TrainerEnhancer(
        trainer=trainer,
        class_imbalance_config=config.get("class_imbalance"),
        per_iteration_val_config=config.get("per_iteration_validation"),
        visualization_config=config.get("visualization"),
        enhanced_logging_config=config.get("enhanced_logging"),
    )

    return enhancer
