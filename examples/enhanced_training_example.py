#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Example script demonstrating enhanced SAM3 training with class imbalance handling.

This script shows how to:
1. Analyze class imbalance in dataset
2. Setup per-iteration validation
3. Enable visualization
4. Use enhanced logging
5. Train with class-aware loss weighting
"""

import json
import logging
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset_imbalance(coco_json_path: str) -> Dict:
    """
    Analyze class imbalance in COCO dataset.

    Args:
        coco_json_path: Path to COCO JSON file

    Returns:
        Dictionary with imbalance analysis
    """
    from sam3.train.utils.class_imbalance import ClassImbalanceAnalyzer

    logger.info("Analyzing dataset class imbalance...")
    analyzer = ClassImbalanceAnalyzer(coco_json_path)

    # Get statistics
    stats = analyzer.get_statistics_summary()
    split_stats = analyzer.get_split_statistics()
    class_weights = analyzer.get_class_weights(strategy="balanced")

    logger.info("=" * 80)
    logger.info("DATASET IMBALANCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Total instances: {stats['total_instances']}")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Number of classes: {stats['num_classes']}")
    logger.info(f"Imbalance ratio (max/min): {stats['imbalance_ratio']:.2f}")
    logger.info("-" * 80)

    logger.info("Per-split statistics:")
    for split, split_stat in split_stats.items():
        logger.info(f"\n{split}:")
        logger.info(f"  Total instances: {split_stat['total_instances']}")
        logger.info(f"  Total images: {split_stat['total_images']}")
        logger.info(f"  Classes: {split_stat['num_classes']}")

    logger.info("\nClass weights (for balanced training):")
    for class_id, weight in sorted(class_weights.items()):
        class_name = stats["class_names"].get(class_id, f"Class {class_id}")
        logger.info(f"  {class_name}: {weight:.4f}")

    logger.info("=" * 80)

    return {
        "stats": stats,
        "split_stats": split_stats,
        "class_weights": class_weights,
        "analyzer": analyzer,
    }


def setup_enhanced_training_config() -> Dict:
    """
    Setup configuration for enhanced training.

    Returns:
        Configuration dictionary
    """
    config = {
        "class_imbalance": {
            "enabled": True,
            "strategy": "weighted_sampling",
            "weight_power": 1.0,
            "analyze_statistics": True,
        },
        "per_iteration_validation": {
            "enabled": True,
            "frequency": 50,
            "num_batches": 5,
            "log_metrics": True,
        },
        "visualization": {
            "enabled": True,
            "frequency": 100,
            "num_samples": 4,
            "save_dir": "./logs/visualizations",
            "overlay_alpha": 0.6,
            "draw_bboxes": True,
            "draw_centroids": True,
        },
        "enhanced_logging": {
            "enabled": True,
            "log_every_n_iterations": 50,
            "class_wise_metrics": True,
            "aggregation_strategy": "mean",
        },
        "split_aware_loading": True,
        "preserve_attributes": True,
    }
    return config


def create_enhanced_trainer(trainer, coco_json_path: str, enhanced_config: Dict):
    """
    Create an enhanced trainer with all new features.

    Args:
        trainer: Base trainer instance
        coco_json_path: Path to COCO JSON file
        enhanced_config: Enhancement configuration

    Returns:
        Enhanced trainer instance
    """
    from sam3.train.utils.trainer_enhancer import create_trainer_enhancer

    # Update config with annotation file for class imbalance analysis
    enhanced_config["class_imbalance"]["annotation_file"] = coco_json_path

    # Create enhancer
    enhancer = create_trainer_enhancer(
        trainer=trainer, enhanced_config=enhanced_config
    )

    logger.info("Enhanced trainer created with:")
    logger.info("  - Class imbalance handling")
    logger.info("  - Per-iteration validation")
    logger.info("  - Visualization")
    logger.info("  - Enhanced logging")

    return enhancer


def training_loop_example(
    trainer,
    train_loader: DataLoader,
    enhancer,
    num_epochs: int = 10,
    max_iterations: Optional[int] = None,
):
    """
    Example training loop with enhanced features.

    Args:
        trainer: Enhanced trainer instance
        train_loader: Training data loader
        enhancer: Trainer enhancer instance
        num_epochs: Number of training epochs
        max_iterations: Max iterations to run (for testing)
    """
    logger.info("Starting enhanced training...")

    iteration = 0
    max_iters = max_iterations or float("inf")

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Training step (simplified example)
            # In actual implementation, this would be your model forward/backward pass
            loss = torch.tensor(1.0)  # Placeholder

            # Every N iterations, perform enhanced operations
            if iteration % 10 == 0:
                # Update aggregated metrics
                metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": 0.0001,
                }
                enhancer.update_metrics(metrics, iteration)

                # Per-iteration validation
                val_metrics = enhancer.validate_per_iteration(iteration)
                if val_metrics:
                    logger.info(f"Iteration {iteration}: Validation metrics: {val_metrics}")

                # Track class-wise metrics (example)
                for class_id in range(5):  # Assuming 5 classes
                    class_loss = loss.item() * (1.0 + 0.1 * class_id)
                    enhancer.update_class_metrics(class_id, class_loss)

                # Log every 50 iterations
                if iteration % 50 == 0 and iteration > 0:
                    logger.info(
                        f"Iteration {iteration}: Loss = {loss.item():.4f}"
                    )

            iteration += 1

            if iteration >= max_iters:
                logger.info(f"Reached max iterations ({max_iters})")
                return

        logger.info(f"Epoch {epoch + 1} completed")


def main():
    """Main example function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced SAM3 training with class imbalance handling"
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        required=True,
        help="Path to COCO JSON annotation file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze imbalance, don't train",
    )

    args = parser.parse_args()

    # Step 1: Analyze dataset
    analysis = analyze_dataset_imbalance(args.coco_json)

    if args.analyze_only:
        logger.info("Analysis complete. Exiting.")
        return

    # Step 2: Setup configuration
    enhanced_config = setup_enhanced_training_config()

    # Step 3: Create trainer (example - you would use your actual trainer)
    logger.info("Creating trainer... (This is a placeholder)")
    # trainer = create_your_trainer(config)

    # Step 4: Create enhanced trainer
    # enhancer = create_enhanced_trainer(trainer, args.coco_json, enhanced_config)

    # Step 5: Create data loaders
    # train_loader = create_train_loader(args.coco_json)

    # Step 6: Run training with enhancements
    # training_loop_example(trainer, train_loader, enhancer, num_epochs=10, max_iterations=100)

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
