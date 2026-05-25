# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Per-iteration validation module for monitoring training progress.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PerIterationValidator:
    """
    Validates model on a small validation set every N iterations during training.
    Useful for early stopping and monitoring convergence.
    """

    def __init__(
        self,
        val_loader,
        model,
        loss_fn: Callable,
        device: torch.device,
        frequency: int = 50,
        num_batches: int = 5,
        logger_fn: Optional[Callable] = None,
    ):
        """
        Initialize the validator.

        Args:
            val_loader: Validation DataLoader
            model: Model to validate
            loss_fn: Loss function
            device: Device to run validation on
            frequency (int): Validate every N iterations
            num_batches (int): Number of validation batches to use per validation run
            logger_fn: Optional callback function for logging metrics
        """
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.frequency = frequency
        self.num_batches = num_batches
        self.logger_fn = logger_fn
        
        # Create iterator from val_loader
        self.val_iter = iter(val_loader)

    def should_validate(self, iteration: int) -> bool:
        """Check if validation should run at this iteration."""
        return iteration % self.frequency == 0

    def validate(self, iteration: int) -> Dict[str, float]:
        """
        Run validation on a small subset of data.

        Args:
            iteration (int): Current training iteration

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        metrics = {
            "val_loss": 0.0,
            "val_batches": 0,
        }

        with torch.no_grad():
            for batch_idx in range(self.num_batches):
                batch = self._get_next_batch()
                if batch is None:
                    break

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                outputs = self.model(batch)

                # Compute loss
                loss = self.loss_fn(outputs, batch)

                if isinstance(loss, dict):
                    batch_loss = loss.get("loss", 0.0)
                else:
                    batch_loss = loss

                metrics["val_loss"] += batch_loss.item()
                metrics["val_batches"] += 1

        self.model.train()

        # Average metrics
        if metrics["val_batches"] > 0:
            metrics["val_loss"] /= metrics["val_batches"]

        # Log if callback provided
        if self.logger_fn is not None:
            self.logger_fn(metrics, iteration)

        return metrics

    def _get_next_batch(self):
        """Get next batch from validation loader, cycle if needed."""
        try:
            batch = next(self.val_iter)
        except StopIteration:
            # Restart iterator
            self.val_iter = iter(self.val_loader)
            batch = next(self.val_iter)

        return batch

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch tensors to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)
        else:
            return batch


class IterationMetricsAggregator:
    """
    Aggregates metrics over multiple iterations instead of per-image logging.
    """

    def __init__(self, log_frequency: int = 50):
        """
        Initialize the aggregator.

        Args:
            log_frequency (int): Log aggregated metrics every N iterations
        """
        self.log_frequency = log_frequency
        self.iteration_counter = 0
        self.metrics_buffer = {}

    def update(self, metrics: Dict[str, float], iteration: int) -> bool:
        """
        Update metrics buffer.

        Args:
            metrics (Dict): Metrics dictionary
            iteration (int): Current iteration

        Returns:
            bool: Whether aggregated metrics should be logged
        """
        self.iteration_counter = iteration

        # Add metrics to buffer
        for key, value in metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)

        # Check if we should log
        if iteration % self.log_frequency == 0 and iteration > 0:
            return True
        return False

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all buffered iterations.

        Returns:
            Dict with aggregated metrics (mean, std, min, max)
        """
        aggregated = {}
        
        for key, values in self.metrics_buffer.items():
            if values:
                values_array = np.array(values)
                aggregated[f"{key}/mean"] = float(np.mean(values_array))
                aggregated[f"{key}/std"] = float(np.std(values_array))
                aggregated[f"{key}/min"] = float(np.min(values_array))
                aggregated[f"{key}/max"] = float(np.max(values_array))

        return aggregated

    def reset(self):
        """Reset the metrics buffer."""
        self.metrics_buffer = {}

    def log_and_reset(self, log_fn: Callable):
        """
        Get aggregated metrics, log them, and reset buffer.

        Args:
            log_fn: Function to call with aggregated metrics
        """
        metrics = self.get_aggregated_metrics()
        if metrics:
            log_fn(metrics, self.iteration_counter)
        self.reset()


class ClassWiseMetricsTracker:
    """
    Tracks metrics separately for each class to help identify imbalance issues.
    """

    def __init__(self, num_classes: int):
        """
        Initialize tracker.

        Args:
            num_classes (int): Number of classes
        """
        self.num_classes = num_classes
        self.class_metrics = {i: {"loss": [], "count": 0} for i in range(num_classes)}

    def update(self, class_id: int, loss: float, weight: float = 1.0):
        """
        Update metrics for a class.

        Args:
            class_id (int): Class ID
            loss (float): Loss value
            weight (float): Weight for this sample
        """
        if class_id < self.num_classes:
            self.class_metrics[class_id]["loss"].append(loss * weight)
            self.class_metrics[class_id]["count"] += weight

    def get_class_wise_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get per-class statistics.

        Returns:
            Dict mapping class ID to stats
        """
        stats = {}
        for class_id, metrics in self.class_metrics.items():
            losses = np.array(metrics["loss"])
            count = metrics["count"]
            
            if len(losses) > 0:
                stats[class_id] = {
                    "mean_loss": float(np.mean(losses)),
                    "std_loss": float(np.std(losses)) if len(losses) > 1 else 0.0,
                    "count": float(count),
                }
            else:
                stats[class_id] = {
                    "mean_loss": 0.0,
                    "std_loss": 0.0,
                    "count": 0.0,
                }

        return stats

    def reset(self):
        """Reset class metrics."""
        self.class_metrics = {i: {"loss": [], "count": 0} for i in range(self.num_classes)}
