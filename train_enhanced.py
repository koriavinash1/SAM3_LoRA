#!/usr/bin/env python3
"""
Enhanced SAM3 LoRA Training Script with Comprehensive Logging and Profiling.

This script trains a LoRA-adapted SAM3 model with detailed logging, profiling,
and experiment tracking.

Features:
    - Comprehensive training logging with file output
    - Loss and metric tracking
    - Learning rate monitoring
    - Checkpoint management with best model tracking
    - Profiling of training steps
    - JSON summary of training results
    - Resume from checkpoint support

Usage:
    python train_enhanced.py \\
        --config configs/lora_config_example.yaml \\
        --log-dir logs/training \\
        --checkpoint-dir checkpoints/training \\
        --resume logs/training/checkpoint.pt
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from src.lora.lora_utils import LoRAConfig
from src.data.dataset import create_dataloaders
from src.train.train_lora import LoRATrainer
from src.utils import setup_logger, Profiler

# Setup logging
logger = None


def setup_experiment_logging(log_dir: str, log_level: int = logging.INFO):
    """
    Setup logging for the training experiment.

    Args:
        log_dir: Directory to save logs
        log_level: Logging level

    Returns:
        Configured logger
    """
    global logger

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "training.log"

    logger = setup_logger(
        name=__name__,
        log_file=str(log_file),
        level=log_level,
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config

    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Configuration loaded:\n{json.dumps(config, indent=2)}")
    return config


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer for LoRA parameters only.

    Args:
        model: Model to optimize
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    # Get only LoRA parameters (trainable)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    n_lora_params = sum(p.numel() for p in lora_params)

    logger.info(f"Optimizer configuration:")
    logger.info(f"  Number of LoRA parameters to train: {n_lora_params:,}")

    optimizer_name = config["training"]["optimizer"].lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    logger.info(f"  Optimizer: {optimizer_name}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Weight decay: {weight_decay}")

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config["training"].get("betas", [0.9, 0.999]),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config["training"].get("betas", [0.9, 0.999]),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=config["training"].get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        num_training_steps: Total number of training steps

    Returns:
        Scheduler instance or None
    """
    scheduler_name = config["training"].get("scheduler", "cosine")
    logger.info(f"Learning rate scheduler: {scheduler_name}")

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
    elif scheduler_name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
        )
    elif scheduler_name == "inverse_sqrt":
        warmup_steps = config["training"].get("warmup_steps", 100)
        logger.info(f"  Warmup steps: {warmup_steps}")

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(warmup_steps) ** 0.5 / float(step) ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}. Using None.")
        scheduler = None

    return scheduler


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        """Initialize metrics tracking."""
        self.losses = []
        self.learning_rates = []
        self.step_times = []
        self.best_loss = float("inf")
        self.best_epoch = 0

    def update(self, loss: float, lr: float, step_time: float):
        """Update metrics with new values."""
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step_times.append(step_time)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = len(self.losses)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.losses:
            return {}

        return {
            "total_steps": len(self.losses),
            "final_loss": round(self.losses[-1], 4),
            "best_loss": round(self.best_loss, 4),
            "best_epoch": self.best_epoch,
            "avg_loss": round(sum(self.losses) / len(self.losses), 4),
            "avg_lr": round(sum(self.learning_rates) / len(self.learning_rates), 6),
            "avg_step_time": round(sum(self.step_times) / len(self.step_times), 4),
        }


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: str,
    epoch: int,
    metrics: TrainingMetrics,
    profiler: Optional[Profiler] = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        epoch: Current epoch number
        metrics: Metrics tracker
        profiler: Optional profiler

    Returns:
        Average epoch loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    if profiler is None:
        profiler = Profiler(name=f"epoch_{epoch}")

    logger.info(f"\nEpoch {epoch} Training:")
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        with profiler.context(f"batch_{batch_idx}"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Metrics
            total_loss += loss.item()
            num_batches += 1
            step_time = profiler.timers.get(f"batch_{batch_idx}").mean_time

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            metrics.update(loss.item(), current_lr, step_time)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.6f}",
            })

            # Log every N batches
            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"loss={loss.item():.4f}, lr={current_lr:.6f}, "
                    f"time={step_time:.4f}s"
                )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader,
    device: str,
    epoch: int,
    profiler: Optional[Profiler] = None,
) -> float:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device
        epoch: Current epoch number
        profiler: Optional profiler

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    if profiler is None:
        profiler = Profiler(name=f"val_epoch_{epoch}")

    logger.info(f"\nEpoch {epoch} Validation:")

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            with profiler.context(f"val_batch_{batch_idx}"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}")

    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metrics: TrainingMetrics,
    checkpoint_path: str,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics.get_summary(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Checkpoint saved to {checkpoint_path}")


def main(args):
    """Main training function."""
    # Setup logging
    setup_experiment_logging(
        log_dir=args.log_dir,
        log_level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("="*60)
    logger.info("SAM3 LoRA Enhanced Training")
    logger.info("="*60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Load configuration
    config = load_config(args.config)

    # Main profiler
    main_profiler = Profiler(name="full_training")

    # Build model
    logger.info("\n--- Building Model ---")
    with main_profiler.context("model_building"):
        model = build_sam3_image_model(
            config["model"]["checkpoint"],
            device=device,
        )

        # Apply LoRA
        lora_config = LoRAConfig(**config["lora"])
        from src.lora.lora_utils import inject_lora
        model = inject_lora(model, lora_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Create data loaders
    logger.info("\n--- Loading Data ---")
    with main_profiler.context("data_loading"):
        train_loader, val_loader = create_dataloaders(
            config["data"],
            batch_size=config["training"].get("batch_size", 8),
            num_workers=config["training"].get("num_workers", 4),
        )

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create optimizer and scheduler
    logger.info("\n--- Setting up Optimizer ---")
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = create_scheduler(optimizer, config, num_training_steps)

    # Training metrics
    metrics = TrainingMetrics()

    # Training loop
    logger.info("\n--- Starting Training ---")
    num_epochs = config["training"]["epochs"]

    with main_profiler.context("training_loop"):
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Train epoch
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                metrics=metrics,
                profiler=main_profiler,
            )

            # Validate epoch
            if val_loader:
                val_loss = validate_epoch(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    epoch=epoch,
                    profiler=main_profiler,
                )
            else:
                val_loss = train_loss

            # Save checkpoint
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save last checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                checkpoint_path=str(checkpoint_dir / "last_checkpoint.pt"),
            )

            # Save best checkpoint
            if train_loss < metrics.best_loss:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    checkpoint_path=str(checkpoint_dir / "best_checkpoint.pt"),
                )

    # Save profiling report
    logger.info(f"\n--- Profiling Report ---")
    if args.profile_report:
        main_profiler.save_summary(args.profile_report)
        logger.info(f"✓ Profiling report saved to {args.profile_report}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)

    summary = metrics.get_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    main_profiler.print_summary()

    # Save training summary
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Summary saved to {summary_path}")

    logger.info("="*60)
    logger.info("Training completed successfully")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced SAM3 LoRA Training with Logging and Profiling"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )

    # Directory arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/training",
        help="Directory for log files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/training",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--profile-report",
        type=str,
        help="Path to save profiling report as JSON",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        help="Path to save training summary as JSON",
    )

    # Training arguments
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from",
    )

    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {str(e)}", exc_info=True)
        else:
            print(f"Fatal error: {str(e)}")
        exit(1)
