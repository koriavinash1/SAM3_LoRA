#!/usr/bin/env python3
"""
Enhanced SAM3 LoRA Inference Script with Profiling and Comprehensive Logging.

This script runs inference with a LoRA fine-tuned SAM3 model with detailed
profiling of timing and memory usage.

Features:
    - Comprehensive logging with file output
    - Profiling of model loading, preprocessing, and inference
    - Memory tracking (GPU and CPU)
    - Detailed timing statistics
    - JSON profiling report export

Usage:
    python inference_enhanced.py \\
        --config configs/full_lora_config.yaml \\
        --image path/to/image.jpg \\
        --prompt "crack" \\
        --output output.png \\
        --log-dir logs/inference \\
        --profile-report profile.json
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Sam3Model, Sam3Processor

from src.utils import setup_logger, Profiler
from lora_layers import apply_lora_to_model, load_lora_weights, LoRAConfig

# Setup logging
logger = None


def setup_experiment_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO):
    """
    Setup logging for the inference experiment.

    Args:
        log_dir: Directory to save logs
        log_level: Logging level
    """
    global logger

    log_file = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "inference.log"

    logger = setup_logger(
        name=__name__,
        log_file=str(log_file) if log_file else None,
        level=log_level,
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model_with_lora(
    base_model_name: str,
    lora_weights_path: str,
    lora_config_path: Optional[str] = None,
    device: str = "cuda",
    profiler: Optional[Profiler] = None,
) -> tuple:
    """
    Load SAM3 model with LoRA weights.

    Args:
        base_model_name: Base SAM3 model name/path
        lora_weights_path: Path to LoRA weights
        lora_config_path: Path to LoRA config (optional)
        device: Device to load model on
        profiler: Optional profiler for timing measurements

    Returns:
        Tuple of (model, processor)
    """
    if profiler is None:
        profiler = Profiler(name="model_loading")

    logger.info(f"Loading base model: {base_model_name}")
    with profiler.context("load_base_model"):
        model = Sam3Model.from_pretrained(base_model_name)
        processor = Sam3Processor.from_pretrained(base_model_name)
    logger.info("✓ Base model loaded")

    # Load LoRA config if provided
    if lora_config_path:
        logger.info(f"Loading LoRA config from: {lora_config_path}")
        import yaml
        with open(lora_config_path, "r") as f:
            config = yaml.safe_load(f)
        lora_config = LoRAConfig(**config["lora"])
    else:
        logger.warning("No LoRA config provided. Using default configuration.")
        lora_config = LoRAConfig()

    logger.info("Applying LoRA to model...")
    with profiler.context("apply_lora"):
        model = apply_lora_to_model(model, lora_config)
    logger.info("✓ LoRA applied to model")

    logger.info(f"Loading LoRA weights from: {lora_weights_path}")
    with profiler.context("load_lora_weights"):
        load_lora_weights(model, lora_weights_path)
    logger.info("✓ LoRA weights loaded")

    logger.info(f"Moving model to device: {device}")
    with profiler.context("move_to_device"):
        model.to(device)
        model.eval()
    logger.info("✓ Model on device")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    return model, processor, profiler


def preprocess_image(
    image_path: str,
    processor: Sam3Processor,
    device: str = "cuda",
    profiler: Optional[Profiler] = None,
) -> tuple:
    """
    Load and preprocess image.

    Args:
        image_path: Path to input image
        processor: SAM3 processor
        device: Device
        profiler: Optional profiler

    Returns:
        Tuple of (image, processed_inputs)
    """
    if profiler is None:
        profiler = Profiler(name="preprocessing")

    logger.info(f"Loading image from: {image_path}")
    with profiler.context("load_image"):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        logger.info(f"  Image size: {original_size}")

    logger.info("Preprocessing image...")
    with profiler.context("preprocess"):
        # Note: actual preprocessing depends on SAM3 processor implementation
        processed_inputs = None  # Will be prepared during inference
        logger.info(f"  Image shape for model: {image.size}")

    return image, processed_inputs, profiler


def run_inference(
    model: Sam3Model,
    processor: Sam3Processor,
    image: Image.Image,
    text_prompts: List[str],
    device: str = "cuda",
    threshold: float = 0.5,
    profiler: Optional[Profiler] = None,
) -> Dict[str, Any]:
    """
    Run inference on image with multiple prompts.

    Args:
        model: SAM3 model with LoRA
        processor: SAM3 processor
        image: Input PIL image
        text_prompts: List of text prompts
        device: Device
        threshold: Detection confidence threshold
        profiler: Optional profiler

    Returns:
        Dictionary containing results
    """
    if profiler is None:
        profiler = Profiler(name="inference")

    results = {
        "prompts": {},
        "metadata": {
            "device": device,
            "threshold": threshold,
            "image_size": image.size,
        }
    }

    logger.info(f"Running inference with {len(text_prompts)} prompt(s)")
    logger.info(f"  Prompts: {text_prompts}")
    logger.info(f"  Threshold: {threshold}")

    # Measure memory before inference
    profiler.measure_memory("before_inference")

    try:
        with profiler.context("full_inference"):
            for i, prompt in enumerate(text_prompts):
                logger.info(f"Processing prompt {i+1}/{len(text_prompts)}: '{prompt}'")

                with profiler.context(f"inference_{prompt}"):
                    # Prepare inputs
                    inputs = processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt",
                    )

                    # Move to device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Extract predictions
                    masks = outputs.pred_masks if hasattr(outputs, "pred_masks") else None
                    iou_predictions = outputs.iou_pred if hasattr(outputs, "iou_pred") else None

                    if masks is not None and iou_predictions is not None:
                        # Find high-confidence detections
                        high_conf = iou_predictions[0] > threshold
                        n_detections = high_conf.sum().item()
                        max_conf = iou_predictions[0].max().item()

                        results["prompts"][prompt] = {
                            "n_detections": n_detections,
                            "max_confidence": float(max_conf),
                            "masks_shape": masks.shape if masks is not None else None,
                        }

                        logger.info(
                            f"  ✓ Found {n_detections} detections "
                            f"(max confidence: {max_conf:.3f})"
                        )
                    else:
                        logger.warning(f"  ⚠ No outputs for prompt '{prompt}'")
                        results["prompts"][prompt] = {
                            "n_detections": 0,
                            "max_confidence": 0.0,
                            "error": "No output masks or IOU predictions",
                        }

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        results["error"] = str(e)

    # Measure memory after inference
    profiler.measure_memory("after_inference")

    return results


def visualize_results(
    image: Image.Image,
    results: Dict[str, Any],
    output_path: str,
    profiler: Optional[Profiler] = None,
):
    """
    Create visualization of inference results.

    Args:
        image: Input image
        results: Inference results dictionary
        output_path: Path to save visualization
        profiler: Optional profiler
    """
    if profiler is None:
        profiler = Profiler(name="visualization")

    logger.info("Creating visualization...")

    with profiler.context("create_visualization"):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("Inference Results")

        # Add text summary
        summary_text = "Prompt Results:\n"
        for prompt, prompt_results in results["prompts"].items():
            if "error" not in prompt_results:
                summary_text += (
                    f"  • '{prompt}': {prompt_results['n_detections']} detections "
                    f"(max conf: {prompt_results['max_confidence']:.3f})\n"
                )
            else:
                summary_text += f"  • '{prompt}': Error - {prompt_results['error']}\n"

        ax.text(
            0.02, 0.98, summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.axis("off")
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Visualization saved to {output_path}")

        plt.close()


def main(args):
    """Main inference function."""
    # Setup logging
    setup_experiment_logging(
        log_dir=args.log_dir,
        log_level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("="*60)
    logger.info("SAM3 LoRA Enhanced Inference")
    logger.info("="*60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Main profiler
    main_profiler = Profiler(name="full_inference")

    # Load model with LoRA
    logger.info("\n--- Loading Model ---")
    model, processor, loading_profiler = load_model_with_lora(
        base_model_name=args.model,
        lora_weights_path=args.weights,
        lora_config_path=args.config,
        device=device,
        profiler=main_profiler,
    )

    # Preprocess image
    logger.info("\n--- Preprocessing ---")
    image, _, prep_profiler = preprocess_image(
        image_path=args.image,
        processor=processor,
        device=device,
        profiler=main_profiler,
    )

    # Run inference
    logger.info("\n--- Running Inference ---")
    results = run_inference(
        model=model,
        processor=processor,
        image=image,
        text_prompts=args.prompt,
        device=device,
        threshold=args.threshold,
        profiler=main_profiler,
    )

    # Create visualization
    logger.info("\n--- Visualization ---")
    visualize_results(
        image=image,
        results=results,
        output_path=args.output,
        profiler=main_profiler,
    )

    # Save profiling report
    if args.profile_report:
        logger.info(f"\nSaving profiling report to {args.profile_report}")
        main_profiler.save_summary(args.profile_report)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Inference Summary")
    logger.info("="*60)

    main_profiler.print_summary()

    # Save results JSON
    if args.results_json:
        results_path = Path(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Results saved to {results_path}")

    logger.info("="*60)
    logger.info("Inference completed successfully")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced SAM3 LoRA Inference with Profiling"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam3-base",
        help="Base SAM3 model name or path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to LoRA weights file",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to LoRA configuration YAML file",
    )

    # Input/output arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["object"],
        help="Text prompt(s) for segmentation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        help="Path to save results as JSON",
    )

    # Inference arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )

    # Profiling arguments
    parser.add_argument(
        "--profile-report",
        type=str,
        help="Path to save profiling report as JSON",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/inference",
        help="Directory for log files",
    )
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
