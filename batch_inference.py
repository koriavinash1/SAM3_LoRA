#!/usr/bin/env python3
"""
Utility script for batch inference on multiple images.

This script processes multiple images with the same LoRA model,
generating results and profiling reports.

Usage:
    python batch_inference.py \\
        --weights path/to/weights.pt \\
        --image-dir path/to/images \\
        --prompt "crack" \\
        --output-dir results \\
        --generate-report
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from inference_enhanced import (
    setup_experiment_logging,
    load_model_with_lora,
    preprocess_image,
    run_inference,
    visualize_results,
)
from src.utils import Profiler

logger = None


def main(args):
    """Main batch inference function."""
    # Setup logging
    setup_experiment_logging(
        log_dir=args.log_dir,
        log_level=logging.DEBUG if args.debug else logging.INFO,
    )
    global logger
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("SAM3 LoRA Batch Inference")
    logger.info("="*60)

    # Find images
    image_dir = Path(args.image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        logger.error(f"No images found in {image_dir}")
        return

    logger.info(f"Found {len(images)} images to process")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    logger.info("\n--- Loading Model ---")
    model, processor, _ = load_model_with_lora(
        base_model_name=args.model,
        lora_weights_path=args.weights,
        lora_config_path=args.config,
        device="cuda" if not args.cpu else "cpu",
    )

    # Process each image
    all_results = {}
    batch_profiler = Profiler(name="batch_inference")

    logger.info("\n--- Processing Images ---")
    for idx, image_path in enumerate(images, 1):
        logger.info(f"\n[{idx}/{len(images)}] Processing: {image_path.name}")

        try:
            with batch_profiler.context(f"image_{image_path.stem}"):
                # Load and preprocess image
                image, _, _ = preprocess_image(
                    str(image_path),
                    processor,
                    device="cuda" if not args.cpu else "cpu",
                )

                # Run inference
                results = run_inference(
                    model=model,
                    processor=processor,
                    image=image,
                    text_prompts=args.prompt,
                    device="cuda" if not args.cpu else "cpu",
                    threshold=args.threshold,
                )

                # Visualize results
                output_image_path = output_dir / f"{image_path.stem}_result.png"
                visualize_results(
                    image=image,
                    results=results,
                    output_path=str(output_image_path),
                )

                # Save results JSON
                output_json_path = output_dir / f"{image_path.stem}_results.json"
                with open(output_json_path, "w") as f:
                    json.dump(results, f, indent=2)

                all_results[image_path.name] = results
                logger.info(f"✓ Saved results to {output_image_path}")

        except Exception as e:
            logger.error(f"✗ Error processing {image_path.name}: {str(e)}")
            all_results[image_path.name] = {"error": str(e)}

    # Save batch results
    logger.info("\n--- Saving Results ---")
    batch_results_path = output_dir / "batch_results.json"
    with open(batch_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"✓ Batch results saved to {batch_results_path}")

    # Save profiling report
    if args.profile_report:
        batch_profiler.save_summary(args.profile_report)
        logger.info(f"✓ Profiling report saved to {args.profile_report}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Batch Inference Summary")
    logger.info("="*60)

    successful = sum(1 for r in all_results.values() if "error" not in r)
    failed = len(all_results) - successful

    logger.info(f"Total images: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if successful > 0:
        batch_profiler.print_summary()

    logger.info("="*60)
    logger.info("Batch inference completed!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference with SAM3 LoRA"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam3-base",
        help="Base SAM3 model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to LoRA config",
    )

    # Input/output arguments
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_results",
        help="Output directory for results",
    )

    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["object"],
        help="Text prompts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU",
    )

    # Profiling arguments
    parser.add_argument(
        "--profile-report",
        type=str,
        help="Save profiling report",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/batch_inference",
        help="Log directory",
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
