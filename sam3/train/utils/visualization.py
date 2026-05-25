# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Visualization utilities for segmentation predictions and ground truth.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from pycocotools import mask as mask_util

import logging


logger = logging.getLogger(__name__)


def get_colormap(num_colors: int = 256) -> np.ndarray:
    """
    Generate a colormap for visualizing different instances/classes.

    Args:
        num_colors (int): Number of colors to generate

    Returns:
        np.ndarray: Color map of shape (num_colors, 3) with RGB values
    """
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3))
    colors[0] = [0, 0, 0]  # Background is black
    return colors.astype(np.uint8)


def decode_rle_mask(rle: Dict, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Decode RLE-encoded mask to binary mask array.

    Args:
        rle (Dict): RLE encoded mask
        image_size (Tuple): (height, width) of image

    Returns:
        np.ndarray: Binary mask of shape (height, width)
    """
    if rle is None:
        return None
    try:
        mask = mask_util.decode(rle)
        return mask.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to decode RLE mask: {e}")
        return None


def denormalize_bbox(bbox: torch.Tensor, image_size: Tuple[int, int]) -> List[float]:
    """
    Convert normalized bbox to pixel coordinates.

    Args:
        bbox (torch.Tensor): Normalized bbox [x, y, w, h] or [x1, y1, x2, y2]
        image_size (Tuple): (height, width) of image

    Returns:
        List: Bbox in pixel coordinates
    """
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()

    height, width = image_size
    if len(bbox) == 4:
        # Assuming XYWH normalized format
        x, y, w, h = bbox
        x1, y1 = int(x * width), int(y * height)
        x2, y2 = int((x + w) * width), int((y + h) * height)
        return [x1, y1, x2, y2]
    return bbox


class SegmentationVisualizer:
    """
    Utility class for visualizing segmentation predictions and ground truth.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_colors: int = 256,
        font_scale: float = 0.5,
        line_width: int = 2,
    ):
        """
        Initialize the visualizer.

        Args:
            image_size (Tuple): (height, width) of image
            num_colors (int): Number of colors for colormap
            font_scale (float): Font scale for text
            line_width (int): Line width for bboxes
        """
        self.image_size = image_size
        self.colormap = get_colormap(num_colors)
        self.font_scale = font_scale
        self.line_width = line_width

    def visualize_instance_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        class_ids: List[int],
        class_names: Optional[Dict[int, str]] = None,
        scores: Optional[List[float]] = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Overlay instance masks on image.

        Args:
            image (np.ndarray): Input image (H, W, 3) in BGR
            masks (List[np.ndarray]): List of binary masks (H, W)
            class_ids (List[int]): Class ID for each mask
            class_names (Dict): Mapping of class ID to name
            scores (List[float]): Confidence scores for each mask
            alpha (float): Alpha blending factor

        Returns:
            np.ndarray: Visualized image
        """
        vis_image = image.copy().astype(np.float32)
        overlay = np.zeros_like(vis_image)

        for idx, (mask, class_id) in enumerate(zip(masks, class_ids)):
            if mask is None:
                continue

            color = self.colormap[class_id % len(self.colormap)].astype(np.float32)
            overlay[mask > 0] = color

        # Blend overlay with image
        vis_image = (1 - alpha) * vis_image + alpha * overlay
        vis_image = np.clip(vis_image, 0, 255).astype(np.uint8)

        # Draw labels
        for idx, (mask, class_id) in enumerate(zip(masks, class_ids)):
            if mask is None:
                continue

            # Find center of mask
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) == 0:
                continue

            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))

            # Create label text
            label = f"C{class_id}"
            if class_names is not None:
                label = class_names.get(class_id, label)
            if scores is not None and idx < len(scores):
                label += f"({scores[idx]:.2f})"

            # Put text on image
            cv2.putText(
                vis_image,
                label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.line_width,
            )

        return vis_image

    def visualize_bboxes(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_ids: List[int],
        class_names: Optional[Dict[int, str]] = None,
        scores: Optional[List[float]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image (np.ndarray): Input image (H, W, 3) in BGR
            bboxes (List): List of [x1, y1, x2, y2] bboxes
            class_ids (List[int]): Class ID for each bbox
            class_names (Dict): Mapping of class ID to name
            scores (List[float]): Confidence scores for each bbox
            color (Tuple): RGB color for bbox

        Returns:
            np.ndarray: Image with drawn bboxes
        """
        vis_image = image.copy()

        for idx, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = [int(c) for c in bbox]

            # Draw bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.line_width)

            # Create label
            label = f"C{class_id}"
            if class_names is not None:
                label = class_names.get(class_id, label)
            if scores is not None and idx < len(scores):
                label += f"({scores[idx]:.2f})"

            # Put text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.line_width,
            )

            # Mark centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(vis_image, (cx, cy), 3, color, -1)

        return vis_image

    def visualize_combined(
        self,
        image: np.ndarray,
        gt_masks: Optional[List[np.ndarray]] = None,
        gt_class_ids: Optional[List[int]] = None,
        pred_masks: Optional[List[np.ndarray]] = None,
        pred_class_ids: Optional[List[int]] = None,
        pred_scores: Optional[List[float]] = None,
        bboxes: Optional[List[List[float]]] = None,
        class_names: Optional[Dict[int, str]] = None,
        class_prompts: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Create a comprehensive visualization showing GT and predictions.

        Args:
            image (np.ndarray): Input image
            gt_masks (List): Ground truth masks
            gt_class_ids (List): Ground truth class IDs
            pred_masks (List): Predicted masks
            pred_class_ids (List): Predicted class IDs
            pred_scores (List): Prediction confidence scores
            bboxes (List): Bounding boxes
            class_names (Dict): Class name mapping
            class_prompts (List): Text prompts for classes

        Returns:
            np.ndarray: Combined visualization
        """
        height, width = self.image_size

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))

        # Plot 1: Original image with GT masks
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Image with Ground Truth")
        if gt_masks is not None:
            for mask, class_id in zip(gt_masks, gt_class_ids):
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Plot 2: Predictions with bboxes
        ax2 = fig.add_subplot(1, 3, 2)
        vis_img = image.copy()
        if pred_masks is not None:
            vis_img = self.visualize_instance_masks(
                vis_img, pred_masks, pred_class_ids, class_names, pred_scores
            )
        ax2.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Predictions")

        # Plot 3: Class distribution text
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis("off")

        # Add class information text
        text_info = "Classes in Image:\n"
        if class_prompts is not None:
            for prompt in class_prompts:
                text_info += f"• {prompt}\n"
        elif gt_class_ids is not None:
            unique_classes = set(gt_class_ids)
            for class_id in sorted(unique_classes):
                class_name = (
                    class_names.get(class_id, f"Class {class_id}")
                    if class_names
                    else f"Class {class_id}"
                )
                text_info += f"• {class_name}\n"

        ax3.text(0.1, 0.5, text_info, fontsize=10, verticalalignment="center")

        plt.tight_layout()
        return fig

    def create_grid_visualization(
        self,
        images: List[np.ndarray],
        all_masks: List[List[np.ndarray]],
        all_class_ids: List[List[int]],
        class_names: Optional[Dict[int, str]] = None,
        rows: int = 2,
        cols: int = 3,
    ) -> np.ndarray:
        """
        Create a grid of visualizations.

        Args:
            images (List): List of images
            all_masks (List): List of mask lists
            all_class_ids (List): List of class ID lists
            class_names (Dict): Class name mapping
            rows (int): Number of rows
            cols (int): Number of columns

        Returns:
            np.ndarray: Grid visualization
        """
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()

        for idx, (image, masks, class_ids) in enumerate(
            zip(images, all_masks, all_class_ids)
        ):
            if idx >= len(axes):
                break

            vis_img = self.visualize_instance_masks(image, masks, class_ids, class_names)
            axes[idx].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Sample {idx}")
            axes[idx].axis("off")

        # Hide unused subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        return fig


def save_visualization(
    fig: plt.Figure, output_path: str, dpi: int = 100, close_fig: bool = True
) -> None:
    """
    Save matplotlib figure to file.

    Args:
        fig (plt.Figure): Matplotlib figure
        output_path (str): Path to save figure
        dpi (int): DPI for saved figure
        close_fig (bool): Whether to close figure after saving
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
