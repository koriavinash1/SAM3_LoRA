# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Utilities for handling class imbalance in datasets.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ClassImbalanceAnalyzer:
    """
    Analyzes and provides utilities for handling class imbalance in datasets.
    """

    def __init__(self, annotation_file: str):
        """
        Initialize the analyzer with a COCO JSON file.

        Args:
            annotation_file (str): Path to COCO JSON annotation file
        """
        with open(annotation_file, "r") as f:
            self.coco = json.load(f)

        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.coco["categories"]}
        self._compute_class_statistics()

    def _compute_class_statistics(self):
        """Compute class frequency and other statistics."""
        self.class_counts = Counter()
        self.class_instances_per_image = defaultdict(Counter)
        self.image_class_coverage = defaultdict(set)

        for ann in self.coco["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            self.class_counts[cat_id] += 1
            self.class_instances_per_image[img_id][cat_id] += 1
            self.image_class_coverage[img_id].add(cat_id)

        self.total_instances = sum(self.class_counts.values())
        self.total_images = len(self.coco["images"])

    def get_class_weights(self, strategy: str = "inverse_frequency") -> Dict[int, float]:
        """
        Compute class weights for loss computation.

        Args:
            strategy (str): Weighting strategy. Options:
                - "inverse_frequency": 1 / frequency
                - "balanced": N / (n_classes * frequency)
                - "effective_number": (1 - beta) / (1 - beta^frequency) where beta=0.9999

        Returns:
            Dict mapping category IDs to weights
        """
        if strategy == "inverse_frequency":
            weights = {}
            for cat_id in self.cat_id_to_name.keys():
                freq = self.class_counts.get(cat_id, 1)
                weights[cat_id] = 1.0 / freq
        elif strategy == "balanced":
            n_classes = len(self.cat_id_to_name)
            weights = {}
            for cat_id in self.cat_id_to_name.keys():
                freq = self.class_counts.get(cat_id, 1)
                weights[cat_id] = self.total_instances / (n_classes * freq)
        elif strategy == "effective_number":
            beta = 0.9999
            weights = {}
            for cat_id in self.cat_id_to_name.keys():
                freq = self.class_counts.get(cat_id, 1)
                weights[cat_id] = (1 - beta) / (1 - beta ** freq)
        else:
            raise ValueError(f"Unknown weighting strategy: {strategy}")

        # Normalize weights to sum to number of classes
        n_classes = len(self.cat_id_to_name)
        total_weight = sum(weights.values())
        weights = {cat_id: (w * n_classes / total_weight) for cat_id, w in weights.items()}

        return weights

    def get_class_frequencies(self) -> Dict[int, float]:
        """
        Get frequency of each class (as fraction of total instances).

        Returns:
            Dict mapping category IDs to frequencies
        """
        return {
            cat_id: self.class_counts.get(cat_id, 0) / self.total_instances
            for cat_id in self.cat_id_to_name.keys()
        }

    def get_imbalance_ratio(self) -> float:
        """
        Get imbalance ratio (max_freq / min_freq).

        Returns:
            float: Imbalance ratio
        """
        if not self.class_counts:
            return 1.0
        max_count = max(self.class_counts.values())
        min_count = min(c for c in self.class_counts.values() if c > 0)
        return max_count / min_count if min_count > 0 else 1.0

    def get_statistics_summary(self) -> Dict:
        """
        Get summary statistics of class distribution.

        Returns:
            Dict with statistics
        """
        counts = list(self.class_counts.values())
        return {
            "total_instances": self.total_instances,
            "total_images": self.total_images,
            "num_classes": len(self.cat_id_to_name),
            "mean_instances_per_class": np.mean(counts) if counts else 0,
            "std_instances_per_class": np.std(counts) if counts else 0,
            "min_instances": min(counts) if counts else 0,
            "max_instances": max(counts) if counts else 0,
            "imbalance_ratio": self.get_imbalance_ratio(),
            "class_counts": dict(self.class_counts),
            "class_names": {k: self.cat_id_to_name[k] for k in self.cat_id_to_name},
        }

    def get_instances_per_image_stats(self) -> Dict:
        """
        Get statistics about instances per image.

        Returns:
            Dict with per-image statistics
        """
        instances_per_image = [
            len([ann for ann in self.coco["annotations"] if ann["image_id"] == img_id])
            for img_id in range(len(self.coco["images"]))
        ]
        instances_per_image = [x for x in instances_per_image if x > 0]

        return {
            "mean_instances_per_image": np.mean(instances_per_image) if instances_per_image else 0,
            "std_instances_per_image": np.std(instances_per_image) if instances_per_image else 0,
            "min_instances_per_image": min(instances_per_image) if instances_per_image else 0,
            "max_instances_per_image": max(instances_per_image) if instances_per_image else 0,
            "images_with_no_annotations": len(self.coco["images"]) - len(
                set(ann["image_id"] for ann in self.coco["annotations"])
            ),
        }

    def get_split_statistics(self) -> Dict[str, Dict]:
        """
        Get class distribution statistics per split (train/val/test).

        Returns:
            Dict with per-split statistics
        """
        split_stats = defaultdict(lambda: Counter())
        split_image_counts = defaultdict(int)

        for img in self.coco["images"]:
            split = img.get("split", "unknown")
            split_image_counts[split] += 1

        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            # Find the image to get split
            img = next((img for img in self.coco["images"] if img["id"] == img_id), None)
            if img:
                split = img.get("split", "unknown")
                split_stats[split][ann["category_id"]] += 1

        return {
            split: {
                "counts": dict(counter),
                "total_instances": sum(counter.values()),
                "total_images": split_image_counts[split],
                "num_classes": len(counter),
            }
            for split, counter in split_stats.items()
        }


def create_class_weights_tensor(
    class_weights: Dict[int, float], num_classes: int, device: torch.device
) -> torch.Tensor:
    """
    Create a weight tensor for use with loss functions.

    Args:
        class_weights (Dict): Mapping of class IDs to weights
        num_classes (int): Total number of classes
        device (torch.device): Device to place tensor on

    Returns:
        torch.Tensor: Weight tensor of shape (num_classes,)
    """
    weights = torch.ones(num_classes, device=device)
    for class_id, weight in class_weights.items():
        if class_id < num_classes:
            weights[class_id] = weight
    return weights
