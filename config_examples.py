#!/usr/bin/env python3
"""
Configuration examples for SAM3 LoRA training.

This file contains example configurations for different training scenarios.
Copy and modify as needed for your use case.
"""

# Example 1: Basic LoRA Fine-tuning
BASIC_CONFIG = """
# Basic LoRA fine-tuning configuration
model:
  checkpoint: "facebook/sam3-base"

lora:
  rank: 8
  alpha: 16
  dropout: 0.1
  target_modules:
    - "qkv"
    - "proj"

data:
  train_dataset_path: "data/train"
  val_dataset_path: "data/val"
  image_size: 1008
  num_samples: 1000

training:
  epochs: 5
  batch_size: 4
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"
  num_workers: 2
"""

# Example 2: Large-scale training
LARGE_SCALE_CONFIG = """
# Large-scale LoRA training configuration
model:
  checkpoint: "facebook/sam3-base"

lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - "qkv"
    - "proj"
    - "mlp"

data:
  train_dataset_path: "data/large_train"
  val_dataset_path: "data/large_val"
  image_size: 1024
  num_samples: 50000

training:
  epochs: 20
  batch_size: 32
  learning_rate: 5e-5
  weight_decay: 0.001
  optimizer: "adamw"
  scheduler: "inverse_sqrt"
  warmup_steps: 10000
  num_workers: 8
  gradient_clip: 1.0
"""

# Example 3: Fast fine-tuning (minimal resources)
FAST_FINETUNE_CONFIG = """
# Fast fine-tuning for quick experimentation
model:
  checkpoint: "facebook/sam3-base"

lora:
  rank: 4
  alpha: 8
  dropout: 0.2
  target_modules:
    - "qkv"

data:
  train_dataset_path: "data/train"
  val_dataset_path: "data/val"
  image_size: 512
  num_samples: 100

training:
  epochs: 2
  batch_size: 2
  learning_rate: 2e-4
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "linear"
  num_workers: 2
"""

# Example 4: Specialized task (e.g., medical imaging)
MEDICAL_CONFIG = """
# Medical image segmentation configuration
model:
  checkpoint: "facebook/sam3-base"

lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "qkv"
    - "proj"
    - "mlp"

data:
  train_dataset_path: "data/medical_train"
  val_dataset_path: "data/medical_val"
  image_size: 512
  num_samples: 5000

training:
  epochs: 15
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_steps: 500
  num_workers: 4
"""

def save_examples():
    """Save example configurations to files."""
    examples = {
        "configs/basic_lora_config.yaml": BASIC_CONFIG,
        "configs/large_scale_config.yaml": LARGE_SCALE_CONFIG,
        "configs/fast_finetune_config.yaml": FAST_FINETUNE_CONFIG,
        "configs/medical_config.yaml": MEDICAL_CONFIG,
    }

    for path, content in examples.items():
        with open(path, "w") as f:
            f.write(content.strip())
        print(f"Created: {path}")

if __name__ == "__main__":
    import os
    os.makedirs("configs", exist_ok=True)
    save_examples()
    print("\n✓ Configuration examples saved to configs/ directory")
