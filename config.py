"""
Configuration management for object detection training and inference.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_path: str = './data'
    train_split_file: str = 'Segmentation/train.txt'
    val_split_file: str = 'Segmentation/val.txt'
    num_classes: int = 21
    max_samples: int = None
    
    def __post_init__(self):
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = 'resnet50'
    pretrained: bool = False
    pretrained_backbone: bool = False
    num_classes: int = 21
    min_size: int = 800
    max_size: int = 1333


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 5
    lr_gamma: float = 0.5
    print_freq: int = 50
    num_workers: int = 1
    device: str = 'cuda'
    output_dir: str = './weights_from_scratch'
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    checkpoint_path: str = './weights_from_scratch/best_model.pth'
    confidence_threshold: float = 0.5
    device: str = 'cuda'
    output_dir: str = './inference_results'
    
    def __post_init__(self):
        if not os.path.exists(self.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_quick_train_config() -> Config:
    """Get configuration for quick training (100 samples)."""
    config = Config()
    config.data.max_samples = 100
    return config
