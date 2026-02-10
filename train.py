"""
Object detection training from scratch using Faster R-CNN.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import Config, get_quick_train_config
from logger import setup_logger
from pascal_dataset import PascalVoc
from utility.engine import train_one_epoch, evaluate
import utility.utils as utils
import utility.transforms as T

logger = setup_logger("training")


def get_transform(train: bool = True) -> T.Compose:
    """Get data augmentation transforms."""
    transforms = [T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float32)]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_model_from_scratch(num_classes: int = 21) -> nn.Module:
    """Create Faster R-CNN with ResNet50 backbone without pre-trained weights."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    logger.info(f"Model created from scratch with {num_classes} classes")
    return model


def load_data(
    data_path: str,
    train_list: List[str],
    val_list: List[str],
    test_list: List[str],
    batch_size: int = 4,
    max_samples: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """Load Pascal VOC dataset with train/val/test splits."""
    dataset = PascalVoc(data_path, get_transform(train=True))
    dataset_val = PascalVoc(data_path, get_transform(train=False))
    
    train_idx = [i for i, img in enumerate(dataset.annot) if img in train_list]
    val_idx = [i for i, img in enumerate(dataset_val.annot) if img in val_list]
    test_idx = [i for i, img in enumerate(dataset_val.annot) if img in test_list]
    
    if max_samples is not None:
        train_size = int(max_samples * 0.7)
        val_size = int(max_samples * 0.2)
        test_size = max_samples - train_size - val_size
        
        train_idx = train_idx[:train_size]
        val_idx = val_idx[:val_size]
        test_idx = test_idx[:test_size]
        
        logger.info(f"Using limited dataset: {max_samples} samples (quick training mode)")
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset_val, val_idx)
    test_dataset = torch.utils.data.Subset(dataset_val, test_idx)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1, collate_fn=utils.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, collate_fn=utils.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, collate_fn=utils.collate_fn
    )
    
    logger.info(f"Data loaded - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, test_idx


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict,
    history: Dict,
    filepath: Path
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **metrics,
        'history': history
    }
    torch.save(checkpoint, filepath)


def plot_training_curves(history: Dict, output_dir: Path) -> None:
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['epochs'], history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(history['epochs'], history['val_map'], 'g-', linewidth=2, label='mAP@0.50:0.95')
    axes[1].plot(history['epochs'], history['val_map50'], 'r--', linewidth=2, label='mAP@0.50')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mAP', fontsize=12)
    axes[1].set_title('Validation mAP', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {output_dir}/training_curves.png")


def train_model(
    config: Config,
    train_list: List[str],
    val_list: List[str],
    test_list: List[str]
) -> Dict:
    """Main training function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, test_loader, test_idx = load_data(
        config.data.data_path,
        train_list,
        val_list,
        test_list,
        config.training.batch_size,
        config.data.max_samples
    )
    
    model = create_model_from_scratch(num_classes=config.model.num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.lr_step_size,
        gamma=config.training.lr_gamma
    )
    
    history = {
        'train_loss': [],
        'val_map': [],
        'val_map50': [],
        'epochs': [],
        'best_map': 0.0
    }
    
    logger.info("="*60)
    logger.info("Starting Training from Scratch")
    logger.info("="*60)
    
    start_time = time.time()
    
    for epoch in range(config.training.epochs):
        epoch_start = time.time()
        
        train_loss, _ = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            print_freq=config.training.print_freq
        )
        
        lr_scheduler.step()
        
        logger.info("="*60)
        logger.info(f"Validation - Epoch {epoch + 1}/{config.training.epochs}")
        logger.info("="*60)
        
        coco_evaluator, stats = evaluate(model, val_loader, device)
        
        val_map = stats['bbox'][0]
        val_map50 = stats['bbox'][1]
        
        history['train_loss'].append(train_loss)
        history['val_map'].append(val_map)
        history['val_map50'].append(val_map50)
        history['epochs'].append(epoch + 1)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val mAP@0.50:0.95: {val_map:.4f}")
        logger.info(f"  Val mAP@0.50: {val_map50:.4f}")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        metrics = {
            'train_loss': train_loss,
            'val_map': val_map,
            'val_map50': val_map50
        }
        
        save_checkpoint(
            model, optimizer, lr_scheduler, epoch + 1,
            metrics, history, output_dir / 'latest_checkpoint.pth'
        )
        
        if val_map > history['best_map']:
            history['best_map'] = val_map
            save_checkpoint(
                model, optimizer, lr_scheduler, epoch + 1,
                metrics, history, output_dir / 'best_model.pth'
            )
            logger.info(f"  New best model saved! mAP: {val_map:.4f}")
        
        logger.info("="*60)
    
    total_time = time.time() - start_time
    logger.info(f"Training Complete!")
    logger.info(f"Total Time: {total_time/60:.2f} minutes")
    logger.info(f"Best mAP: {history['best_map']:.4f}")
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_training_curves(history, output_dir)
    
    logger.info("="*60)
    logger.info("Final Test Set Evaluation")
    logger.info("="*60)
    
    best_checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    coco_evaluator, test_stats = evaluate(model, test_loader, device)
    
    logger.info("Test Results:")
    logger.info(f"  mAP@0.50:0.95: {test_stats['bbox'][0]:.4f}")
    logger.info(f"  mAP@0.50: {test_stats['bbox'][1]:.4f}")
    logger.info(f"  mAP@0.75: {test_stats['bbox'][2]:.4f}")
    logger.info("="*60)
    
    with open(output_dir / 'test_indices.json', 'w') as f:
        json.dump(test_idx, f)
    
    return history


def load_splits(config: Config) -> Tuple[List[str], List[str], List[str]]:
    """Load train/val/test splits from files."""
    with open(config.data.train_split_file, 'r') as f:
        train_list = [f'{line.strip()}.xml' for line in f]
    
    with open(config.data.val_split_file, 'r') as f:
        val_list = [f'{line.strip()}.xml' for line in f]
    
    test_list = val_list[:len(val_list)//2]
    val_list = val_list[len(val_list)//2:]
    
    logger.info(f"Dataset splits loaded:")
    logger.info(f"  Train: {len(train_list)} samples")
    logger.info(f"  Val: {len(val_list)} samples")
    logger.info(f"  Test: {len(test_list)} samples")
    
    return train_list, val_list, test_list


def main():
    """Main entry point."""
    config = get_quick_train_config()
    
    train_list, val_list, test_list = load_splits(config)
    history = train_model(config, train_list, val_list, test_list)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Model weights saved in: {config.training.output_dir}")
    logger.info(f"Best mAP achieved: {history['best_map']:.4f}")
    logger.info("Run evaluate_model.py for detailed evaluation")


if __name__ == '__main__':
    main()
