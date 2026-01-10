"""
Simple Object Detection Training from Scratch
Train a Faster R-CNN model without pre-trained weights on Pascal VOC dataset
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

from pascal_dataset import PascalVoc, voc_classes
from utility.engine import train_one_epoch, evaluate
import utility.utils as utils
import utility.transforms as T


def get_transform(train=True):
    """Data augmentation transforms"""
    transforms = [T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float32)]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_model_from_scratch(num_classes=21):
    """
    Create Faster R-CNN with ResNet50 backbone WITHOUT pre-trained weights
    This trains the model completely from scratch
    """
    # Create model without pre-trained weights
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    
    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print(f"✓ Model created from scratch with {num_classes} classes")
    return model


def load_data(data_path, train_list, val_list, test_list, batch_size=4, max_samples=None):
    """Load Pascal VOC dataset"""
    # Create datasets
    dataset = PascalVoc(data_path, get_transform(train=True))
    dataset_val = PascalVoc(data_path, get_transform(train=False))
    
    # Get indices for each split
    train_idx = [i for i, img in enumerate(dataset.annot) if img in train_list]
    val_idx = [i for i, img in enumerate(dataset_val.annot) if img in val_list]
    test_idx = [i for i, img in enumerate(dataset_val.annot) if img in test_list]
    
    # Limit dataset size if specified (for quick testing)
    if max_samples is not None:
        train_size = int(max_samples * 0.7)  # 70% for training
        val_size = int(max_samples * 0.2)    # 20% for validation
        test_size = max_samples - train_size - val_size  # Remaining for test
        
        train_idx = train_idx[:train_size]
        val_idx = val_idx[:val_size]
        test_idx = test_idx[:test_size]
        
        print(f"📊 Using limited dataset: {max_samples} samples total (quick training mode)")
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset_val, val_idx)
    test_dataset = torch.utils.data.Subset(dataset_val, test_idx)
    
    # Create data loaders
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
    
    print(f"✓ Data loaded - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, test_idx


def train_model(config):
    """Main training function"""
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"✓ Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, test_idx = load_data(
        config['data_path'],
        config['train_list'],
        config['val_list'],
        config['test_list'],
        config['batch_size'],
        config.get('max_samples', None)  # Pass max_samples parameter
    )
    
    # Create model from scratch
    model = create_model_from_scratch(num_classes=config['num_classes'])
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step'], gamma=config['lr_gamma']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_map': [],
        'val_map50': [],
        'epochs': [],
        'best_map': 0.0
    }
    
    print(f"\n{'='*60}")
    print(f"Starting Training from Scratch")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train one epoch
        train_loss, _ = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=50
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        print("\n" + "="*60)
        print(f"Validation - Epoch {epoch + 1}/{config['epochs']}")
        print("="*60)
        
        coco_evaluator, stats = evaluate(model, val_loader, device)
        
        # Extract mAP metrics
        val_map = stats['bbox'][0]  # mAP @ IoU=0.50:0.95
        val_map50 = stats['bbox'][1]  # mAP @ IoU=0.50
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_map'].append(val_map)
        history['val_map50'].append(val_map50)
        history['epochs'].append(epoch + 1)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val mAP@0.50:0.95: {val_map:.4f}")
        print(f"   Val mAP@0.50: {val_map50:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_map': val_map,
            'val_map50': val_map50,
            'history': history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config['output_dir'], 'latest_checkpoint.pth'))
        
        # Save best model
        if val_map > history['best_map']:
            history['best_map'] = val_map
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"   ⭐ New best model saved! mAP: {val_map:.4f}")
        
        print("="*60 + "\n")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best mAP: {history['best_map']:.4f}")
    print(f"{'='*60}\n")
    
    # Save final training history
    with open(os.path.join(config['output_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    plot_training_curves(history, config['output_dir'])
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    coco_evaluator, test_stats = evaluate(model, test_loader, device)
    
    print(f"\n📊 Test Results:")
    print(f"   mAP@0.50:0.95: {test_stats['bbox'][0]:.4f}")
    print(f"   mAP@0.50: {test_stats['bbox'][1]:.4f}")
    print(f"   mAP@0.75: {test_stats['bbox'][2]:.4f}")
    print("="*60 + "\n")
    
    # Save test indices
    with open(os.path.join(config['output_dir'], 'test_indices.json'), 'w') as f:
        json.dump(test_idx, f)
    
    return history


def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['epochs'], history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot mAP
    axes[1].plot(history['epochs'], history['val_map'], 'g-', linewidth=2, label='mAP@0.50:0.95')
    axes[1].plot(history['epochs'], history['val_map50'], 'r--', linewidth=2, label='mAP@0.50')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mAP', fontsize=12)
    axes[1].set_title('Validation mAP', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {output_dir}/training_curves.png")


if __name__ == '__main__':
    
    # Configuration
    config = {
        # Data
        'data_path': './data',
        'output_dir': './weights_from_scratch',
        'num_classes': 21,  # 20 classes + background
        'max_samples': 100,  # Limit to 100 samples for quick 15-min training (set to None for full dataset)
        
        # Training
        'epochs': 20,
        'batch_size': 1,  # Reduced to 1 for GPUs with limited memory (<4GB)
        'lr': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step': 5,
        'lr_gamma': 0.5,
    }
    
    # Load train/val/test splits
    print("\n" + "="*60)
    print("Loading Dataset Splits")
    print("="*60)
    
    with open('Segmentation/train.txt', 'r') as f:
        train_list = [f'{line.strip()}.xml' for line in f]
    
    with open('Segmentation/val.txt', 'r') as f:
        val_list = [f'{line.strip()}.xml' for line in f]
    
    # Use a portion of val as test (or load from separate file if available)
    test_list = val_list[:len(val_list)//2]  # Use half of val as test
    val_list = val_list[len(val_list)//2:]   # Other half as val
    
    config['train_list'] = train_list
    config['val_list'] = val_list
    config['test_list'] = test_list
    
    print(f"Train samples: {len(train_list)}")
    print(f"Val samples: {len(val_list)}")
    print(f"Test samples: {len(test_list)}")
    
    # Train model
    history = train_model(config)
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"   - Model weights saved in: {config['output_dir']}/")
    print(f"   - Best mAP achieved: {history['best_map']:.4f}")
    print(f"   - Run inference.py to test the model\n")
