"""
Model evaluation and benchmarking for object detection.
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import Config
from logger import setup_logger
from pascal_dataset import PascalVoc, VOC_CLASSES
from utility.engine import evaluate
import utility.utils as utils
import utility.transforms as T

logger = setup_logger("evaluation")


def get_model_size(model: torch.nn.Module) -> Tuple[float, int, int]:
    """Calculate model size in MB and parameter counts."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return size_mb, total_params, trainable_params


def benchmark_fps(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    num_iterations: int = 100,
    image_size: Tuple[int, int] = (640, 480)
) -> Tuple[float, float]:
    """Benchmark inference FPS."""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model([dummy_input[0]])
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model([dummy_input[0]])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    return fps, avg_time


def evaluate_per_class(coco_evaluator) -> list:
    """Extract per-class AP from COCO evaluator."""
    det_precision = coco_evaluator.coco_eval['bbox'].eval['precision']
    
    per_class_ap = []
    for cls_idx in range(det_precision.shape[2]):
        ap = det_precision[0, :, cls_idx, 0, 2].mean()
        per_class_ap.append(float(ap))
    
    return per_class_ap


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
    return model, checkpoint


def main():
    logger.info("="*80)
    logger.info("MODEL EVALUATION & BENCHMARKING")
    logger.info("="*80)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = Path('./weights_from_scratch/best_model.pth')
    
    if not checkpoint_path.exists():
        logger.error("Model checkpoint not found. Train model first using train.py")
        return
    
    logger.info("Loading Model...")
    model, checkpoint = load_model(checkpoint_path, 21, device)
    
    logger.info("="*80)
    logger.info("1. MODEL SIZE ANALYSIS")
    logger.info("="*80)
    
    size_mb, total_params, trainable_params = get_model_size(model)
    
    logger.info("Model Statistics:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Model Size: {size_mb:.2f} MB")
    
    checkpoint_size = checkpoint_path.stat().st_size / (1024**2)
    logger.info(f"  Checkpoint File Size: {checkpoint_size:.2f} MB")
    
    logger.info("="*80)
    logger.info("2. INFERENCE SPEED BENCHMARK")
    logger.info("="*80)
    
    test_sizes = [(640, 480), (800, 600), (1024, 768)]
    fps_results = {}
    
    for size in test_sizes:
        fps, avg_time = benchmark_fps(model, device, image_size=size, num_iterations=100)
        fps_results[f"{size[0]}x{size[1]}"] = {
            'fps': fps,
            'time_ms': avg_time * 1000
        }
        logger.info(f"Resolution {size[0]}x{size[1]}: FPS={fps:.2f}, Time={avg_time*1000:.2f}ms")
    
    logger.info("="*80)
    logger.info("3. TEST SET EVALUATION")
    logger.info("="*80)
    
    with open('./weights_from_scratch/test_indices.json', 'r') as f:
        test_idx = json.load(f)
    
    dataset = PascalVoc('./data', T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)]))
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                            num_workers=2, collate_fn=utils.collate_fn)
    
    logger.info(f"Evaluating on {len(test_idx)} test images...")
    
    coco_evaluator, stats = evaluate(model, test_loader, device)
    bbox_stats = stats['bbox']
    
    logger.info("Detection Metrics (COCO-style):")
    logger.info(f"  mAP @ IoU=0.50:0.95: {bbox_stats[0]:.4f}")
    logger.info(f"  mAP @ IoU=0.50: {bbox_stats[1]:.4f}")
    logger.info(f"  mAP @ IoU=0.75: {bbox_stats[2]:.4f}")
    logger.info(f"  mAP (small): {bbox_stats[3]:.4f}")
    logger.info(f"  mAP (medium): {bbox_stats[4]:.4f}")
    logger.info(f"  mAP (large): {bbox_stats[5]:.4f}")
    
    logger.info("="*80)
    logger.info("4. PER-CLASS AVERAGE PRECISION")
    logger.info("="*80)
    
    per_class_ap = evaluate_per_class(coco_evaluator)
    
    logger.info(f"{'Class':<20} {'AP@0.50:0.95':<15}")
    logger.info("-" * 35)
    
    for class_name, ap in zip(VOC_CLASSES[1:], per_class_ap[1:]):
        logger.info(f"{class_name:<20} {ap:.4f}")
    
    mean_ap = sum(per_class_ap[1:]) / len(per_class_ap[1:])
    logger.info("-" * 35)
    logger.info(f"{'Mean (all classes)':<20} {mean_ap:.4f}")
    
    logger.info("="*80)
    logger.info("5. GENERATING EVALUATION REPORT")
    logger.info("="*80)
    
    report = {
        'model_info': {
            'architecture': 'Faster R-CNN ResNet-50 FPN',
            'training_mode': 'From Scratch (No Pre-training)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': size_mb,
            'checkpoint_size_mb': checkpoint_size,
            'training_epochs': checkpoint['epoch']
        },
        'performance_metrics': {
            'map_iou_50_95': float(bbox_stats[0]),
            'map_iou_50': float(bbox_stats[1]),
            'map_iou_75': float(bbox_stats[2]),
            'map_small': float(bbox_stats[3]),
            'map_medium': float(bbox_stats[4]),
            'map_large': float(bbox_stats[5])
        },
        'inference_speed': fps_results,
        'per_class_ap': {
            class_name: float(ap) 
            for class_name, ap in zip(VOC_CLASSES[1:], per_class_ap[1:])
        }
    }
    
    report_path = Path('./weights_from_scratch/evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Detailed report saved to: {report_path}")
    
    logger.info("="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: Faster R-CNN ResNet-50 FPN (From Scratch)")
    logger.info(f"Parameters: {total_params:,} ({size_mb:.2f} MB)")
    logger.info(f"Best Epoch: {checkpoint['epoch']}")
    logger.info(f"Test mAP@0.50:0.95: {bbox_stats[0]:.4f}")
    logger.info(f"Test mAP@0.50: {bbox_stats[1]:.4f}")
    logger.info(f"Inference FPS (640x480): {fps_results['640x480']['fps']:.2f}")
    logger.info("="*80)
    logger.info("Evaluation complete! Check evaluation_report.json for details.")


if __name__ == '__main__':
    main()
