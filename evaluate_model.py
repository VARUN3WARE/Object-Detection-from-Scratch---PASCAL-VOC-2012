"""
Model Evaluation and Benchmarking Script
Compute model size, FPS, and detailed mAP metrics
"""

import os
import json
import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from pascal_dataset import PascalVoc, voc_classes
from utility.engine import evaluate
import utility.utils as utils
import utility.transforms as T


def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return size_mb, total_params, trainable_params


def benchmark_fps(model, device, batch_size=1, num_iterations=100, image_size=(640, 480)):
    """Benchmark inference FPS"""
    
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model([dummy_input[0]])
    
    # Benchmark
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


def evaluate_per_class(coco_evaluator):
    """Extract per-class AP from COCO evaluator"""
    
    # Detection metrics
    det_precision = coco_evaluator.coco_eval['bbox'].eval['precision']
    
    # Per-class AP at IoU=0.50:0.95
    per_class_ap = []
    for cls_idx in range(det_precision.shape[2]):
        ap = det_precision[0, :, cls_idx, 0, 2].mean()
        per_class_ap.append(float(ap))
    
    return per_class_ap


def main():
    
    print("\n" + "="*80)
    print(" "*20 + "MODEL EVALUATION & BENCHMARKING")
    print("="*80 + "\n")
    
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = './weights_from_scratch/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print("❌ Model checkpoint not found. Train model first using train_from_scratch.py")
        return
    
    # Load model
    print("📦 Loading Model...")
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    # 1. Model Size
    print("\n" + "="*80)
    print("1. MODEL SIZE ANALYSIS")
    print("="*80)
    
    size_mb, total_params, trainable_params = get_model_size(model)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters:      {total_params:,}")
    print(f"   Trainable Parameters:  {trainable_params:,}")
    print(f"   Model Size:            {size_mb:.2f} MB")
    
    # Save checkpoint size
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024**2)
    print(f"   Checkpoint File Size:  {checkpoint_size:.2f} MB")
    
    # 2. FPS Benchmark
    print("\n" + "="*80)
    print("2. INFERENCE SPEED BENCHMARK")
    print("="*80)
    
    print("\n⏱️  Running FPS benchmark...")
    
    # Different image sizes
    test_sizes = [(640, 480), (800, 600), (1024, 768)]
    fps_results = {}
    
    for size in test_sizes:
        fps, avg_time = benchmark_fps(model, device, image_size=size, num_iterations=100)
        fps_results[f"{size[0]}x{size[1]}"] = {
            'fps': fps,
            'time_ms': avg_time * 1000
        }
        print(f"\n   Resolution {size[0]}x{size[1]}:")
        print(f"      FPS: {fps:.2f}")
        print(f"      Average Time: {avg_time*1000:.2f} ms")
    
    # 3. Test Set Evaluation
    print("\n" + "="*80)
    print("3. TEST SET EVALUATION")
    print("="*80)
    
    # Load test data
    with open('./weights_from_scratch/test_indices.json', 'r') as f:
        test_idx = json.load(f)
    
    dataset = PascalVoc('./data', T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)]))
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                            num_workers=2, collate_fn=utils.collate_fn)
    
    print(f"\n🔍 Evaluating on {len(test_idx)} test images...")
    
    coco_evaluator, stats = evaluate(model, test_loader, device)
    
    # Extract metrics
    bbox_stats = stats['bbox']
    
    print("\n📊 Detection Metrics (COCO-style):")
    print(f"\n   mAP @ IoU=0.50:0.95:  {bbox_stats[0]:.4f}")
    print(f"   mAP @ IoU=0.50:       {bbox_stats[1]:.4f}")
    print(f"   mAP @ IoU=0.75:       {bbox_stats[2]:.4f}")
    print(f"   mAP (small):          {bbox_stats[3]:.4f}")
    print(f"   mAP (medium):         {bbox_stats[4]:.4f}")
    print(f"   mAP (large):          {bbox_stats[5]:.4f}")
    
    # Per-class evaluation
    print("\n" + "="*80)
    print("4. PER-CLASS AVERAGE PRECISION")
    print("="*80)
    
    per_class_ap = evaluate_per_class(coco_evaluator)
    
    print(f"\n{'Class':<20} {'AP@0.50:0.95':<15}")
    print("-" * 35)
    
    for idx, (class_name, ap) in enumerate(zip(voc_classes[1:], per_class_ap[1:])):
        print(f"{class_name:<20} {ap:.4f}")
    
    print("-" * 35)
    print(f"{'Mean (all classes)':<20} {sum(per_class_ap[1:]) / len(per_class_ap[1:]):.4f}")
    
    # 5. Generate Report
    print("\n" + "="*80)
    print("5. GENERATING EVALUATION REPORT")
    print("="*80)
    
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
            for class_name, ap in zip(voc_classes[1:], per_class_ap[1:])
        }
    }
    
    # Save report
    report_path = './weights_from_scratch/evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print(" "*25 + "EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Model: Faster R-CNN ResNet-50 FPN (From Scratch)")
    print(f"✅ Parameters: {total_params:,} ({size_mb:.2f} MB)")
    print(f"✅ Best Epoch: {checkpoint['epoch']}")
    print(f"✅ Test mAP@0.50:0.95: {bbox_stats[0]:.4f}")
    print(f"✅ Test mAP@0.50: {bbox_stats[1]:.4f}")
    print(f"✅ Inference FPS (640x480): {fps_results['640x480']['fps']:.2f}")
    print(f"✅ Avg Inference Time: {fps_results['640x480']['time_ms']:.2f} ms")
    
    print("\n" + "="*80)
    print("\n🎉 Evaluation complete! Check evaluation_report.json for full details.\n")


if __name__ == '__main__':
    main()
