"""
Object Detection Inference Script
Run inference with trained model on images/videos
"""

import os
import json
import argparse
import time
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pascal_dataset import PascalVoc, voc_classes
import utility.transforms as T
import utility.utils as utils


def load_model(checkpoint_path, num_classes=21, device='cuda'):
    """Load trained model from checkpoint"""
    
    # Create model architecture
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val mAP: {checkpoint.get('val_map', 'N/A'):.4f}")
    
    return model


def predict_image(model, image, device, threshold=0.5):
    """Run inference on a single image"""
    
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    img_tensor, _ = transform(image, None)
    
    # Run inference
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
    
    # Filter by threshold
    keep = prediction['scores'] > threshold
    boxes = prediction['boxes'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    
    return boxes, labels, scores


def draw_predictions(image, boxes, labels, scores, class_names=voc_classes):
    """Draw bounding boxes and labels on image"""
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    img = image.copy()
    
    # Colors for different classes (BGR format for cv2)
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
    colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[label % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{class_names[label]}: {score:.2f}'
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(img, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def inference_on_image(model, image_path, output_path, device, threshold=0.5):
    """Run inference on a single image and save result"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run inference
    start_time = time.time()
    boxes, labels, scores = predict_image(model, image, device, threshold)
    inference_time = time.time() - start_time
    
    # Draw predictions
    result_image = draw_predictions(image, boxes, labels, scores)
    
    # Add inference info
    info_text = f"Detections: {len(boxes)} | Time: {inference_time*1000:.1f}ms | FPS: {1/inference_time:.1f}"
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"✓ Saved: {output_path} ({len(boxes)} detections, {inference_time*1000:.1f}ms)")
    
    return result_image, boxes, labels, scores


def inference_on_folder(model, input_folder, output_folder, device, threshold=0.5, max_images=10):
    """Run inference on all images in a folder"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n{'='*60}")
    print(f"Running inference on {len(image_files)} images")
    print(f"{'='*60}\n")
    
    total_time = 0
    total_detections = 0
    
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f'pred_{img_file}')
        
        try:
            image = cv2.imread(input_path)
            start_time = time.time()
            boxes, labels, scores = predict_image(model, image, device, threshold)
            inference_time = time.time() - start_time
            
            result_image = draw_predictions(image, boxes, labels, scores)
            cv2.imwrite(output_path, result_image)
            
            total_time += inference_time
            total_detections += len(boxes)
            
            print(f"  {img_file}: {len(boxes)} detections, {inference_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"  Error processing {img_file}: {e}")
    
    avg_time = total_time / len(image_files) if image_files else 0
    avg_fps = 1 / avg_time if avg_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"  Average time: {avg_time*1000:.1f}ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Results saved to: {output_folder}")
    print(f"{'='*60}\n")


def inference_on_video(model, video_path, output_path, device, threshold=0.5, max_frames=None):
    """Run inference on video and save result"""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"  Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")
    print(f"{'='*60}\n")
    
    frame_count = 0
    total_inference_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # Run inference
        start_time = time.time()
        boxes, labels, scores = predict_image(model, frame, device, threshold)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Draw predictions
        result_frame = draw_predictions(frame, boxes, labels, scores)
        
        # Add info
        info_text = f"Frame: {frame_count+1}/{total_frames} | FPS: {1/inference_time:.1f} | Detections: {len(boxes)}"
        cv2.putText(result_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(result_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
    avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Video processing complete!")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Output saved to: {output_path}")
    print(f"{'='*60}\n")


def inference_on_test_set(model, device, threshold=0.5, num_samples=8):
    """Run inference on test dataset and create visualization grid"""
    
    # Load test indices
    weights_dir = './weights_from_scratch'
    if not os.path.exists(f'{weights_dir}/test_indices.json'):
        print("Test indices not found. Run training first.")
        return
    
    with open(f'{weights_dir}/test_indices.json', 'r') as f:
        test_idx = json.load(f)
    
    # Load dataset
    dataset = PascalVoc('./data', T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)]))
    test_dataset = torch.utils.data.Subset(dataset, test_idx[:num_samples])
    
    print(f"\n{'='*60}")
    print(f"Running inference on {num_samples} test images")
    print(f"{'='*60}\n")
    
    # Create figure
    rows = (num_samples + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img_tensor, target = test_dataset[i]
        
        # Run inference
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        
        # Filter by threshold
        keep = prediction['scores'] > threshold
        boxes = prediction['boxes'][keep].cpu().numpy()
        labels = prediction['labels'][keep].cpu().numpy()
        scores = prediction['scores'][keep].cpu().numpy()
        
        # Convert to displayable image
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Draw predictions
        axes[i].imshow(img)
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(x1, y1-5, f'{voc_classes[label]}: {score:.2f}',
                        color='white', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        img_name = dataset.dataset.get_img_name(test_idx[i])
        axes[i].set_title(f'{img_name} ({len(boxes)} detections)', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = './inference_results/test_set_predictions.png'
    os.makedirs('./inference_results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Test set predictions saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--checkpoint', type=str, default='./weights_from_scratch/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--folder', type=str, help='Path to input folder')
    parser.add_argument('--output', type=str, default='./inference_results',
                       help='Path to output folder')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--test-set', action='store_true',
                       help='Run inference on test dataset')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}\n")
    
    model = load_model(args.checkpoint, device=args.device)
    
    # Run inference based on mode
    if args.test_set:
        inference_on_test_set(model, args.device, args.threshold)
    
    elif args.image:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, 'prediction_' + os.path.basename(args.image))
        inference_on_image(model, args.image, output_path, args.device, args.threshold)
        print(f"\n✅ Inference complete! Check: {output_path}\n")
    
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder {args.folder} does not exist")
            return
        inference_on_folder(model, args.folder, args.output, args.device, args.threshold)
    
    elif args.video:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, 'prediction_' + os.path.basename(args.video))
        inference_on_video(model, args.video, output_path, args.device, args.threshold)
        print(f"\n✅ Video inference complete! Check: {output_path}\n")
    
    else:
        # Default: run on test set
        print("No input specified. Running on test dataset...\n")
        inference_on_test_set(model, args.device, args.threshold)
        print("\n✅ Use --image, --video, or --folder for custom inference\n")


if __name__ == '__main__':
    main()
