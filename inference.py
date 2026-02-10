"""
Object detection inference script for images and videos.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from logger import setup_logger
from pascal_dataset import PascalVoc, VOC_CLASSES
import utility.transforms as T

logger = setup_logger("inference")


def load_model(checkpoint_path: str, num_classes: int = 21, device: str = 'cuda') -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}, Val mAP: {checkpoint.get('val_map', 0):.4f}")
    
    return model


def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a single image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    img_tensor, _ = transform(image, None)
    
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
    
    keep = prediction['scores'] > threshold
    boxes = prediction['boxes'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    
    return boxes, labels, scores


def draw_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    img = image.copy()
    colors = plt.cm.hsv(np.linspace(0, 1, len(VOC_CLASSES))).tolist()
    colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[label % len(colors)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f'{VOC_CLASSES[label]}: {score:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def inference_on_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: str,
    threshold: float = 0.5
):
    """Run inference on a single image and save result."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    start_time = time.time()
    boxes, labels, scores = predict_image(model, image, device, threshold)
    inference_time = time.time() - start_time
    
    result_image = draw_predictions(image, boxes, labels, scores)
    
    info_text = f"Detections: {len(boxes)} | Time: {inference_time*1000:.1f}ms | FPS: {1/inference_time:.1f}"
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, result_image)
    logger.info(f"Saved: {output_path} ({len(boxes)} detections, {inference_time*1000:.1f}ms)")


def inference_on_folder(
    model: torch.nn.Module,
    input_folder: str,
    output_folder: str,
    device: str,
    threshold: float = 0.5,
    max_images: int = 10
):
    """Run inference on all images in a folder."""
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Running inference on {len(image_files)} images")
    
    total_time = 0
    total_detections = 0
    
    for img_file in image_files:
        output_path = output_dir / f'pred_{img_file.name}'
        
        try:
            image = cv2.imread(str(img_file))
            start_time = time.time()
            boxes, labels, scores = predict_image(model, image, device, threshold)
            inference_time = time.time() - start_time
            
            result_image = draw_predictions(image, boxes, labels, scores)
            cv2.imwrite(str(output_path), result_image)
            
            total_time += inference_time
            total_detections += len(boxes)
            
            logger.info(f"  {img_file.name}: {len(boxes)} detections, {inference_time*1000:.1f}ms")
        except Exception as e:
            logger.error(f"  Error processing {img_file.name}: {e}")
    
    if image_files:
        avg_time = total_time / len(image_files)
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        logger.info(f"Average time: {avg_time*1000:.1f}ms, FPS: {avg_fps:.1f}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Results saved to: {output_folder}")


def inference_on_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: str,
    threshold: float = 0.5,
    max_frames: Optional[int] = None
):
    """Run inference on video and save result."""
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"  Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")
    
    frame_count = 0
    total_inference_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        start_time = time.time()
        boxes, labels, scores = predict_image(model, frame, device, threshold)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        result_frame = draw_predictions(frame, boxes, labels, scores)
        
        info_text = f"Frame: {frame_count+1}/{total_frames} | FPS: {1/inference_time:.1f} | Detections: {len(boxes)}"
        cv2.putText(result_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(result_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            logger.info(f"  Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    if frame_count > 0:
        avg_inference_time = total_inference_time / frame_count
        avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        logger.info(f"Average inference time: {avg_inference_time*1000:.1f}ms")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Output saved to: {output_path}")


def inference_on_test_set(
    model: torch.nn.Module,
    device: str,
    threshold: float = 0.5,
    num_samples: int = 8
):
    """Run inference on test dataset and create visualization grid."""
    weights_dir = Path('./weights_from_scratch')
    test_indices_file = weights_dir / 'test_indices.json'
    
    if not test_indices_file.exists():
        logger.error("Test indices not found. Run training first.")
        return
    
    with open(test_indices_file, 'r') as f:
        test_idx = json.load(f)
    
    dataset = PascalVoc('./data', T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)]))
    test_dataset = torch.utils.data.Subset(dataset, test_idx[:num_samples])
    
    logger.info(f"Running inference on {num_samples} test images")
    
    rows = (num_samples + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img_tensor, target = test_dataset[i]
        
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        
        keep = prediction['scores'] > threshold
        boxes = prediction['boxes'][keep].cpu().numpy()
        labels = prediction['labels'][keep].cpu().numpy()
        scores = prediction['scores'][keep].cpu().numpy()
        
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        axes[i].imshow(img)
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(x1, y1-5, f'{VOC_CLASSES[label]}: {score:.2f}',
                        color='white', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        img_name = dataset.dataset.get_img_name(test_idx[i])
        axes[i].set_title(f'{img_name} ({len(boxes)} detections)', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = Path('./inference_results/test_set_predictions.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Test set predictions saved to: {output_path}")


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
    
    logger.info("Loading Model")
    model = load_model(args.checkpoint, device=args.device)
    
    if args.test_set:
        inference_on_test_set(model, args.device, args.threshold)
    elif args.image:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'prediction_{Path(args.image).name}'
        inference_on_image(model, args.image, str(output_path), args.device, args.threshold)
        logger.info(f"Inference complete! Check: {output_path}")
    elif args.folder:
        if not Path(args.folder).exists():
            logger.error(f"Folder {args.folder} does not exist")
            return
        inference_on_folder(model, args.folder, args.output, args.device, args.threshold)
    elif args.video:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'prediction_{Path(args.video).name}'
        inference_on_video(model, args.video, str(output_path), args.device, args.threshold)
        logger.info(f"Video inference complete! Check: {output_path}")
    else:
        logger.info("No input specified. Running on test dataset...")
        inference_on_test_set(model, args.device, args.threshold)
        logger.info("Use --image, --video, or --folder for custom inference")


if __name__ == '__main__':
    main()
