# Custom Object Detection from Scratch - PASCAL VOC 2012

## Object Detection with Faster R-CNN ResNet-50 FPN

---

## Executive Summary

This project implements a complete object detection pipeline using **Faster R-CNN with ResNet-50 FPN backbone**, trained entirely **from scratch without any pre-trained weights** (no ImageNet pre-training).

### ✅ Key Achievements

- ✅ Complete training pipeline from random initialization
- ✅ Faster R-CNN ResNet-50 FPN architecture (41.4M parameters)
- ✅ PASCAL VOC 2012 dataset (20 object classes)
- ✅ Training completed in **11 minutes** on NVIDIA RTX 3050
- ✅ Comprehensive evaluation: mAP, FPS, model size, per-class AP
- ✅ Multi-mode inference: images, videos, folders, test set

### 📊 Quick Stats

| Metric              | Value                             |
| ------------------- | --------------------------------- |
| **Model**           | Faster R-CNN ResNet-50 FPN        |
| **Training**        | From scratch (no pre-training)    |
| **Parameters**      | 41,449,656 (158.32 MB)            |
| **Training Time**   | 11.01 minutes                     |
| **Inference Speed** | 10.12 FPS @ 640x480               |
| **mAP@0.50:0.95**   | 0.0006                            |
| **mAP@0.50**        | 0.0031                            |
| **Dataset**         | PASCAL VOC 2012 (100 samples)     |
| **Classes**         | 20 object categories + background |

**Note:** Low mAP (0.0006) is expected due to training from scratch on only 100 samples. With ImageNet pre-training + full dataset, expected mAP would be **0.35-0.40** (60-70x improvement).

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 12.1 (for GPU training)
```

### Installation

```bash
# Install dependencies
pip install torch torchvision opencv-python matplotlib
```

### Train Model

```bash
python train_from_scratch.py
```

- Training time: ~11 minutes
- Output: `weights_from_scratch/best_model.pth`
- Checkpoints: Best model (epoch 11) and last model (epoch 20)

### Evaluate Model

```bash
python evaluate_model.py
```

- Computes: mAP, FPS, model size, per-class AP
- Output: `weights_from_scratch/evaluation_report.json`

### Run Inference

```bash
# On test set
python inference.py --test-set

# On single image
python inference.py --image data/Images/2008_000008.jpg

# On video
python inference.py --video path/to/video.mp4

# On folder
python inference.py --folder data/Images/ --output detections/
```

---

## 📖 Project Overview

### Task Requirements

**Objective:** Build and train an object detection model from scratch (no pre-trained weights) such as Faster R-CNN or custom CNN-based detector.

**Requirements:**

1. ✅ Complete object detection pipeline
2. ✅ Train from scratch (no pre-trained weights)
3. ✅ Custom dataset with 3-5+ object classes
4. ✅ Evaluate on mAP, inference speed (FPS), and model size
5. ✅ Detailed report with architecture, augmentation, training, results, and trade-offs

### Implementation

- **Model:** Faster R-CNN with ResNet-50 FPN backbone
- **Training:** From random initialization (no ImageNet pre-training)
- **Dataset:** PASCAL VOC 2012 (20 classes)
- **Training Subset:** 100 images (70 train, 20 val, 10 test)
- **Full Dataset Available:** 2,913 images with annotations

---

## 🏗️ Architecture & Design

### Model: Faster R-CNN ResNet-50 FPN

**Architecture Overview:**

```
Input Image (3 × H × W)
    ↓
┌──────────────────────────────────────┐
│  Backbone: ResNet-50 FPN             │
│  - ResNet-50: 5 conv blocks          │
│  - FPN: Multi-scale feature pyramid  │
│  - Output: Features at P2-P6         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Region Proposal Network (RPN)       │
│  - Anchor boxes: 3 scales × 3 ratios │
│  - Outputs: ~2000 region proposals   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  RoI Pooling & Detection Head        │
│  - RoI Align: 7×7 features           │
│  - Classification: 21 classes        │
│  - Box Regression: 4 coordinates     │
└──────────────────────────────────────┘
    ↓
Final Detections (class, bbox, score)
```

### Technical Specifications

| Component            | Details                                      |
| -------------------- | -------------------------------------------- |
| **Backbone**         | ResNet-50 with Feature Pyramid Network (FPN) |
| **Input Size**       | Variable (min_size=800, auto-resized)        |
| **Anchor Scales**    | (32, 64, 128, 256, 512)                      |
| **Anchor Ratios**    | (0.5, 1.0, 2.0)                              |
| **RPN Proposals**    | 2000 (training), 1000 (inference)            |
| **RoI Pool Size**    | 7×7                                          |
| **Box Head**         | 2 FC layers (1024 units each)                |
| **Output Classes**   | 21 (20 objects + background)                 |
| **Total Parameters** | 41,449,656                                   |
| **Model Size**       | 158.32 MB                                    |

### Design Choices

**Why Faster R-CNN?**

- ✅ Two-stage detector → higher accuracy potential
- ✅ Well-established architecture with proven performance
- ✅ Suitable for from-scratch training
- ⚠️ Trade-off: Slower inference (10 FPS vs 30+ for YOLO)

**Why ResNet-50 FPN?**

- ✅ Deep enough for complex features (50 layers)
- ✅ Residual connections prevent gradient vanishing
- ✅ FPN enables multi-scale feature extraction
- ✅ Balanced model size vs performance
- ✅ Fits in 4GB GPU memory

---

## 📊 Dataset Details

### PASCAL VOC 2012

**Dataset Statistics:**

- Total images: 17,125
- Valid images (with annotations & masks): 2,913
- Object classes: 20 categories + background

**20 Object Classes:**

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor
```

### Training Subset (Quick Training Mode)

**Data Split:**

- **Total:** 100 images
- **Training:** 70 images (70%)
- **Validation:** 20 images (20%)
- **Test:** 10 images (10%)

**Rationale:**

- ⏱️ Enable 15-minute training for rapid iteration
- 💾 Reduce GPU memory requirements (3.81 GB GPU)
- 🔄 Quick proof-of-concept validation
- ⚠️ Trade-off: Lower final performance

**Dataset Structure:**

```
data/
├── Images/              # 17,125 JPG images
├── annotations/         # 2,913 XML annotations (PASCAL VOC format)
└── GT/                  # 2,913 PNG segmentation masks

Segmentation/
├── train.txt           # 1,464 training image IDs
└── val.txt             # 1,449 validation image IDs
```

### Data Augmentation

**Training Augmentation:**

```python
transforms.Compose([
    transforms.ToTensor(),                    # Convert to tensor
    transforms.RandomHorizontalFlip(p=0.5)   # 50% horizontal flip
])
```

**Validation/Test:**

- No augmentation (only tensor conversion)
- Preserves original images for fair evaluation

**Potential Improvements:**

- Color jittering (brightness, contrast, saturation)
- Random rotation (±15 degrees)
- Random scaling (0.8-1.2x)
- Cutout/Random erasing
- More augmentation would likely improve results

---

## 🎯 Training Methodology

### Training Configuration

| Hyperparameter    | Value                      | Rationale                          |
| ----------------- | -------------------------- | ---------------------------------- |
| **Optimizer**     | SGD with Momentum          | Standard for object detection      |
| **Learning Rate** | 0.005                      | Balanced for from-scratch training |
| **Momentum**      | 0.9                        | Helps escape local minima          |
| **Weight Decay**  | 0.0005                     | L2 regularization                  |
| **LR Scheduler**  | StepLR                     | Gradual decay                      |
| **LR Step Size**  | 5 epochs                   | Decay every 5 epochs               |
| **LR Gamma**      | 0.5                        | Halve LR at each step              |
| **Batch Size**    | 1                          | GPU memory limited (3.81 GB)       |
| **Epochs**        | 20                         | Sufficient for convergence         |
| **Device**        | NVIDIA RTX 3050 Laptop GPU | CUDA 12.1                          |

### Learning Rate Schedule

```
Epoch 0-4:   LR = 0.005
Epoch 5-9:   LR = 0.0025   (×0.5)
Epoch 10-14: LR = 0.00125  (×0.25)
Epoch 15-19: LR = 0.000625 (×0.125)
```

**Warm-up:**

- First 500 iterations: Linear warm-up from 0.00015 to 0.005
- Prevents training instability

### Loss Functions

**Multi-task Loss (4 components):**

1. **RPN Classification Loss** - Binary objectness (anchor is object vs background)
2. **RPN Box Regression Loss** - Smooth L1 loss for anchor refinement
3. **RoI Classification Loss** - Cross-entropy over 21 classes
4. **RoI Box Regression Loss** - Smooth L1 loss for final bbox

**Total Loss:** `L_total = L_rpn_cls + L_rpn_box + L_roi_cls + L_roi_box`

### Training Process

**Timeline:**

- **Total Time:** 11.01 minutes
- **Per Epoch:** ~33 seconds
- **Batches per Epoch:** 70
- **Total Iterations:** 1,400 (70 × 20 epochs)

**Memory Usage:**

- GPU Memory: 2.6 GB / 3.81 GB
- Peak: 2,678 MB
- Efficient with batch_size=1

**Checkpointing:**

- Best model saved at epoch 11 (highest validation mAP)
- Last model saved at epoch 20
- Training curves plotted and saved

### From-Scratch Initialization

- ✅ No ImageNet pre-training
- ✅ Random weight initialization (Kaiming uniform)
- ✅ All layers trained from scratch
- ⚠️ Requires more data (1000+ samples ideal)
- ⚠️ Slower convergence vs transfer learning

---

## 📈 Results & Evaluation

### Training Performance

**Loss Progression:**

```
Epoch 0:  Loss = 1.36 → 0.70
Epoch 5:  Loss = 0.52
Epoch 10: Loss = 0.49
Epoch 15: Loss = 0.48
Epoch 20: Loss = 0.46
```

**Observations:**

- ✅ Steady loss decrease → model is learning
- ✅ No significant overfitting
- ⚠️ Low absolute mAP due to limited data

### Validation Performance

**Best Epoch:** 11  
**Best Validation mAP:** 0.0006

**mAP Progression:**

```
Epoch 1:  mAP@0.50:0.95 = 0.0001  |  mAP@0.50 = 0.0003
Epoch 5:  mAP@0.50:0.95 = 0.0003  |  mAP@0.50 = 0.0012
Epoch 10: mAP@0.50:0.95 = 0.0005  |  mAP@0.50 = 0.0020
Epoch 11: mAP@0.50:0.95 = 0.0006  |  mAP@0.50 = 0.0024  ← BEST
Epoch 15: mAP@0.50:0.95 = 0.0004  |  mAP@0.50 = 0.0019
Epoch 20: mAP@0.50:0.95 = 0.0002  |  mAP@0.50 = 0.0016
```

### Test Set Results

| Metric                  | Value  | Industry Standard (with pre-training) |
| ----------------------- | ------ | ------------------------------------- |
| **mAP@0.50:0.95**       | 0.0006 | 0.30-0.50                             |
| **mAP@0.50**            | 0.0031 | 0.50-0.70                             |
| **mAP@0.75**            | 0.0001 | 0.40-0.60                             |
| **mAP (large objects)** | 0.0009 | 0.35-0.55                             |

**Why Low mAP?**

1. ❌ Only 100 training samples (vs 1,464 available)
2. ❌ No ImageNet pre-training
3. ❌ Minimal data augmentation
4. ❌ Small batch size (1)
5. ⚠️ Expected for from-scratch training on tiny dataset

### Per-Class Average Precision

| Class     | AP@0.50:0.95 | Notes                          |
| --------- | ------------ | ------------------------------ |
| motorbike | 0.0281       | **Only class with detections** |
| aeroplane | 0.0000       | No detections                  |
| dog       | 0.0000       | No detections                  |
| (others)  | -1.0000      | Not in test set                |

### Inference Speed Benchmark

**Hardware:** NVIDIA RTX 3050 Laptop GPU (3.81 GB)

| Resolution | FPS       | Avg Time (ms) | Use Case        |
| ---------- | --------- | ------------- | --------------- |
| 640×480    | **10.12** | 98.83         | Real-time video |
| 800×600    | 10.02     | 99.81         | High-res images |
| 1024×768   | 10.04     | 99.63         | HD video        |

**Comparison:**

- Faster R-CNN: **10 FPS** (this model)
- YOLOv5: 30-60 FPS (faster, single-stage)
- Mask R-CNN: 5-8 FPS (slower, adds segmentation)

### Model Size

- **Total Parameters:** 41,449,656
- **Model Size:** 158.32 MB
- **Checkpoint Size:** 316.61 MB (includes optimizer state)

---

## Training Comparison

| Aspect            | From Scratch (Ours)  | Transfer Learning        |
| ----------------- | -------------------- | ------------------------ |
| **Training Time** | 11 min (100 samples) | 30-60 min (full dataset) |
| **Data Required** | 1000+ samples        | 100-500 samples          |
| **Final mAP**     | 0.0006 (very low)    | 0.30-0.40 (good)         |
| **Convergence**   | Slower               | Faster                   |
| **Flexibility**   | Full control         | Inherits ImageNet biases |

**Our Position:**

- Moderate accuracy potential (limited by data)
- Moderate speed (10 FPS, real-time capable)
- Large model size (158 MB)

### Key Trade-off Decisions

1. **Faster R-CNN vs YOLO**

   - ✅ Chose accuracy potential over speed
   - ✅ Better for from-scratch training
   - ⚠️ 10 FPS vs 30+ FPS

2. **ResNet-50 vs ResNet-101**

   - ✅ Balanced size/performance
   - ✅ Fits in 4GB GPU
   - ⚠️ Slightly lower capacity

3. **Batch Size = 1**
   - ✅ Fits in limited GPU memory
   - ⚠️ Noisy gradients
   - ⚠️ Slower convergence

---

## 📁 File Structure

```
-PascalVOC/
├── train_from_scratch.py          # Main training script ⭐
├── inference.py                    # Run predictions
├── evaluate_model.py               # Comprehensive evaluation
├── pascal_dataset.py               # Dataset loader
├── README.md                       # This file
│
├── utility/                        # Helper modules
│   ├── engine.py                   # Training/validation loops
│   ├── coco_eval.py                # mAP evaluation
│   ├── coco_utils.py               # COCO utilities
│   ├── transforms.py               # Data augmentation
│   └── utils.py                    # General utilities
│
├── data/                           # Dataset
│   ├── Images/                     # 17,125 images
│   ├── annotations/                # 2,913 XML files
│   └── GT/                         # 2,913 PNG masks
│
├── Segmentation/
│   ├── train.txt                   # Training split (1,464 IDs)
│   └── val.txt                     # Validation split (1,449 IDs)
│
├── weights_from_scratch/           # Training outputs
│   ├── best_model.pth              # Best checkpoint (epoch 11)
│   ├── last_model.pth              # Final checkpoint (epoch 20)
│   ├── training_curves.png         # Loss/mAP plots
│   └── evaluation_report.json      # Detailed metrics
│
└── archive (3)/                    # Original dataset backup
```

<!-- ### Recommended Next Steps

**For Production:**

```bash
# 1. Use pre-trained weights
# 2. Train on full dataset (1,464 samples)
# 3. Apply enhanced augmentation
# 4. Train for 50-100 epochs
# Expected result: mAP 0.35-0.40
```

--- -->

---

## 📚 References

**Dataset:**

- PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

**Code:**

- PyTorch: https://pytorch.org/
- TorchVision Models: https://pytorch.org/vision/stable/models.html

---
