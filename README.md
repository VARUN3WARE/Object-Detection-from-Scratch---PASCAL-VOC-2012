# Object Detection from Scratch - PASCAL VOC 2012

Production-ready object detection pipeline using Faster R-CNN ResNet-50 FPN, trained entirely from scratch without pre-trained weights.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Evaluate model
python evaluate.py

# Run inference
python inference.py --test-set
```

## Performance

| Metric          | Value                      |
| --------------- | -------------------------- |
| Architecture    | Faster R-CNN ResNet-50 FPN |
| Parameters      | 41.4M (158 MB)             |
| Training Time   | 11 min (100 samples)       |
| Inference Speed | 10 FPS @ 640x480           |
| mAP@0.50:0.95   | 0.0006\*                   |
| Classes         | 20 object categories       |

\*Low mAP due to training from scratch on limited data. With pre-training + full dataset: 0.35-0.40 expected.

## Project Structure

```
Image-Processing-PascalVOC/
├── train.py                 # Training script
├── evaluate.py              # Model evaluation
├── inference.py             # Inference
├── pascal_dataset.py        # Dataset loader
├── config.py                # Configuration
├── logger.py                # Logging setup
├── requirements.txt         # Dependencies
├── utility/                 # Helper modules
│   ├── engine.py           # Train/eval loops
│   ├── coco_eval.py        # mAP metrics
│   └── transforms.py       # Augmentation
├── data/                    # PASCAL VOC 2012
│   ├── Images/             # 17,125 images
│   ├── annotations/        # XML annotations
│   └── GT/                 # Segmentation masks
└── weights_from_scratch/   # Outputs
```

## Usage

### Training

```bash
python train.py
```

Outputs:

- `weights_from_scratch/best_model.pth` - Best checkpoint
- `weights_from_scratch/training_curves.png` - Loss/mAP plots
- `weights_from_scratch/training_history.json` - Metrics

### Evaluation

```bash
python evaluate.py
```

Computes:

- Model size and parameters
- Inference FPS at multiple resolutions
- COCO-style mAP metrics
- Per-class average precision

### Inference

```bash
# Test set
python inference.py --test-set

# Single image
python inference.py --image path/to/image.jpg --output results/

# Video
python inference.py --video path/to/video.mp4 --output results/

# Folder
python inference.py --folder path/to/images/ --output results/
```

## Configuration

Customize training in `config.py`:

```python
from config import Config

config = Config()
config.training.epochs = 50
config.training.batch_size = 2
config.data.max_samples = None  # Use full dataset
```

## Architecture

**Model:** Faster R-CNN with ResNet-50 FPN backbone

Key specifications:

- Input: Variable size (min 800px)
- Anchor scales: 32, 64, 128, 256, 512
- RPN proposals: 2000 (train), 1000 (inference)
- RoI pooling: 7x7
- Output: 21 classes (20 objects + background)

**Objects classes:**
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

## Training Details

### Hyperparameters

- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.005 with StepLR decay
- Weight decay: 0.0005
- Batch size: 1
- Epochs: 20

### Data Augmentation

- Random horizontal flip (50%)
- Tensor normalization

### From-Scratch Training

- No ImageNet pre-training
- Random weight initialization
- Requires more data for optimal performance

## Dataset

**PASCAL VOC 2012**

- Total images: 17,125
- Valid samples (with annotations): 2,913
- Default split: 70% train, 20% val, 10% test
- Quick mode: 100 samples for rapid iteration

## Logging

All scripts use structured logging:

```python
from logger import setup_logger
logger = setup_logger("module_name")
logger.info("Message")
```

Logs include timestamps, module names, and severity levels.

## Development

### Code Standards

- Type hints for all functions
- Structured logging (no print statements)
- Centralized configuration
- PEP 8 compliant

### Make Commands

```bash
make install    # Install dependencies
make train      # Train model
make evaluate   # Evaluate model
make inference  # Run inference
make clean      # Remove cache/outputs
```

## Performance Optimization

To improve performance:

1. **Use pre-trained weights:**

   ```python
   model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
   ```

2. **Train on full dataset:**

   ```python
   config.data.max_samples = None
   ```

3. **Increase training epochs:**

   ```python
   config.training.epochs = 50
   ```

4. **Add more augmentation:**
   - Color jittering
   - Random rotation
   - Random scaling

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.1+ (GPU training)
- 4GB+ GPU memory

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/)

## License

MIT License - See LICENSE file for details.
