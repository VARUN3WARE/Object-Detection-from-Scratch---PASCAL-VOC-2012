"""
PASCAL VOC dataset loader for object detection.
"""

import os
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset

torch.manual_seed(123)

VOC_CLASSES = (
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

# Backwards compatibility
voc_classes = VOC_CLASSES


class PascalVoc(Dataset):
    """PASCAL VOC dataset for object detection with instance segmentation masks."""
    
    def __init__(self, root: str, transforms):
        self.root = root
        self.transforms = transforms
        
        all_imgs = set([f[:-4] for f in os.listdir(os.path.join(root, 'Images')) if f.endswith('.jpg')])
        all_annot = set([f[:-4] for f in os.listdir(os.path.join(root, 'annotations')) if f.endswith('.xml')])
        all_masks = set([f[:-4] for f in os.listdir(os.path.join(root, 'GT')) if f.endswith('.png')])
        
        valid_ids = all_imgs & all_annot & all_masks
        valid_ids = natsorted(list(valid_ids))
        
        self.imgs = [f'{id}.jpg' for id in valid_ids]
        self.annot = [f'{id}.xml' for id in valid_ids]
        self.masks = [f'{id}.png' for id in valid_ids]
        
        self.class_to_label = {name: i for i, name in enumerate(VOC_CLASSES)}
        self.idx_to_class = {i: name for i, name in enumerate(VOC_CLASSES)}
        
        print(f"Dataset initialized with {len(self.imgs)} valid samples")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = os.path.join(self.root, 'Images', self.imgs[idx])
        annot_path = os.path.join(self.root, 'annotations', self.annot[idx])
        mask_path = os.path.join(self.root, 'GT', self.masks[idx])
        
        img = Image.open(img_path).convert('RGB')

        tree = ET.parse(annot_path)
        root = tree.getroot()

        labels = []
        bboxes = []
        iscrowd = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            difficult = int(obj.find('difficult').text)
            
            boxes = obj.find('bndbox')
            xmin = float(boxes.find('xmin').text) - 1
            ymin = float(boxes.find('ymin').text) - 1
            xmax = float(boxes.find('xmax').text) - 1
            ymax = float(boxes.find('ymax').text) - 1

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_label[class_name])
            iscrowd.append(bool(difficult))

        mask = np.array(Image.open(mask_path))
        
        masks = []
        for i, box in enumerate(bboxes):
            msk_array = np.zeros_like(mask, dtype=np.uint8)
            box = [int(x) for x in box]
            crop = mask[box[1]:box[3], box[0]:box[2]]
            msk_array[box[1]:box[3], box[0]:box[2]] = crop
            msk_array[msk_array == labels[i]] = 1
            msk_array[msk_array == 255] = 1
            msk_array[msk_array != 1] = 0
            masks.append(msk_array)

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        iscrowd = torch.BoolTensor(iscrowd)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': bboxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)

    def get_img_name(self, idx: int) -> str:
        return self.imgs[idx]
