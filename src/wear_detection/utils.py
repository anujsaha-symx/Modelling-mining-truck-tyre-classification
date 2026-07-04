import json
import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 4
CLASS_NAMES = {1: 'Good-Tire', 2: 'Bad-Tire', 3: 'Non-Tire'}
OLD_ROOT = 'D:\\Modelling-mining-truck-tyre-classification'
CURRENT_ROOT = 'D:\\Tyre_Classification'


def collate_fn(batch):
    return tuple(zip(*batch))


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_checkpoint(model, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def get_model(num_classes=NUM_CLASSES, freeze_backbone=False):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights='DEFAULT'
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    return T.Compose(transforms)


def fix_path(path):
    return path.replace(OLD_ROOT, CURRENT_ROOT)


class WearDetectionDataset(Dataset):
    def __init__(self, coco_path, split_csv, transforms=None):
        self.transforms = transforms or T.Compose([T.ToTensor()])

        with open(coco_path, 'r') as f:
            self.coco = json.load(f)

        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.basename_to_img = {}
        for img in self.coco['images']:
            fname = os.path.basename(img['file_name'])
            self.basename_to_img[fname] = img

        split_df = pd.read_csv(split_csv)
        self.samples = []
        missing = 0
        for _, row in split_df.iterrows():
            fname = os.path.basename(row['filepath'])
            if fname in self.basename_to_img:
                self.samples.append({
                    'img_path': fix_path(row['filepath']),
                    'img_info': self.basename_to_img[fname],
                })
            else:
                missing += 1
        if missing:
            print(f'Warning: {missing} images in split not found in COCO annotations')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        img_info = sample['img_info']

        image = Image.open(img_path).convert('RGB')
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] + 1)
            area.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'area': area,
            'iscrowd': iscrowd,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
