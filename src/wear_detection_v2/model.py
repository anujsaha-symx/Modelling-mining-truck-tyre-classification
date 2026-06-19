import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 4
CLASS_NAMES = {1: 'Tire', 2: 'Cut', 3: 'Non-Tire'}

def get_frcnn_model_v2(num_classes=NUM_CLASSES, pretrained=True):
    weights = 'DEFAULT' if pretrained else None
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model