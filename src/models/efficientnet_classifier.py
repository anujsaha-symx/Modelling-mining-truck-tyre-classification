from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_efficientnet_b0(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)

    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable
