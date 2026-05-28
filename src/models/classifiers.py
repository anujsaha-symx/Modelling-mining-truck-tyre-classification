from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torchvision import models

SUPPORTED_CLASSIFIERS = ("efficientnet", "convnext")

@dataclass(frozen=True)
class ClassifierBuildConfig:
    model_name: str
    num_classes: int = 2
    pretrained: bool = True
class TyreWearClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        network: nn.Module,
        head_module: nn.Module,
        gradcam_target_layer: nn.Module,
        class_names: tuple[str, ...],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.network = network
        self.gradcam_target_layer = gradcam_target_layer
        self.class_names = class_names
        self.image_size = image_size
        self._head_param_ids = {id(parameter) for parameter in head_module.parameters()}
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
    def freeze_backbone(self) -> None:
        for parameter in self.network.parameters():
            parameter.requires_grad = id(parameter) in self._head_param_ids
    def unfreeze_backbone(self) -> None:
        for parameter in self.network.parameters():
            parameter.requires_grad = True
    def register_gradcam_hooks(self, forward_hook=None, backward_hook=None) -> list[Any]:
        handles: list[Any] = []
        if forward_hook is not None:
            handles.append(self.gradcam_target_layer.register_forward_hook(forward_hook))
        if backward_hook is not None:
            handles.append(self.gradcam_target_layer.register_full_backward_hook(backward_hook))
        return handles

def _build_efficientnet(config: ClassifierBuildConfig, class_names: tuple[str, ...]) -> TyreWearClassifier:
    weights = models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
    network = models.efficientnet_b0(weights=weights)
    in_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(in_features, config.num_classes)
    return TyreWearClassifier(
        model_name=config.model_name,
        network=network,
        head_module=network.classifier[1],
        gradcam_target_layer=network.features[-1],
        class_names=class_names,
    )
def _build_convnext(config: ClassifierBuildConfig, class_names: tuple[str, ...]) -> TyreWearClassifier:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None
    network = models.convnext_tiny(weights=weights)
    in_features = network.classifier[2].in_features
    network.classifier[2] = nn.Linear(in_features, config.num_classes)
    return TyreWearClassifier(
        model_name=config.model_name,
        network=network,
        head_module=network.classifier[2],
        gradcam_target_layer=network.features[-1],
        class_names=class_names,
    )

def build_classifier(config: ClassifierBuildConfig, class_names: tuple[str, ...]) -> TyreWearClassifier:
    if config.model_name == "efficientnet":
        return _build_efficientnet(config=config, class_names=class_names)
    if config.model_name == "convnext":
        return _build_convnext(config=config, class_names=class_names)
    raise ValueError(f"Unsupported classifier model: {config.model_name}")
def save_classifier_checkpoint(
    destination: Path,
    model: TyreWearClassifier,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float]],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model.model_name,
        "class_names": list(model.class_names),
        "image_size": model.image_size,
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, destination)
def load_classifier_checkpoint(weights_path: Path, device: torch.device | str) -> tuple[TyreWearClassifier, dict[str, Any]]:
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = tuple(checkpoint.get("class_names", ["good", "bad"]))
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint.get("image_size", 224))
    model = build_classifier(
        config=ClassifierBuildConfig(model_name=model_name, num_classes=len(class_names), pretrained=False),
        class_names=class_names,
    )
    model.image_size = image_size
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint