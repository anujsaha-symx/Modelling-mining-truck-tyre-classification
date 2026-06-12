from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.transforms import build_eval_transforms
from src.utils.common import VERIFICATION_CHECKPOINTS_ROOT, get_device


LABELS = {0: "non_tyre", 1: "tyre"}


def create_verifier_model(architecture: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True):
    from torchvision.models import (
        EfficientNet_B0_Weights,
        MobileNet_V3_Small_Weights,
        efficientnet_b0,
        mobilenet_v3_small,
    )
    import torch.nn as nn

    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_verifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_verifier_model(
        architecture=checkpoint.get("architecture", "efficientnet_b0"),
        num_classes=checkpoint.get("num_classes", 2),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def verify_image(image_path: Path, checkpoint_path: Path, threshold: float | None = None) -> dict:
    device = get_device()
    model, checkpoint = load_verifier(checkpoint_path, device)
    decision_threshold = threshold if threshold is not None else float(checkpoint.get("threshold", 0.5))
    transform = build_eval_transforms()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        tyre_confidence = float(probabilities[1].item())
        is_tyre = tyre_confidence >= decision_threshold

    return {"is_tyre": bool(is_tyre), "confidence": tyre_confidence, "threshold": decision_threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether an image contains a tyre.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=VERIFICATION_CHECKPOINTS_ROOT / "best_verifier.pt")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    result = verify_image(args.image, args.checkpoint, threshold=args.threshold)
    print(json.dumps({"is_tyre": result["is_tyre"], "confidence": round(result["confidence"], 6)}, indent=2))


if __name__ == "__main__":
    main()
