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
from src.models.efficientnet_classifier import create_efficientnet_b0
from src.utils.common import CHECKPOINTS_ROOT, IDX_TO_CLASS, VERIFICATION_CHECKPOINTS_ROOT, get_device
from src.verification.verify import verify_image


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_efficientnet_b0(
        num_classes=checkpoint.get("num_classes", 2),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(image_path: Path, checkpoint_path: Path) -> tuple[str, float]:
    device = get_device()
    model = load_model(checkpoint_path, device)
    transform = build_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_idx].item())

    return IDX_TO_CLASS[pred_idx].upper(), confidence


def predict_with_verification(
    image_path: Path,
    classifier_checkpoint: Path,
    verifier_checkpoint: Path,
    verifier_threshold: float | None = None,
) -> dict:
    verification = verify_image(image_path, verifier_checkpoint, threshold=verifier_threshold)
    result = {
        "is_tyre": verification["is_tyre"],
        "verification_confidence": verification["confidence"],
    }

    if not verification["is_tyre"]:
        result["message"] = "No tyre detected"
        return result

    prediction, confidence = predict_image(image_path, classifier_checkpoint)
    result["prediction"] = prediction
    result["confidence"] = confidence
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tyre condition from a single image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_ROOT / "best_model.pt")
    parser.add_argument("--verifier-checkpoint", type=Path, default=VERIFICATION_CHECKPOINTS_ROOT / "best_verifier.pt")
    parser.add_argument("--verifier-threshold", type=float, default=None)
    args = parser.parse_args()

    result = predict_with_verification(
        args.image,
        classifier_checkpoint=args.checkpoint,
        verifier_checkpoint=args.verifier_checkpoint,
        verifier_threshold=args.verifier_threshold,
    )

    if not result["is_tyre"]:
        print("No tyre detected")
        print(json.dumps({"is_tyre": False, "confidence": round(result["verification_confidence"], 6)}, indent=2))
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
