from __future__ import annotations
import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.classification_dataset import build_eval_transforms  # noqa: E402
from src.models.classifiers import load_classifier_checkpoint  # noqa: E402

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run single-image tyre wear classification inference.")
    parser.add_argument("image", help="Image path for classification.")
    parser.add_argument("--weights", required=True, help="Path to classifier checkpoint.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()

def run(args: Namespace | None = None) -> dict[str, object]:
    args = args or parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output) if args.output else weights_path.resolve().parents[1] / "single_image_prediction.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_classifier_checkpoint(weights_path=weights_path, device=device)
    transform = build_eval_transforms(image_size=int(checkpoint.get("image_size", 224)))
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(probabilities.argmax())
    class_names = checkpoint.get("class_names", ["good", "bad"])
    result = {
        "image_path": str(Path(args.image).resolve()),
        "predicted_label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {class_names[index]: float(value) for index, value in enumerate(probabilities.tolist())},
        "model_name": checkpoint.get("model_name"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    run()