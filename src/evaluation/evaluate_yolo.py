from __future__ import annotations
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.yolo_detector import evaluate_yolo_detector, generate_sample_predictions

YOLO_ROOT = PROJECT_ROOT / "datasets" / "yolo"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "yolo_detection"
CHECKPOINTS_ROOT = PROJECT_ROOT / "outputs" / "checkpoints" / "yolo"

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate trained YOLO tyre detector on test split.")
    parser.add_argument("--weights", default=str(CHECKPOINTS_ROOT / "best.pt"), help="Path to trained YOLO weights.")
    parser.add_argument("--data", default=str(YOLO_ROOT / "data.yaml"), help="Path to YOLO data.yaml.")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample predictions to export.")
    return parser.parse_args()

def run(args: Namespace | None = None) -> None:
    args = args or parse_args()
    data_yaml = Path(args.data)
    weights_path = Path(args.weights)
    metrics = evaluate_yolo_detector(weights_path=weights_path, data_yaml=data_yaml, output_root=OUTPUT_ROOT)
    generate_sample_predictions(
        weights_path=weights_path,
        source_dir=YOLO_ROOT / "images" / "test",
        output_root=OUTPUT_ROOT,
        limit=args.samples,
    )
    print(f"YOLO evaluation complete: {metrics}")

if __name__ == "__main__":
    run()