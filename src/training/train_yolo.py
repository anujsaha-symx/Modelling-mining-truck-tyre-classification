from __future__ import annotations
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.prepare_yolo_dataset import prepare_yolo_dataset
from src.models.yolo_detector import YoloTrainingConfig, evaluate_yolo_detector, generate_sample_predictions, train_yolo_detector

YOLO_ROOT = PROJECT_ROOT / "datasets" / "yolo"
SPLITS_ROOT = PROJECT_ROOT / "datasets" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "yolo_detection"
CHECKPOINTS_ROOT = PROJECT_ROOT / "outputs" / "checkpoints" / "yolo"

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Prepare pseudo YOLO dataset and train YOLOv8n tyre detector.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", default="auto", help="Batch size or 'auto'.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint to start from.")
    parser.add_argument("--device", default=None, help="Training device such as cpu, 0, or 0,1.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation and sample predictions.")
    return parser.parse_args()
def _parse_batch(batch_value: str) -> int | str:
    if isinstance(batch_value, str) and batch_value.lower() == "auto":
        return -1
    return int(batch_value)

def _copy_best_metrics_to_root() -> None:
    evaluation_metrics = OUTPUT_ROOT / "evaluation" / "metrics.json"
    if evaluation_metrics.exists():
        destination = OUTPUT_ROOT / "metrics.json"
        shutil.copy2(evaluation_metrics, destination)

def run(args: Namespace | None = None) -> Path:
    args = args or parse_args()
    prepared = prepare_yolo_dataset(splits_root=SPLITS_ROOT, yolo_root=YOLO_ROOT)
    config = YoloTrainingConfig(
        data_yaml=prepared.data_yaml_path,
        output_root=OUTPUT_ROOT,
        checkpoints_root=CHECKPOINTS_ROOT,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=_parse_batch(args.batch),
        patience=args.patience,
        device=args.device,
    )
    best_weights = train_yolo_detector(config)
    if not args.skip_eval:
        evaluate_yolo_detector(weights_path=best_weights, data_yaml=prepared.data_yaml_path, output_root=OUTPUT_ROOT)
        generate_sample_predictions(
            weights_path=best_weights,
            source_dir=YOLO_ROOT / "images" / "test",
            output_root=OUTPUT_ROOT,
        )
        _copy_best_metrics_to_root()
    print(f"YOLO training finished with best weights at {best_weights}")
    return best_weights

if __name__ == "__main__":
    run()