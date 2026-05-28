from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.yolo_detector import run_inference_and_save

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "checkpoints" / "yolo" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "yolo_detection" / "inference"
DEFAULT_CROPS = PROJECT_ROOT / "outputs" / "crops"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO tyre detection inference and save ROI crops.")
    parser.add_argument("source", help="Image path, folder path, video path, or webcam index such as 0.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained YOLO weights.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for side-by-side visualizations.")
    parser.add_argument("--crops-dir", default=str(DEFAULT_CROPS), help="Directory for cropped tyre ROIs.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument(
        "--fallback-review-threshold",
        type=float,
        default=0.40,
        help="Recommend review when detection confidence is below this value or fallback crop is used.",
    )
    parser.add_argument(
        "--disable-low-confidence-fallback",
        action="store_true",
        help="Keep the detected ROI even when confidence is below the review threshold.",
    )
    return parser.parse_args()

def run() -> None:
    args = parse_args()
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass
    run_inference_and_save(
        weights_path=Path(args.weights),
        source=source,
        output_root=Path(args.output_dir),
        crops_root=Path(args.crops_dir),
        conf_threshold=args.conf,
        fallback_review_threshold=args.fallback_review_threshold,
        use_fallback_for_low_confidence=not args.disable_low_confidence_fallback,
    )

if __name__ == "__main__":
    run()