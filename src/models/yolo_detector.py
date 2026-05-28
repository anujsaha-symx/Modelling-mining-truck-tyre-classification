from __future__ import annotations
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

FALLBACK_BOX = (0.05, 0.05, 0.95, 0.95)
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}

@dataclass(frozen=True)
class YoloTrainingConfig:
    data_yaml: Path
    output_root: Path
    checkpoints_root: Path
    model_name: str = "yolov8n.pt"
    experiment_name: str = "yolov8n_pseudo_detection"
    epochs: int = 30
    imgsz: int = 640
    pretrained: bool = True
    batch: int = -1
    patience: int = 10
    seed: int = 42
    deterministic: bool = True
    workers: int = 0
    device: str | int | None = None
def _require_ultralytics() -> None:
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Run `pip install ultralytics`.") from IMPORT_ERROR
def _reset_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
def _copy_weights(source_dir: Path, checkpoints_root: Path) -> None:
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    weights_dir = source_dir / "weights"
    for filename in ("best.pt", "last.pt"):
        weight_path = weights_dir / filename
        if weight_path.exists():
            shutil.copy2(weight_path, checkpoints_root / filename)
def _copy_if_exists(source_path: Path, destination_path: Path) -> None:
    if source_path.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
def _archive_run_artifacts(run_root: Path, archive_root: Path, artifact_names: tuple[str, ...]) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    for artifact_name in artifact_names:
        _copy_if_exists(run_root / artifact_name, archive_root / artifact_name)
def _write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
def _is_stream_source(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    return Path(str(source)).suffix.lower() in VIDEO_SUFFIXES
def _draw_panel_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    titled = image_bgr.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(titled, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return titled
def _create_video_writer(output_root: Path, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    destination = output_root / "comparison_stream.mp4"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer at {destination}")
    return writer
def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
def export_training_curves(results_csv_path: Path, output_path: Path) -> Path:
    dataframe = pd.read_csv(results_csv_path)
    dataframe.columns = [column.strip() for column in dataframe.columns]
    epochs = dataframe["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for column in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"):
        if column in dataframe:
            axes[0].plot(epochs, dataframe[column], marker="o", label=column)
    axes[0].set_title("YOLO Training Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    for column in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if column in dataframe:
            axes[1].plot(epochs, dataframe[column], marker="o", label=column)
    axes[1].set_title("YOLO Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path

def train_yolo_detector(config: YoloTrainingConfig) -> Path:
    _require_ultralytics()
    run_root = config.output_root / config.experiment_name
    _reset_directory(run_root)
    config.checkpoints_root.mkdir(parents=True, exist_ok=True)
    print(f"Starting YOLO training using {config.data_yaml}")
    model = YOLO(config.model_name)
    model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        pretrained=config.pretrained,
        batch=config.batch,
        patience=config.patience,
        project=str(config.output_root),
        name=config.experiment_name,
        exist_ok=True,
        seed=config.seed,
        deterministic=config.deterministic,
        workers=config.workers,
        device=config.device,
        verbose=True,
        plots=True,
    )
    results_csv_path = run_root / "results.csv"
    if results_csv_path.exists():
        export_training_curves(results_csv_path=results_csv_path, output_path=run_root / "results.png")
    _copy_weights(run_root, config.checkpoints_root)
    _archive_run_artifacts(
        run_root=run_root,
        archive_root=config.output_root / "artifacts",
        artifact_names=(
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxF1_curve.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "labels.jpg",
            "args.yaml",
            "results.csv",
        ),
    )
    best_weights = config.checkpoints_root / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected trained weights not found: {best_weights}")
    print(f"Training complete. Best weights stored at {best_weights}")
    return best_weights

def evaluate_yolo_detector(weights_path: Path, data_yaml: Path, output_root: Path, experiment_name: str = "evaluation") -> dict:
    _require_ultralytics()
    run_root = output_root / experiment_name
    _reset_directory(run_root)
    model = YOLO(str(weights_path))
    print(f"Evaluating YOLO weights {weights_path}")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        project=str(output_root),
        name=experiment_name,
        exist_ok=True,
        plots=True,
    )
    metrics_payload = {
        "map50": _safe_float(metrics.box.map50),
        "map50_95": _safe_float(metrics.box.map),
        "precision": _safe_float(metrics.box.mp),
        "recall": _safe_float(metrics.box.mr),
    }
    metrics_path = run_root / "metrics.json"
    _write_json(metrics_payload, metrics_path)
    _write_json(metrics_payload, output_root / "metrics.json")
    _archive_run_artifacts(
        run_root=run_root,
        archive_root=output_root / "artifacts" / experiment_name,
        artifact_names=(
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxF1_curve.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "results.csv",
        ),
    )
    return metrics_payload

def generate_sample_predictions(
    weights_path: Path,
    source_dir: Path,
    output_root: Path,
    experiment_name: str = "sample_predictions",
    conf_threshold: float = 0.25,
    limit: int = 12,
) -> Path:
    _require_ultralytics()
    run_root = output_root / experiment_name
    _reset_directory(run_root)
    source_images = sorted(path for path in source_dir.iterdir() if path.is_file())[:limit]
    if not source_images:
        raise FileNotFoundError(f"No images found for sample predictions in {source_dir}")
    model = YOLO(str(weights_path))
    model.predict(
        source=[str(path) for path in source_images],
        conf=conf_threshold,
        save=True,
        project=str(output_root),
        name=experiment_name,
        exist_ok=True,
        verbose=True,
    )
    return run_root

def _fallback_crop(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_bgr.shape[:2]
    x1 = int(width * FALLBACK_BOX[0])
    y1 = int(height * FALLBACK_BOX[1])
    x2 = int(width * FALLBACK_BOX[2])
    y2 = int(height * FALLBACK_BOX[3])
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)

def _select_best_detection(result, conf_threshold: float):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best_box = None
    best_confidence = -1.0
    for box in result.boxes:
        confidence = float(box.conf.item())
        if confidence >= conf_threshold and confidence > best_confidence:
            best_confidence = confidence
            best_box = box
    return best_box

def _create_side_by_side(original_bgr: np.ndarray, annotated_bgr: np.ndarray, crop_bgr: np.ndarray) -> np.ndarray:
    panel_height = 480
    def resize_panel(image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        scale = panel_height / height
        resized = cv2.resize(image_bgr, (max(1, int(width * scale)), panel_height))
        return resized
    titled_panels = (
        _draw_panel_title(original_bgr, "Original"),
        _draw_panel_title(annotated_bgr, "Detected BBox"),
        _draw_panel_title(crop_bgr, "Cropped Tyre ROI"),
    )
    panels = [resize_panel(panel) for panel in titled_panels]
    max_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        pad_width = max_width - panel.shape[1]
        padded.append(cv2.copyMakeBorder(panel, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.hstack(padded)

def run_inference_and_save(
    weights_path: Path,
    source: str | int,
    output_root: Path,
    crops_root: Path,
    conf_threshold: float = 0.25,
    fallback_review_threshold: float = 0.40,
    use_fallback_for_low_confidence: bool = True,
) -> None:
    _require_ultralytics()
    output_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    results = model.predict(source=source, conf=conf_threshold, stream=True, verbose=True)
    stream_source = _is_stream_source(source)
    summary = {
        "source": str(source),
        "source_type": "stream" if stream_source else "file_or_folder",
        "confidence_threshold": conf_threshold,
        "fallback_review_threshold": fallback_review_threshold,
        "use_fallback_for_low_confidence": use_fallback_for_low_confidence,
        "frames_processed": 0,
        "fallback_count": 0,
        "low_confidence_count": 0,
        "review_recommended_count": 0,
    }
    video_writer = None
    try:
        for index, result in enumerate(results, start=1):
            summary["frames_processed"] += 1
            source_name = Path(str(result.path)).stem if result.path else f"frame_{index:04d}"
            original_bgr = result.orig_img.copy()
            annotated_bgr = result.plot()
            best_box = _select_best_detection(result, conf_threshold)
            fallback_used = False
            confidence = None
            fallback_reason = None
            review_recommended = False
            detected_box = None
            if best_box is None:
                crop_bgr, crop_box = _fallback_crop(original_bgr)
                fallback_used = True
                fallback_reason = "no_detection_above_threshold"
            else:
                x1, y1, x2, y2 = [int(value) for value in best_box.xyxy[0].tolist()]
                detected_box = (x1, y1, x2, y2)
                crop_box = (x1, y1, x2, y2)
                crop_bgr = original_bgr[y1:y2, x1:x2]
                confidence = float(best_box.conf.item())
                if confidence < fallback_review_threshold:
                    summary["low_confidence_count"] += 1
                    review_recommended = True
                    if use_fallback_for_low_confidence:
                        crop_bgr, crop_box = _fallback_crop(original_bgr)
                        fallback_used = True
                        fallback_reason = "low_confidence_detection"
            if crop_bgr.size == 0:
                crop_bgr, crop_box = _fallback_crop(original_bgr)
                fallback_used = True
                fallback_reason = "empty_crop_after_detection"
            if fallback_used:
                summary["fallback_count"] += 1
                review_recommended = True
                print(f"Using fallback crop for {result.path} due to {fallback_reason}.", file=sys.stderr)
            if review_recommended:
                summary["review_recommended_count"] += 1
            crop_name = f"{source_name}_crop_{index:04d}.jpg"
            crop_path = crops_root / crop_name
            cv2.imwrite(str(crop_path), crop_bgr)
            side_by_side = _create_side_by_side(original_bgr, annotated_bgr, crop_bgr)
            if stream_source:
                if video_writer is None:
                    video_writer = _create_video_writer(output_root=output_root, frame_shape=side_by_side.shape)
                video_writer.write(side_by_side)
            vis_path = output_root / f"{source_name}_comparison_{index:04d}.jpg"
            cv2.imwrite(str(vis_path), side_by_side)
            metadata = {
                "source": str(result.path),
                "crop_path": str(crop_path),
                "visualization_path": str(vis_path),
                "confidence_threshold": conf_threshold,
                "fallback_review_threshold": fallback_review_threshold,
                "detection_confidence": confidence,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "review_recommended": review_recommended,
                "detected_box_xyxy": detected_box,
                "crop_box_xyxy": crop_box,
            }
            _write_json(metadata, output_root / f"{source_name}_comparison_{index:04d}.json")
    finally:
        if video_writer is not None:
            video_writer.release()
    summary_path = output_root / "inference_summary.json"
    _write_json(summary, summary_path)

def save_crop_preview(crop_path: Path, output_path: Path) -> None:
    with Image.open(crop_path) as image:
        image.save(output_path)