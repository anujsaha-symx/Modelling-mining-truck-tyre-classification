from __future__ import annotations
import json
import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.classification_dataset import (  # noqa: E402
    CLASS_NAMES,
    TyreClassificationDataset,
    build_classification_manifests,
    build_eval_transforms,
)
from src.models.classifiers import load_classifier_checkpoint  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification"
SPLITS_ROOT = PROJECT_ROOT / "datasets" / "splits"
DEDUPLICATED_MANIFEST = PROJECT_ROOT / "outputs" / "metrics" / "deduplicated_dataset.csv"
CROPS_ROOT = PROJECT_ROOT / "outputs" / "crops"

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate a tyre wear classifier checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to classifier weights.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for bad class probability.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _derive_output_root(weights_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    return weights_path.resolve().parents[1]
def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _plot_confusion_matrix(matrix: np.ndarray, class_names: tuple[str, ...], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
def _plot_roc_curve(true_labels: np.ndarray, positive_scores: np.ndarray, output_path: Path) -> float | None:
    if len(np.unique(true_labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(true_labels, positive_scores)
    auc_value = roc_auc_score(true_labels, positive_scores)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return float(auc_value)
def _plot_pr_curve(true_labels: np.ndarray, positive_scores: np.ndarray, output_path: Path) -> float | None:
    if len(np.unique(true_labels)) < 2:
        return None
    precision_values, recall_values, _ = precision_recall_curve(true_labels, positive_scores)
    average_precision = average_precision_score(true_labels, positive_scores)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall_values, precision_values, label=f"AP = {average_precision:.4f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend(loc="lower left")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return float(average_precision)

def evaluate_checkpoint(
    weights_path: Path,
    split_name: str,
    batch_size: int,
    num_workers: int,
    threshold: float,
    output_root: Path,
    manifest_override: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    device = _device()
    model, checkpoint = load_classifier_checkpoint(weights_path=weights_path, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    if manifest_override is None:
        manifests, _ = build_classification_manifests(
            splits_root=SPLITS_ROOT,
            deduplicated_manifest_path=DEDUPLICATED_MANIFEST,
            crops_root=CROPS_ROOT,
        )
        manifest = manifests[split_name]
    else:
        manifest = manifest_override.reset_index(drop=True)
    dataset = TyreClassificationDataset(manifest=manifest, transform=build_eval_transforms(image_size=image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[float] = []
    total_loss = 0.0
    total_count = 0
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= threshold).long()
            batch_size_actual = inputs.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_count += batch_size_actual
            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    true_labels = np.asarray(all_targets, dtype=np.int64)
    predicted_labels = np.asarray(all_predictions, dtype=np.int64)
    positive_scores = np.asarray(all_probabilities, dtype=np.float32)
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1_score": float(f1_score(true_labels, predicted_labels, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
        "sample_count": int(total_count),
        "threshold": threshold,
        "split": split_name,
        "model_name": model.model_name,
    }
    confusion = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    _plot_confusion_matrix(confusion, class_names=class_names, output_path=output_root / "confusion_matrix.png")
    metrics["roc_auc"] = _plot_roc_curve(true_labels, positive_scores, output_path=output_root / "roc_curve.png")
    metrics["average_precision"] = _plot_pr_curve(true_labels, positive_scores, output_path=output_root / "pr_curve.png")
    report_dict = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    )
    _save_json(report_dict, output_root / "classification_report.json")
    (output_root / "classification_report.txt").write_text(report_text, encoding="utf-8")
    _save_json(metrics, output_root / "metrics.json")
    _save_failure_cases(manifest=manifest, probabilities=positive_scores, predictions=predicted_labels, output_root=output_root, split_name=split_name, class_names=class_names)
    LOGGER.info("Saved classifier evaluation artifacts to %s", output_root)
    return metrics

def run(args: Namespace | None = None) -> dict[str, float | int | None]:
    args = args or parse_args()
    weights_path = Path(args.weights)
    output_root = _derive_output_root(weights_path=weights_path, override=args.output_dir)
    configure_logging(output_root / "evaluate_classifier.log")
    metrics = evaluate_checkpoint(
        weights_path=weights_path,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        output_root=output_root,
    )
    LOGGER.info("Classifier evaluation complete: %s", metrics)
    return metrics

if __name__ == "__main__":
    run()