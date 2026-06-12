from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import TyreClassificationDataset, load_split_dataframe
from src.data.transforms import build_eval_transforms
from src.models.efficientnet_classifier import create_efficientnet_b0
from src.utils.common import (
    CLASS_NAMES,
    FIGURES_ROOT,
    IDX_TO_CLASS,
    METRICS_ROOT,
    PREDICTIONS_ROOT,
    ensure_dir,
    get_device,
    save_json,
)


def run_inference(model, dataloader, device: torch.device) -> dict:
    model.eval()
    probabilities = []
    predictions = []
    labels = []
    filepaths = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            probabilities.extend(probs[:, 1].cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            labels.extend(targets.cpu().numpy().tolist())
            filepaths.extend(batch["filepath"])

    return {
        "filepaths": filepaths,
        "labels": np.array(labels),
        "predictions": np.array(predictions),
        "probabilities": np.array(probabilities),
    }


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }
    return metrics


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: Path) -> np.ndarray:
    matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return matrix


def save_roc_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> dict:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_score = roc_auc_score(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": float(auc_score)}


def save_pr_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> dict:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ap_score = average_precision_score(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return {"precision": precision.tolist(), "recall": recall.tolist(), "average_precision": float(ap_score)}


def export_failure_cases(results: dict, output_dir: Path) -> dict:
    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]
    filepaths = results["filepaths"]

    if output_dir.exists():
        shutil.rmtree(output_dir)

    fp_dir = ensure_dir(output_dir / "false_positives")
    fn_dir = ensure_dir(output_dir / "false_negatives")

    records = []
    false_positive_count = 0
    false_negative_count = 0

    for filepath, true_idx, pred_idx, prob_bad in zip(filepaths, labels, predictions, probabilities):
        if true_idx == pred_idx:
            continue

        predicted_name = IDX_TO_CLASS[int(pred_idx)]
        true_name = IDX_TO_CLASS[int(true_idx)]
        confidence = prob_bad if pred_idx == 1 else (1.0 - prob_bad)

        source = Path(filepath)
        if true_idx == 0 and pred_idx == 1:
            destination = fp_dir / source.name
            false_positive_count += 1
        else:
            destination = fn_dir / source.name
            false_negative_count += 1

        shutil.copy2(source, destination)
        records.append(
            {
                "filepath": str(source),
                "true_label": true_name,
                "predicted_label": predicted_name,
                "confidence": round(float(confidence), 6),
            }
        )

    frame = pd.DataFrame(records, columns=["filepath", "true_label", "predicted_label", "confidence"])
    csv_path = output_dir / "failure_analysis.csv"
    frame.to_csv(csv_path, index=False)

    return {
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "failure_analysis_csv": str(csv_path),
    }


def interpret_confusion_matrix(matrix: np.ndarray) -> str:
    tn, fp, fn, tp = matrix.ravel()
    return (
        f"True negatives (good->good): {tn}, false positives (good->bad): {fp}, "
        f"false negatives (bad->good): {fn}, true positives (bad->bad): {tp}."
    )


def evaluate_from_checkpoint(
    checkpoint_path: Path,
    csv_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_efficientnet_b0(
        num_classes=checkpoint.get("num_classes", 2),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    dataset = TyreClassificationDataset(csv_path=csv_path, transform=build_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results = run_inference(model, dataloader, device)
    metrics = compute_metrics(results["labels"], results["predictions"], results["probabilities"])

    ensure_dir(FIGURES_ROOT)
    ensure_dir(METRICS_ROOT)
    failure_dir = ensure_dir(PREDICTIONS_ROOT / "failure_cases")

    matrix = save_confusion_matrix(results["labels"], results["predictions"], FIGURES_ROOT / "confusion_matrix.png")
    roc_data = save_roc_curve(results["labels"], results["probabilities"], FIGURES_ROOT / "roc_curve.png")
    pr_data = save_pr_curve(results["labels"], results["probabilities"], FIGURES_ROOT / "pr_curve.png")
    report = classification_report(
        results["labels"],
        results["predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    failure_summary = export_failure_cases(results, failure_dir)

    final_metrics = {
        **metrics,
        "checkpoint": str(checkpoint_path),
        "evaluated_split": str(csv_path),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_interpretation": interpret_confusion_matrix(matrix),
        "classification_report": report,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "failure_analysis": failure_summary,
    }
    save_json(METRICS_ROOT / "final_metrics.json", final_metrics)
    save_json(METRICS_ROOT / "classification_report.json", report)
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 tyre classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    metrics = evaluate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Evaluation complete")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]:
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
