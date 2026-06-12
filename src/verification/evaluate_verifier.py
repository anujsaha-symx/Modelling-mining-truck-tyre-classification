from __future__ import annotations

import argparse
import json
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

from src.data.dataset import TyreClassificationDataset
from src.data.transforms import build_eval_transforms
from src.utils.common import (
    VERIFICATION_FIGURES_ROOT,
    VERIFICATION_METRICS_ROOT,
    VERIFICATION_PREDICTIONS_ROOT,
    ensure_dir,
    get_device,
    save_json,
)
from src.verification.verify import load_verifier


CLASS_NAMES = ["non_tyre", "tyre"]


def run_inference(model, dataloader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    labels, predictions, probabilities, filepaths = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] >= threshold).long()
            labels.extend(targets.cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs[:, 1].cpu().numpy().tolist())
            filepaths.extend(batch["filepath"])
    return {
        "labels": np.array(labels),
        "predictions": np.array(predictions),
        "probabilities": np.array(probabilities),
        "filepaths": filepaths,
    }


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: Path) -> np.ndarray:
    matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Verifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return matrix


def save_roc_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_score = roc_auc_score(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Verifier ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pr_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ap_score = average_precision_score(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Verifier Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_failures(results: dict, output_dir: Path) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    false_neg_dir = ensure_dir(output_dir / "false_negatives")
    false_pos_dir = ensure_dir(output_dir / "false_positives")
    records = []

    for filepath, true_label, pred_label, tyre_probability in zip(
        results["filepaths"], results["labels"], results["predictions"], results["probabilities"]
    ):
        if true_label == pred_label:
            continue
        source = Path(filepath)
        confidence = tyre_probability if pred_label == 1 else 1.0 - tyre_probability
        if true_label == 1 and pred_label == 0:
            shutil.copy2(source, false_neg_dir / source.name)
        else:
            shutil.copy2(source, false_pos_dir / source.name)
        records.append(
            {
                "filepath": str(source),
                "true_label": CLASS_NAMES[int(true_label)],
                "predicted_label": CLASS_NAMES[int(pred_label)],
                "confidence": round(float(confidence), 6),
            }
        )

    frame = pd.DataFrame(records, columns=["filepath", "true_label", "predicted_label", "confidence"])
    frame.to_csv(output_dir / "failure_analysis.csv", index=False)
    return {
        "false_positives": int(((results["labels"] == 0) & (results["predictions"] == 1)).sum()),
        "false_negatives": int(((results["labels"] == 1) & (results["predictions"] == 0)).sum()),
    }


def evaluate_verifier(checkpoint_path: Path, csv_path: Path, batch_size: int = 32, num_workers: int = 0) -> dict:
    device = get_device()
    model, checkpoint = load_verifier(checkpoint_path, device)
    threshold = float(checkpoint.get("threshold", 0.5))
    dataset = TyreClassificationDataset(csv_path=csv_path, transform=build_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results = run_inference(model, dataloader, device, threshold=threshold)
    metrics = compute_metrics(results["labels"], results["predictions"], results["probabilities"])

    ensure_dir(VERIFICATION_FIGURES_ROOT)
    ensure_dir(VERIFICATION_METRICS_ROOT)
    ensure_dir(VERIFICATION_PREDICTIONS_ROOT)

    matrix = save_confusion_matrix(results["labels"], results["predictions"], VERIFICATION_FIGURES_ROOT / "confusion_matrix.png")
    save_roc_curve(results["labels"], results["probabilities"], VERIFICATION_FIGURES_ROOT / "roc_curve.png")
    save_pr_curve(results["labels"], results["probabilities"], VERIFICATION_FIGURES_ROOT / "pr_curve.png")
    failure_summary = export_failures(results, VERIFICATION_PREDICTIONS_ROOT / "failure_cases")
    report = classification_report(results["labels"], results["predictions"], target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    payload = {
        **metrics,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "failure_analysis": failure_summary,
    }
    save_json(VERIFICATION_METRICS_ROOT / "final_metrics.json", payload)
    save_json(VERIFICATION_METRICS_ROOT / "classification_report.json", report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tyre verifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    metrics = evaluate_verifier(args.checkpoint, args.csv, batch_size=args.batch_size, num_workers=args.num_workers)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
