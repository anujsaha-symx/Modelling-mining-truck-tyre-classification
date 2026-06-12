from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import TyreClassificationDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.utils.common import (
    VERIFICATION_CHECKPOINTS_ROOT,
    VERIFICATION_METRICS_ROOT,
    VERIFICATION_SPLITS_ROOT,
    ensure_dir,
    get_device,
    prepare_output_dirs,
    save_json,
    set_seed,
)
from src.verification.dataset_builder import create_verification_splits
from src.verification.evaluate_verifier import evaluate_verifier, run_inference
from src.verification.verify import create_verifier_model


def create_dataloaders(train_csv: Path, val_csv: Path, batch_size: int, num_workers: int):
    train_ds = TyreClassificationDataset(train_csv, transform=build_train_transforms())
    val_ds = TyreClassificationDataset(val_csv, transform=build_eval_transforms())
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    frame = pd.read_csv(train_csv)
    counts = frame["label_idx"].value_counts().sort_index().to_numpy()
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def find_threshold_for_target_recall(labels: np.ndarray, probabilities: np.ndarray, target_recall: float) -> tuple[float, float]:
    best_threshold = 0.5
    best_precision = -1.0
    achieved_recall = 0.0
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((labels == 1) & (predictions == 1)).sum())
        fn = int(((labels == 1) & (predictions == 0)).sum())
        fp = int(((labels == 0) & (predictions == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if recall >= target_recall and precision > best_precision:
            best_threshold = float(threshold)
            best_precision = precision
            achieved_recall = recall
    return best_threshold, achieved_recall


def run_epoch(model, dataloader, criterion, optimizer, device: torch.device, training: bool) -> dict:
    model.train(training)
    losses = []
    labels, predictions, probabilities = [], [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            if training:
                loss.backward()
                optimizer.step()

        losses.append(loss.item() * images.size(0))
        labels.extend(targets.cpu().numpy().tolist())
        predictions.extend(preds.cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].detach().cpu().numpy().tolist())

    labels_np = np.array(labels)
    predictions_np = np.array(predictions)
    probabilities_np = np.array(probabilities)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return {
        "loss": float(sum(losses) / len(dataloader.dataset)),
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "precision": float(precision_score(labels_np, predictions_np, zero_division=0)),
        "recall": float(recall_score(labels_np, predictions_np, zero_division=0)),
        "f1_score": float(f1_score(labels_np, predictions_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_np, probabilities_np)),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_recall: float, threshold: float, architecture: str, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_recall": best_val_recall,
            "threshold": threshold,
            "architecture": architecture,
            "num_classes": 2,
            "config": config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tyre verification gatekeeper.")
    parser.add_argument("--train-csv", type=Path, default=VERIFICATION_SPLITS_ROOT / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=VERIFICATION_SPLITS_ROOT / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=VERIFICATION_SPLITS_ROOT / "test.csv")
    parser.add_argument("--negative-roots", type=Path, nargs="*")
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--architecture", choices=["efficientnet_b0", "mobilenet_v3_small"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dirs()
    set_seed(args.seed)
    device = get_device()

    if args.auto_build or not (args.train_csv.exists() and args.val_csv.exists() and args.test_csv.exists()):
        if not args.negative_roots:
            raise ValueError("Negative roots are required to build the verifier dataset.")
        create_verification_splits(negative_roots=args.negative_roots, seed=args.seed)

    ensure_dir(VERIFICATION_CHECKPOINTS_ROOT)
    ensure_dir(VERIFICATION_METRICS_ROOT)
    dataloaders = create_dataloaders(args.train_csv, args.val_csv, batch_size=args.batch_size, num_workers=args.num_workers)
    model = create_verifier_model(architecture=args.architecture, pretrained=True)
    model.to(device)

    class_weights = compute_class_weights(args.train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state_dict = None
    best_val_recall = -1.0
    best_threshold = 0.5
    best_val_precision = -1.0
    epochs_without_improvement = 0
    history = []
    config = {
        "architecture": args.architecture,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "target_recall": args.target_recall,
        "seed": args.seed,
        "device": str(device),
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, dataloaders["train"], criterion, optimizer, device, training=True)
        val_metrics = run_epoch(model, dataloaders["val"], criterion, optimizer, device, training=False)
        scheduler.step(val_metrics["loss"])

        val_results = run_inference(model, dataloaders["val"], device, threshold=0.5)
        threshold, threshold_recall = find_threshold_for_target_recall(
            val_results["labels"], val_results["probabilities"], args.target_recall
        )
        threshold_predictions = (val_results["probabilities"] >= threshold).astype(int)
        threshold_tp = int(((val_results["labels"] == 1) & (threshold_predictions == 1)).sum())
        threshold_fp = int(((val_results["labels"] == 0) & (threshold_predictions == 1)).sum())
        threshold_precision = threshold_tp / (threshold_tp + threshold_fp) if (threshold_tp + threshold_fp) else 0.0

        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "selected_threshold": threshold,
            "selected_threshold_recall": threshold_recall,
            "selected_threshold_precision": threshold_precision,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"threshold={threshold:.2f} threshold_recall={threshold_recall:.4f} threshold_precision={threshold_precision:.4f}"
        )

        save_checkpoint(
            VERIFICATION_CHECKPOINTS_ROOT / "last_verifier.pt",
            model,
            optimizer,
            epoch,
            best_val_recall,
            threshold,
            args.architecture,
            config,
        )

        if threshold_recall > best_val_recall or (
            threshold_recall == best_val_recall and threshold_precision > best_val_precision
        ):
            best_val_recall = threshold_recall
            best_threshold = threshold
            best_val_precision = threshold_precision
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                VERIFICATION_CHECKPOINTS_ROOT / "best_verifier.pt",
                model,
                optimizer,
                epoch,
                best_val_recall,
                best_threshold,
                args.architecture,
                config,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(VERIFICATION_METRICS_ROOT / "training_history.csv", index=False)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_json(
        VERIFICATION_METRICS_ROOT / "best_validation_metrics.json",
        {
            "best_val_recall": best_val_recall,
            "best_val_precision": best_val_precision,
            "threshold": best_threshold,
            "target_recall": args.target_recall,
        },
    )

    final_metrics = evaluate_verifier(VERIFICATION_CHECKPOINTS_ROOT / "best_verifier.pt", args.test_csv, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Verifier training complete")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]:
        print(f"test_{key}: {final_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
