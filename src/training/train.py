from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import TyreClassificationDataset, load_split_dataframe
from src.data.split_dataset import create_splits
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.evaluation.evaluate import evaluate_from_checkpoint
from src.models.efficientnet_classifier import create_efficientnet_b0, set_backbone_trainable
from src.utils.common import (
    CHECKPOINTS_ROOT,
    CLASS_NAMES,
    METRICS_ROOT,
    SPLITS_ROOT,
    ensure_dir,
    get_device,
    prepare_output_dirs,
    save_json,
    set_seed,
)


def create_dataloaders(train_csv: Path, val_csv: Path, test_csv: Path, batch_size: int, num_workers: int):
    datasets = {
        "train": TyreClassificationDataset(train_csv, transform=build_train_transforms()),
        "val": TyreClassificationDataset(val_csv, transform=build_eval_transforms()),
        "test": TyreClassificationDataset(test_csv, transform=build_eval_transforms()),
    }

    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return datasets, dataloaders


def compute_class_weights(train_frame: pd.DataFrame) -> torch.Tensor:
    counts = train_frame["label_idx"].value_counts().sort_index().to_numpy()
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, dataloader, criterion, optimizer, device: torch.device, training: bool) -> dict:
    model.train(training)
    running_loss = 0.0
    labels = []
    predictions = []
    probabilities = []

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

        running_loss += loss.item() * images.size(0)
        labels.extend(targets.cpu().numpy().tolist())
        predictions.extend(preds.cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].detach().cpu().numpy().tolist())

    labels_np = np.array(labels)
    predictions_np = np.array(predictions)
    probabilities_np = np.array(probabilities)

    metrics = {
        "loss": running_loss / len(dataloader.dataset),
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "precision": float(precision_score(labels_np, predictions_np, zero_division=0)),
        "recall": float(recall_score(labels_np, predictions_np, zero_division=0)),
        "f1_score": float(f1_score(labels_np, predictions_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_np, probabilities_np)),
    }
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_f1: float, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "num_classes": len(CLASS_NAMES),
            "config": config,
        },
        path,
    )


def generate_training_summary(
    split_stats_path: Path,
    config: dict,
    best_metrics: dict,
    final_metrics: dict,
    history_path: Path,
) -> None:
    with split_stats_path.open("r", encoding="utf-8") as handle:
        split_stats = json.load(handle)
    history = pd.read_csv(history_path)

    lines = [
        "# Training Summary",
        "",
        "## Dataset Statistics",
        "",
        f"- Split statistics source: `{split_stats_path}`",
        f"- Total samples: {split_stats['total_samples']}",
        f"- Total training epochs run: {len(history)}",
        f"- Training samples: {config['train_samples']}",
        f"- Validation samples: {config['val_samples']}",
        f"- Test samples: {config['test_samples']}",
        "",
        "## Training Configuration",
        "",
    ]

    for key in [
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "fine_tune",
        "unfreeze_epoch",
        "patience",
    ]:
        lines.append(f"- {key}: {config[key]}")

    lines.extend(
        [
            "",
            "## Best Validation Metrics",
            "",
            f"- Best epoch: {best_metrics['epoch']}",
            f"- Accuracy: {best_metrics['accuracy']:.4f}",
            f"- Precision: {best_metrics['precision']:.4f}",
            f"- Recall: {best_metrics['recall']:.4f}",
            f"- F1-score: {best_metrics['f1_score']:.4f}",
            f"- ROC-AUC: {best_metrics['roc_auc']:.4f}",
            "",
            "## Test Set Results",
            "",
            f"- Accuracy: {final_metrics['accuracy']:.4f}",
            f"- Precision: {final_metrics['precision']:.4f}",
            f"- Recall: {final_metrics['recall']:.4f}",
            f"- F1-score: {final_metrics['f1_score']:.4f}",
            f"- ROC-AUC: {final_metrics['roc_auc']:.4f}",
            f"- Average Precision: {final_metrics['average_precision']:.4f}",
            "",
            "## Confusion Matrix Interpretation",
            "",
            f"- {final_metrics['confusion_matrix_interpretation']}",
            "",
            "## Failure Analysis Summary",
            "",
            f"- False positives saved: {final_metrics['failure_analysis']['false_positives']}",
            f"- False negatives saved: {final_metrics['failure_analysis']['false_negatives']}",
            f"- Failure CSV: `{final_metrics['failure_analysis']['failure_analysis_csv']}`",
        ]
    )

    summary_path = METRICS_ROOT / "training_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 tyre wear classifier.")
    parser.add_argument("--train-csv", type=Path, default=SPLITS_ROOT / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=SPLITS_ROOT / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=SPLITS_ROOT / "test.csv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--unfreeze-epoch", type=int, default=4)
    parser.add_argument("--auto-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dirs()
    set_seed(args.seed)
    device = get_device()

    if args.auto_split or not (args.train_csv.exists() and args.val_csv.exists() and args.test_csv.exists()):
        create_splits(seed=args.seed)

    train_frame = load_split_dataframe(args.train_csv)
    val_frame = load_split_dataframe(args.val_csv)
    test_frame = load_split_dataframe(args.test_csv)
    datasets, dataloaders = create_dataloaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_efficientnet_b0(pretrained=True, freeze_backbone=True)
    model.to(device)

    class_weights = compute_class_weights(train_frame).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "fine_tune": args.fine_tune,
        "unfreeze_epoch": args.unfreeze_epoch,
        "train_samples": len(train_frame),
        "val_samples": len(val_frame),
        "test_samples": len(test_frame),
        "device": str(device),
    }

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    backbone_unfrozen = False

    for epoch in range(1, args.epochs + 1):
        if args.fine_tune and (not backbone_unfrozen) and epoch >= args.unfreeze_epoch:
            set_backbone_trainable(model, trainable=True)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate * 0.1, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            backbone_unfrozen = True

        train_metrics = run_epoch(model, dataloaders["train"], criterion, optimizer, device, training=True)
        val_metrics = run_epoch(model, dataloaders["val"], criterion, optimizer, device, training=False)
        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "backbone_trainable": backbone_unfrozen,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_score']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )

        save_checkpoint(CHECKPOINTS_ROOT / "last_checkpoint.pt", model, optimizer, epoch, best_val_f1, config)

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(CHECKPOINTS_ROOT / "best_model.pt", model, optimizer, epoch, best_val_f1, config)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history_frame = pd.DataFrame(history)
    history_path = METRICS_ROOT / "training_history.csv"
    history_frame.to_csv(history_path, index=False)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_metrics = history_frame.loc[history_frame["epoch"] == best_epoch].iloc[0].to_dict()
    best_metrics_summary = {
        "epoch": int(best_epoch),
        "accuracy": float(best_metrics["val_accuracy"]),
        "precision": float(best_metrics["val_precision"]),
        "recall": float(best_metrics["val_recall"]),
        "f1_score": float(best_metrics["val_f1_score"]),
        "roc_auc": float(best_metrics["val_roc_auc"]),
        "loss": float(best_metrics["val_loss"]),
    }
    save_json(METRICS_ROOT / "best_validation_metrics.json", best_metrics_summary)

    final_metrics = evaluate_from_checkpoint(
        checkpoint_path=CHECKPOINTS_ROOT / "best_model.pt",
        csv_path=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generate_training_summary(
        split_stats_path=SPLITS_ROOT / "split_stats.json",
        config=config,
        best_metrics=best_metrics_summary,
        final_metrics=final_metrics,
        history_path=history_path,
    )

    print("Training complete")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]:
        print(f"test_{key}: {final_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
