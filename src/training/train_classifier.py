from __future__ import annotations
import json
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.classification_dataset import (  # noqa: E402
    CLASS_NAMES,
    TyreClassificationDataset,
    build_classification_manifests,
    build_eval_transforms,
    build_train_transforms,
    verify_classification_leakage,
)
from src.evaluation.evaluate_classifier import evaluate_checkpoint  # noqa: E402
from src.models.classifiers import (  # noqa: E402
    SUPPORTED_CLASSIFIERS,
    ClassifierBuildConfig,
    build_classifier,
    save_classifier_checkpoint,
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification"
SPLITS_ROOT = PROJECT_ROOT / "datasets" / "splits"
DEDUPLICATED_MANIFEST = PROJECT_ROOT / "outputs" / "metrics" / "deduplicated_dataset.csv"
CROPS_ROOT = PROJECT_ROOT / "outputs" / "crops"

class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = float("-inf")
        self.bad_epoch_count = 0

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.bad_epoch_count = 0
            return False
        self.bad_epoch_count += 1
        return self.bad_epoch_count >= self.patience
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train binary tyre wear classifiers.")
    parser.add_argument("--model", default="all", choices=(*SUPPORTED_CLASSIFIERS, "all"), help="Classifier family to train.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epoch count.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4, help="Learning rate after unfreezing.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="ReduceLROnPlateau patience.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze the backbone after the freeze phase.")
    parser.add_argument("--fine-tune-epoch", type=int, default=4, help="Epoch index at which to unfreeze the backbone.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for bad class probability.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subset limit for debugging.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional val subset limit for debugging.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test subset limit for debugging.")
    return parser.parse_args()
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        print("Warning: Deterministic algorithms not fully available in this environment.", file=sys.stderr)
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
def _class_weights(train_manifest: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
    counts = train_manifest["label_id"].value_counts().sort_index()
    if len(set(counts.tolist())) <= 1:
        return None
    inverse = 1.0 / counts.astype(float)
    normalized = inverse / inverse.sum() * len(inverse)
    weights = torch.tensor(normalized.values, dtype=torch.float32, device=device)
    print(f"Using weighted cross entropy with weights: {normalized.to_dict()}")
    return weights
def _create_dataloader(dataset: TyreClassificationDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)
def _binary_metrics(targets: list[int], probabilities: list[float], threshold: float) -> dict[str, float]:
    targets_array = np.asarray(targets, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities_array >= threshold).astype(np.int64)
    true_positive = int(((predictions == 1) & (targets_array == 1)).sum())
    false_positive = int(((predictions == 1) & (targets_array == 0)).sum())
    false_negative = int(((predictions == 0) & (targets_array == 1)).sum())
    accuracy = float((predictions == targets_array).mean()) if len(targets_array) else 0.0
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_function: nn.Module,
    device: torch.device,
    threshold: float,
    scaler: GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    all_targets: list[int] = []
    all_probabilities: list[float] = []
    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(inputs)
            loss = loss_function(logits, targets)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        batch_size_actual = inputs.size(0)
        total_loss += float(loss.item()) * batch_size_actual
        total_count += batch_size_actual
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_probabilities.extend(probabilities.detach().cpu().numpy().tolist())
    metrics = _binary_metrics(targets=all_targets, probabilities=all_probabilities, threshold=threshold)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics

def _plot_learning_curves(history: list[dict[str, float]], output_path: Path) -> None:
    history_df = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_f1_score"], marker="o", label="train_f1")
    axes[1].plot(history_df["epoch"], history_df["val_f1_score"], marker="o", label="val_f1")
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="val_acc")
    axes[1].set_title("Performance Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
def _export_comparison_table() -> None:
    rows: list[dict[str, object]] = []
    for model_name in SUPPORTED_CLASSIFIERS:
        metrics_path = OUTPUT_ROOT / model_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )
    if not rows:
        return
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
    markdown_lines = ["| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        roc_auc = row["roc_auc"]
        roc_auc_text = "n/a" if roc_auc is None else f"{roc_auc:.4f}"
        markdown_lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {roc_auc_text} |"
        )
    (OUTPUT_ROOT / "model_comparison.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

def train_single_model(model_name: str, args: Namespace) -> Path:
    model_output_root = OUTPUT_ROOT / model_name
    checkpoints_root = model_output_root / "checkpoints"
    model_output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    verify_classification_leakage(
        splits_root=SPLITS_ROOT,
        deduplicated_manifest_path=DEDUPLICATED_MANIFEST,
        output_path=OUTPUT_ROOT / "leakage_audit.json",
    )
    manifests, summaries = build_classification_manifests(
        splits_root=SPLITS_ROOT,
        deduplicated_manifest_path=DEDUPLICATED_MANIFEST,
        crops_root=CROPS_ROOT,
        max_samples_by_split={
            "train": args.max_train_samples,
            "val": args.max_val_samples,
            "test": args.max_test_samples,
        },
    )
    _save_json(
        {
            split_name: {
                "sample_count": summary.sample_count,
                "crop_count": summary.crop_count,
                "fallback_count": summary.fallback_count,
                "class_distribution": summary.class_distribution,
            }
            for split_name, summary in summaries.items()
        },
        model_output_root / "dataset_summary.json",
    )
    device = _device()
    model = build_classifier(
        config=ClassifierBuildConfig(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True),
        class_names=CLASS_NAMES,
    )
    model.image_size = args.image_size
    model.freeze_backbone()
    model.to(device)
    print(f"Training {model_name} on {device}")
    train_dataset = TyreClassificationDataset(manifest=manifests["train"], transform=build_train_transforms(image_size=args.image_size))
    val_dataset = TyreClassificationDataset(manifest=manifests["val"], transform=build_eval_transforms(image_size=args.image_size))
    train_loader = _create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = _create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    class_weights = _class_weights(manifests["train"], device=device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    history: list[dict[str, float]] = []
    best_f1 = float("-inf")
    best_checkpoint_path = checkpoints_root / "best_model.pt"
    last_checkpoint_path = checkpoints_root / "last_model.pt"
    backbone_unfrozen = False
    for epoch in range(1, args.epochs + 1):
        if args.fine_tune and not backbone_unfrozen and epoch >= args.fine_tune_epoch:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
            backbone_unfrozen = True
            print(f"Unfroze {model_name} backbone at epoch {epoch}")
        train_metrics = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_function=loss_function, device=device, threshold=args.threshold, scaler=scaler)
        val_metrics = _run_epoch(model=model, loader=val_loader, optimizer=None, loss_function=loss_function, device=device, threshold=args.threshold, scaler=scaler)
        scheduler.step(val_metrics["loss"])
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        print(f"Epoch {epoch} | train={train_metrics} | val={val_metrics}")
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            save_classifier_checkpoint(
                destination=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_f1,
                history=history,
                extra_metadata={"fine_tuned": backbone_unfrozen, "threshold": args.threshold},
            )
        save_classifier_checkpoint(
            destination=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_f1,
            history=history,
            extra_metadata={"fine_tuned": backbone_unfrozen, "threshold": args.threshold},
        )
        if early_stopping.step(val_metrics["f1_score"]):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    _save_json({"history": history}, model_output_root / "history.json")
    _plot_learning_curves(history=history, output_path=model_output_root / "learning_curves.png")
    metrics = evaluate_checkpoint(
        weights_path=best_checkpoint_path,
        split_name="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        output_root=model_output_root,
        manifest_override=manifests["test"],
    )
    print(f"Final test metrics for {model_name}: {metrics}")
    _export_comparison_table()
    return best_checkpoint_path

def run(args: Namespace | None = None) -> list[Path]:
    args = args or parse_args()
    model_names = list(SUPPORTED_CLASSIFIERS) if args.model == "all" else [args.model]
    trained_checkpoints = [train_single_model(model_name=model_name, args=args) for model_name in model_names]
    return trained_checkpoints

if __name__ == "__main__":
    run()