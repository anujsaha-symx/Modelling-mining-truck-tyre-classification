from __future__ import annotations
import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, precision_score, recall_score
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
)
from src.models.classifiers import load_classifier_checkpoint  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification"

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Comprehensive classifier analysis.")
    parser.add_argument("--model-dir", required=True, help="Path to model output directory (e.g. outputs/classification/efficientnet).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-robustness-samples", type=int, default=5, help="Number of augmented forward passes per sample for robustness test.")
    return parser.parse_args()
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _get_manifest(model_dir: Path, split: str = "test") -> pd.DataFrame:
    manifests, _ = build_classification_manifests(
        splits_root=PROJECT_ROOT / "datasets" / "splits",
        deduplicated_manifest_path=PROJECT_ROOT / "outputs" / "metrics" / "deduplicated_dataset.csv",
        crops_root=PROJECT_ROOT / "outputs" / "crops",
    )
    return manifests[split]

def _load_model_and_run(model_dir: Path, manifest: pd.DataFrame, batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    weights_candidates = sorted((model_dir / "checkpoints").glob("*.pt"))
    if not weights_candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir / 'checkpoints'}")
    weights_path = weights_candidates[-1]
    device = _device()
    model, checkpoint = load_classifier_checkpoint(weights_path=weights_path, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    model.eval()
    dataset = TyreClassificationDataset(manifest=manifest, transform=build_eval_transforms(image_size=image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    all_logits: list[np.ndarray] = []
    all_targets: list[int] = []
    all_metadata: list[dict[str, str]] = []
    with torch.no_grad():
        for inputs, targets, metadata in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_metadata.extend(metadata)
    logits_array = np.concatenate(all_logits, axis=0)
    probabilities = softmax(logits_array, axis=1)
    targets_array = np.asarray(all_targets, dtype=np.int64)
    return probabilities, logits_array, targets_array, checkpoint
def plot_confidence_distribution(probabilities: np.ndarray, targets: np.ndarray, output_path: Path, class_names: tuple[str, ...]) -> None:
    positive_scores = probabilities[:, 1]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for class_index in (0, 1):
        mask = targets == class_index
        scores = positive_scores[mask]
        axes[class_index].hist(scores, bins=40, alpha=0.7, color=f"C{class_index}")
        axes[class_index].set_title(f"Confidence Distribution: {class_names[class_index]} (n={len(scores)})")
        axes[class_index].set_xlabel("Bad-class probability")
        axes[class_index].set_ylabel("Count")
        axes[class_index].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="default threshold")
        axes[class_index].legend()
        axes[class_index].grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
def plot_confidence_correct_vs_incorrect(probabilities: np.ndarray, targets: np.ndarray, predictions: np.ndarray, output_path: Path, class_names: tuple[str, ...]) -> None:
    positive_scores = probabilities[:, 1]
    correct_mask = predictions == targets
    incorrect_mask = ~correct_mask
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(positive_scores[correct_mask], bins=40, alpha=0.7, color="green", label=f"Correct (n={correct_mask.sum()})")
    axes[0].hist(positive_scores[incorrect_mask], bins=40, alpha=0.7, color="red", label=f"Incorrect (n={incorrect_mask.sum()})")
    axes[0].set_xlabel("Bad-class probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence: Correct vs Incorrect Predictions")
    axes[0].axvline(0.5, color="black", linestyle="--", alpha=0.5, label="default threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for class_index in (0, 1):
        class_mask = targets == class_index
        correct_class = correct_mask & class_mask
        incorrect_class = incorrect_mask & class_mask
        axes[1].hist(positive_scores[correct_class], bins=40, alpha=0.5, color="green", label=f"Correct {class_names[class_index]} (n={correct_class.sum()})")
        axes[1].hist(positive_scores[incorrect_class], bins=40, alpha=0.5, color="red", label=f"Incorrect {class_names[class_index]} (n={incorrect_class.sum()})")

    axes[1].set_xlabel("Bad-class probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence by Class: Correct vs Incorrect")
    axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def save_low_confidence_failures(probabilities: np.ndarray, targets: np.ndarray, predictions: np.ndarray, manifest: pd.DataFrame, output_root: Path, n_examples: int = 20) -> None:
    positive_scores = probabilities[:, 1]
    incorrect_mask = predictions != targets
    if not incorrect_mask.any():
        print("No incorrect predictions; skipping low-confidence failure examples.")
        return
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_confs = np.abs(positive_scores[incorrect_mask] - 0.5)
    sorted_order = np.argsort(incorrect_confs)[:n_examples]
    output_root.mkdir(parents=True, exist_ok=True)
    import shutil
    records = []
    for rank, idx in enumerate(sorted_order):
        orig_idx = incorrect_indices[idx]
        row = manifest.iloc[orig_idx]
        src = Path(row["image_path"])
        dst = output_root / f"lowconf_failure_{rank:03d}_{src.name}"
        shutil.copy2(str(src), str(dst))
        records.append({
            "rank": int(rank),
            "image_path": str(src),
            "true_label": row["label"],
            "predicted_label": CLASS_NAMES[int(predictions[orig_idx])],
            "bad_confidence": float(positive_scores[orig_idx]),
            "distance_from_threshold": float(np.abs(positive_scores[orig_idx] - 0.5)),
        })
    _save_json({"low_confidence_failures": records}, output_root / "low_confidence_failures.json")

def threshold_sweep(probabilities: np.ndarray, targets: np.ndarray, output_path: Path) -> dict[str, list[float]]:
    positive_scores = probabilities[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    records: list[dict[str, float]] = []
    best_f1_bad = 0.0
    best_threshold_f1 = 0.5
    best_recall_bad = 0.0
    best_threshold_recall = 0.5
    for threshold in thresholds:
        predictions = (positive_scores >= threshold).astype(np.int64)
        f1_bad = f1_score(targets, predictions, pos_label=1, zero_division=0)
        recall_bad = recall_score(targets, predictions, pos_label=1, zero_division=0)
        precision_bad = precision_score(targets, predictions, pos_label=1, zero_division=0)
        f1_good = f1_score(targets, predictions, pos_label=0, zero_division=0)
        records.append({"threshold": float(threshold), "f1_bad": float(f1_bad), "recall_bad": float(recall_bad), "precision_bad": float(precision_bad), "f1_good": float(f1_good)})
        if f1_bad > best_f1_bad:
            best_f1_bad = f1_bad
            best_threshold_f1 = float(threshold)
        if recall_bad > best_recall_bad:
            best_recall_bad = recall_bad
            best_threshold_recall = float(threshold)
    sweep_df = pd.DataFrame.from_records(records)
    sweep_df.to_csv(output_path.parent / "threshold_sweep.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sweep_df["threshold"], sweep_df["f1_bad"], label="F1 (bad class)")
    axis.plot(sweep_df["threshold"], sweep_df["recall_bad"], label="Recall (bad class)")
    axis.plot(sweep_df["threshold"], sweep_df["precision_bad"], label="Precision (bad class)")
    axis.axvline(best_threshold_f1, color="green", linestyle="--", alpha=0.6, label=f"Best F1 threshold: {best_threshold_f1:.2f}")
    axis.axvline(best_threshold_recall, color="red", linestyle="--", alpha=0.6, label=f"Best recall threshold: {best_threshold_recall:.2f}")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_title("Threshold Sweep (bad class)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    best_recall_at_precision = 0.0
    best_threshold_precision_recall = 0.5
    for threshold in thresholds:
        predictions = (positive_scores >= threshold).astype(np.int64)
        recall_bad = recall_score(targets, predictions, pos_label=1, zero_division=0)
        precision_bad = precision_score(targets, predictions, pos_label=1, zero_division=0)
        if recall_bad >= 0.90 and precision_bad > best_recall_at_precision:
            best_recall_at_precision = precision_bad
            best_threshold_precision_recall = float(threshold)
    return {
        "threshold_sweep": records,
        "best_f1_threshold": best_threshold_f1,
        "best_f1_score": float(best_f1_bad),
        "best_recall_threshold": best_threshold_recall,
        "best_recall_score": float(best_recall_bad),
        "threshold_for_90pct_recall": {"threshold": best_threshold_precision_recall, "precision_at_threshold": float(best_recall_at_precision)},
    }

def plot_calibration(probabilities: np.ndarray, targets: np.ndarray, output_path: Path, n_bins: int = 10) -> dict[str, object]:
    positive_scores = probabilities[:, 1]
    prob_true, prob_pred = calibration_curve(targets, positive_scores, n_bins=n_bins, strategy="uniform")
    brier_score = float(np.mean((positive_scores - targets) ** 2))
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    axis.plot(prob_pred, prob_true, marker="o", label=f"ECE = {ece:.4f}")
    axis.fill_between(prob_pred, 0, prob_true, alpha=0.1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Fraction of positives")
    axis.set_title(f"Reliability Diagram  (Brier = {brier_score:.4f})")
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.3)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {"brier_score": brier_score, "expected_calibration_error": ece}

def robustness_test(
    model_dir: Path, manifest: pd.DataFrame, batch_size: int, num_workers: int, num_samples: int, model_checkpoint: dict | None = None
) -> dict[str, float]:
    weights_candidates = sorted((model_dir / "checkpoints").glob("*.pt"))
    if not weights_candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir / 'checkpoints'}")
    weights_path = weights_candidates[-1]
    device = _device()
    model, checkpoint = load_classifier_checkpoint(weights_path=weights_path, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    model.eval()
    eval_transform = build_eval_transforms(image_size=image_size)
    all_base_probs: list[float] = []
    all_std_probs: list[float] = []
    subset = manifest.sample(min(200, len(manifest)), random_state=42).reset_index(drop=True)
    for row_index, row in subset.iterrows():
        image_path = Path(row["image_path"])
        from PIL import Image
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            base_tensor = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            base_logits = model(base_tensor)
        base_prob = float(softmax(base_logits.cpu().numpy(), axis=1)[0, 1])
        base_probs: list[float] = []
        for _ in range(num_samples):
            aug_transform = build_train_transforms(image_size=image_size)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
            aug_tensor = aug_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                aug_logits = model(aug_tensor)
            aug_prob = float(softmax(aug_logits.cpu().numpy(), axis=1)[0, 1])
            base_probs.append(aug_prob)
        all_base_probs.append(base_prob)
        all_std_probs.append(float(np.std(base_probs)))
    std_array = np.asarray(all_std_probs)
    robustness_metrics = {
        "mean_std": float(np.mean(std_array)),
        "median_std": float(np.median(std_array)),
        "std_percentile_90": float(np.percentile(std_array, 90)),
        "std_percentile_95": float(np.percentile(std_array, 95)),
        "max_std": float(np.max(std_array)),
        "num_samples": len(all_base_probs),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(std_array, bins=30, alpha=0.7)
    axes[0].axvline(robustness_metrics["mean_std"], color="red", linestyle="--", label=f"Mean: {robustness_metrics['mean_std']:.3f}")
    axes[0].set_xlabel("Std of predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Augmentation Robustness: Prediction Stability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(all_base_probs, all_std_probs, alpha=0.5, s=10)
    axes[1].set_xlabel("Base probability (bad class)")
    axes[1].set_ylabel("Std of probability under augmentation")
    axes[1].set_title("Stability vs. Confidence")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    output_path = model_dir / "robustness.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return robustness_metrics

def _categorize_failure(image_path: Path) -> str:
    """Simple heuristic-based failure categorization."""
    from PIL import Image as PILImage
    try:
        with PILImage.open(image_path) as img:
            img = img.convert("L")
            img_array = np.array(img, dtype=np.uint8)
    except Exception:
        return "unknown"
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    mean_brightness = float(img_array.mean())
    std_brightness = float(img_array.std())
    if laplacian_var < 50:
        return "blur"
    if mean_brightness < 50:
        return "lighting"
    if mean_brightness > 200:
        return "lighting"
    if std_brightness < 30:
        return "low_texture"
    return "ambiguous_wear"

def _generate_failure_analysis(probabilities: np.ndarray, targets: np.ndarray, predictions: np.ndarray, manifest: pd.DataFrame, output_root: Path) -> dict[str, object]:
    positive_scores = probabilities[:, 1]
    incorrect_mask = predictions != targets
    if not incorrect_mask.any():
        print("No failure cases found.")
        return {}
    failure_root = output_root / "failure_analysis"
    failure_root.mkdir(parents=True, exist_ok=True)
    categories: dict[str, list[dict[str, object]]] = {
        "blur": [], "lighting": [], "side_angle": [], "low_texture": [],
        "background_clutter": [], "ambiguous_wear": [], "unknown": [],
    }
    incorrect_indices = np.where(incorrect_mask)[0]
    for idx in incorrect_indices:
        row = manifest.iloc[idx]
        image_path = Path(row["image_path"])
        cat = _categorize_failure(image_path)
        if cat not in categories:
            cat = "unknown"
        categories[cat].append({
            "image_path": str(image_path),
            "true_label": row["label"],
            "predicted_label": CLASS_NAMES[int(predictions[idx])],
            "bad_confidence": float(positive_scores[idx]),
        })
    summary = {cat: len(items) for cat, items in categories.items()}
    _save_json({"failure_categories": summary, "details": {cat: items for cat, items in categories.items()}},
               failure_root / "failure_analysis.json")
    figure, axis = plt.subplots(figsize=(10, 5))
    cat_names = [k for k, v in categories.items() if v]
    cat_counts = [len(categories[k]) for k in cat_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_names)))
    axis.barh(cat_names, cat_counts, color=colors)
    axis.set_xlabel("Number of failures")
    axis.set_title(f"Failure Categorization (total={int(incorrect_mask.sum())})")
    for i, v in enumerate(cat_counts):
        axis.text(v + 0.5, i, str(v), va="center")
    figure.tight_layout()
    figure.savefig(failure_root / "failure_categories.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return summary

def analyze_classifier(model_dir: Path, batch_size: int, num_workers: int, num_robustness_samples: int) -> dict[str, object]:
    print(f"Starting classifier analysis for {model_dir}")
    analysis_root = model_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    confidence_root = model_dir / "confidence_analysis"
    confidence_root.mkdir(parents=True, exist_ok=True)
    manifest = _get_manifest(model_dir)
    probabilities, logits, targets, checkpoint = _load_model_and_run(model_dir, manifest, batch_size, num_workers)
    print(f"Loaded model and ran inference on {len(targets)} samples")
    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    # Confidence analysis
    plot_confidence_distribution(probabilities, targets, confidence_root / "confidence_histogram.png", CLASS_NAMES)
    plot_confidence_correct_vs_incorrect(probabilities, targets, predictions, confidence_root / "correct_vs_incorrect_confidence.png", CLASS_NAMES)
    save_low_confidence_failures(probabilities, targets, predictions, manifest, confidence_root / "low_confidence_examples")
    # Threshold sweep
    sweep_results = threshold_sweep(probabilities, targets, analysis_root / "threshold_sweep.png")
    _save_json(sweep_results, analysis_root / "threshold_sweep_results.json")
    # Calibration
    calibration_metrics = plot_calibration(probabilities, targets, analysis_root / "calibration.png")
    _save_json(calibration_metrics, analysis_root / "calibration_metrics.json")
    # Robustness
    robustness_metrics = robustness_test(model_dir, manifest, batch_size, num_workers, num_robustness_samples)
    _save_json(robustness_metrics, analysis_root / "robustness_metrics.json")
    # Failure analysis
    failure_summary = _generate_failure_analysis(probabilities, targets, predictions, manifest, model_dir)
    combined = {
        "sweep": sweep_results,
        "calibration": calibration_metrics,
        "robustness": robustness_metrics,
        "failure_categories": failure_summary,
    }
    output_path = analysis_root / "analysis_results.json"
    _save_json(combined, output_path)
    print(f"Classifier analysis complete. Results saved to {output_path}")
    return combined

def run(args: Namespace | None = None) -> dict[str, object]:
    args = args or parse_args()
    model_dir = Path(args.model_dir)
    return analyze_classifier(
        model_dir=model_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_robustness_samples=args.num_robustness_samples,
    )

if __name__ == "__main__":
    run()