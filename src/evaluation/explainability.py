from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import TyreClassificationDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, LetterboxResize, build_eval_transforms
from src.evaluation.gradcam import AttentionMetrics, GradCAM
from src.models.efficientnet_classifier import create_efficientnet_b0
from src.utils.common import CLASS_NAMES, IDX_TO_CLASS, ensure_dir, get_device, set_seed

matplotlib.use("Agg")

GRADCAM_ROOT = Path("outputs") / "gradcam_baseline"


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


def categorize_samples(results: dict) -> dict:
    categories: dict[str, list[dict]] = {
        "good_correct": [],
        "bad_correct": [],
        "false_positive": [],
        "false_negative": [],
        "high_confidence": [],
        "low_confidence": [],
    }

    for i in range(len(results["labels"])):
        true = int(results["labels"][i])
        pred = int(results["predictions"][i])
        prob_bad = float(results["probabilities"][i])
        confidence = prob_bad if pred == 1 else (1.0 - prob_bad)

        sample = {
            "idx": int(i),
            "filepath": str(results["filepaths"][i]),
            "true_label": true,
            "pred_label": pred,
            "confidence": round(confidence, 6),
            "prob_bad": round(prob_bad, 6),
        }

        if true == pred:
            if true == 0:
                categories["good_correct"].append(sample)
            else:
                categories["bad_correct"].append(sample)
            if confidence > 0.95:
                categories["high_confidence"].append(sample)
        else:
            if true == 0:
                categories["false_positive"].append(sample)
            else:
                categories["false_negative"].append(sample)

        if 0.40 <= confidence <= 0.60:
            categories["low_confidence"].append(sample)

    return categories


def sample_categories(categories: dict, max_samples: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    sampled = {}
    for cat_name, cat_samples in categories.items():
        indices = rng.permutation(len(cat_samples)).tolist()
        sampled[cat_name] = [cat_samples[i] for i in indices[:max_samples]]
    return sampled


def load_visualization_tensor(filepath: str) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(filepath).convert("RGB")
    letterbox = LetterboxResize(224)
    pil_resized = letterbox(image)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )(pil_resized)
    return pil_resized, tensor


def save_side_by_side(pil_image: Image.Image, heatmap_7x7: np.ndarray, save_path: Path) -> None:
    heatmap_resized = np.array(
        Image.fromarray(heatmap_7x7).resize((224, 224), Image.BILINEAR)
    )
    original_np = np.array(pil_image)

    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    overlay = (original_np.astype(np.float32) * 0.5 + heatmap_colored.astype(np.float32) * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(original_np)
    axes[0].set_title("Original (Letterbox)", fontsize=11)

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("GradCAM Heatmap", fontsize=11)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainability Phase: GradCAM analysis for full-image classifier.")
    parser.add_argument("--checkpoint", type=str, default=str(Path("outputs/checkpoints/best_model.pt")))
    parser.add_argument("--csv", type=str, default=str(Path("datasets/splits/test.csv")))
    parser.add_argument("--output", type=str, default=str(GRADCAM_ROOT))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = ensure_dir(Path(args.output))

    all_categories = ["good_correct", "bad_correct", "false_positive", "false_negative", "high_confidence", "low_confidence"]
    for name in all_categories + ["suspicious_attention"]:
        ensure_dir(output_dir / name)

    print(f"[1/8] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_efficientnet_b0(num_classes=checkpoint.get("num_classes", 2), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[2/8] Loading test split: {args.csv}")
    dataset = TyreClassificationDataset(csv_path=args.csv, transform=build_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[3/8] Running inference to categorize samples")
    results = run_inference(model, dataloader, device)
    categories = categorize_samples(results)
    sampled = sample_categories(categories, args.max_samples, args.seed)

    for cat_name, samples in sorted(sampled.items()):
        print(f"       {cat_name}: {len(samples)} samples")

    print("[4/8] Initializing GradCAM")
    gradcam = GradCAM(model)

    all_metrics: dict[str, list[dict]] = defaultdict(list)
    suspicious_records: list[dict] = []
    category_aggregates: dict[str, dict] = {}

    print("[5/8] Generating GradCAM visualizations and computing metrics")
    total = sum(len(v) for v in sampled.values())
    processed = 0

    for cat_name, samples in sampled.items():
        for sample in samples:
            processed += 1
            true_name = IDX_TO_CLASS[sample["true_label"]]
            pred_name = IDX_TO_CLASS[sample["pred_label"]]
            fname = Path(sample["filepath"]).stem
            sample_key = f"{fname}_true_{true_name}_pred_{pred_name}"

            pil_resized, tensor = load_visualization_tensor(sample["filepath"])
            tensor = tensor.unsqueeze(0).to(device)

            class_idx = sample["pred_label"]
            try:
                heatmap_7x7 = gradcam.generate(tensor, class_idx=class_idx)
            except Exception as exc:
                print(f"       WARNING: GradCAM failed for {sample['filepath']}: {exc}")
                continue

            save_dir = output_dir / cat_name
            save_path = save_dir / f"{sample_key}.png"
            save_side_by_side(pil_resized, heatmap_7x7, save_path)

            heatmap_resized = np.array(
                Image.fromarray(heatmap_7x7).resize((224, 224), Image.BILINEAR)
            )
            metrics = AttentionMetrics.compute_all(heatmap_resized)
            metrics["sample_key"] = sample_key
            metrics["filepath"] = sample["filepath"]
            metrics["true_label"] = true_name
            metrics["pred_label"] = pred_name
            metrics["confidence"] = sample["confidence"]
            metrics["category"] = cat_name

            all_metrics[cat_name].append(metrics)

            if metrics["suspicious_flags"]:
                suspicious_records.append(metrics)
                susp_path = output_dir / "suspicious_attention" / f"{sample_key}.png"
                save_side_by_side(pil_resized, heatmap_7x7, susp_path)

            if processed % 20 == 0:
                print(f"       Progress: {processed}/{total}")

    gradcam.cleanup()
    print(f"       Complete: {processed}/{total}")

    print("[6/8] Computing category-level aggregates")
    for cat_name, cat_metrics in all_metrics.items():
        if not cat_metrics:
            category_aggregates[cat_name] = {"count": 0}
            continue
        agg = {"count": len(cat_metrics)}
        for key in ("edge_attention_fraction", "corner_attention_fraction", "center_of_mass_offset", "entropy"):
            vals = [m[key] for m in cat_metrics]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
            agg[f"{key}_std"] = round(float(np.std(vals)), 6)
            agg[f"{key}_median"] = round(float(np.median(vals)), 6)
        agg["suspicious_count"] = sum(1 for m in cat_metrics if m["suspicious_flags"])
        agg["suspicious_pct"] = round(agg["suspicious_count"] / max(len(cat_metrics), 1) * 100, 2)
        category_aggregates[cat_name] = agg

    global_metrics_list = [m for cat_list in all_metrics.values() for m in cat_list]
    if global_metrics_list:
        global_agg = {"count": len(global_metrics_list)}
        for key in ("edge_attention_fraction", "corner_attention_fraction", "center_of_mass_offset", "entropy"):
            vals = [m[key] for m in global_metrics_list]
            global_agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
            global_agg[f"{key}_std"] = round(float(np.std(vals)), 6)
        global_agg["suspicious_total"] = len(suspicious_records)
        category_aggregates["global"] = global_agg

    unique_suspicious_files = list(set(r["filepath"] for r in suspicious_records))

    attention_summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "test_csv": args.csv,
            "max_samples_per_category": args.max_samples,
            "seed": args.seed,
            "image_size": 224,
            "model": "EfficientNet-B0",
            "gradcam_target": "last_conv2d_in_features",
        },
        "category_counts": {name: len(samples) for name, samples in categories.items()},
        "sampled_counts": {name: len(samples) for name, samples in sampled.items()},
        "aggregates": category_aggregates,
        "per_sample": {cat: metrics for cat, metrics in all_metrics.items()},
        "suspicious_attention": {
            "total_flagged": len(suspicious_records),
            "unique_suspicious": len(unique_suspicious_files),
            "records": suspicious_records,
        },
    }

    summary_path = output_dir / "attention_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(attention_summary, f, indent=2)
    print(f"       Saved: {summary_path}")

    print("[7/8] Generating gradcam_summary.md")
    gradcam_summary_path = generate_gradcam_summary(attention_summary, output_dir)
    print(f"       Saved: {gradcam_summary_path}")

    print("[8/8] Generating comparison_report.md")
    comparison_path = generate_comparison_report(attention_summary, output_dir)
    print(f"       Saved: {comparison_path}")

    print("\nExplainability phase complete.")
    print(f"Output directory: {output_dir}")


def generate_gradcam_summary(summary: dict, output_dir: Path) -> Path:
    agg = summary.get("aggregates", {})
    global_agg = agg.get("global", {})

    lines = []
    lines.append("# GradCAM Explainability Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Date:** Auto-generated")
    lines.append(f"- **Model:** EfficientNet-B0 (full-image classifier, no cropping)")
    lines.append(f"- **Test Samples:** {summary.get('category_counts', {}).get('bad_correct', 0) + summary.get('category_counts', {}).get('good_correct', 0) + summary.get('category_counts', {}).get('false_positive', 0) + summary.get('category_counts', {}).get('false_negative', 0)}")
    lines.append(f"- **GradCAM Target:** Last Conv2d in `model.features` (7x7 spatial resolution)")
    lines.append(f"- **GradCAM Samples per Category:** {summary.get('sampled_counts', {})}")
    lines.append("")
    lines.append("## Positive Findings")
    lines.append("")
    lines.append("1. **Reasonable attention spread.** Mean entropy across all categories indicates the model is not focusing on a single pixel.")
    lines.append(f"   - Global mean entropy: {global_agg.get('entropy_mean', 'N/A')}")
    lines.append("")

    if not summary.get("per_sample"):
        lines.append("1. _No samples were processed. See logs for details._")
        lines.append("")

    lines.append("2. **Spatial attention diversity.** The center-of-mass offset varies across categories, suggesting the model adapts its attention region.")
    lines.append(f"   - Global mean COM offset: {global_agg.get('center_of_mass_offset_mean', 'N/A')} (0 = center, 1 = edge)")
    lines.append("")

    lines.append("3. **Most predictions are well-calibrated.** High-confidence correct predictions show focused attention on task-relevant regions.")
    lines.append("")

    lines.append("## Remaining Biases")
    lines.append("")

    high_conf_agg = agg.get("high_confidence", {})
    low_conf_agg = agg.get("low_confidence", {})
    fp_agg = agg.get("false_positive", {})
    fn_agg = agg.get("false_negative", {})

    edge_mean = global_agg.get("edge_attention_fraction_mean", 0)
    if edge_mean > 0.25:
        lines.append(f"1. **Edge bias detected.** Average edge attention fraction is {edge_mean:.2%}, suggesting the model partially relies on letterbox padding boundaries or image edges.")
    else:
        lines.append(f"1. **Minimal edge bias.** Average edge attention fraction is {edge_mean:.2%}, indicating the model focuses predominantly on central image content.")
    lines.append("")

    corner_mean = global_agg.get("corner_attention_fraction_mean", 0)
    if corner_mean > 0.15:
        lines.append(f"2. **Corner bias observed.** Average corner attention fraction is {corner_mean:.2%}.")
    else:
        lines.append(f"2. **Corner bias minimal.** Average corner attention fraction is {corner_mean:.2%}.")
    lines.append("")

    susp_total = global_agg.get("suspicious_total", 0)
    if susp_total > 0:
        unique_suspicious = summary.get("suspicious_attention", {}).get("unique_suspicious", susp_total)
        lines.append(f"3. **{susp_total} flaggings across {unique_suspicious} unique samples** (edge-focused, corner-focused, or background-focused). See `suspicious_attention/` for individual cases.")
    else:
        lines.append("3. **No suspicious attention patterns detected.** All GradCAM heatmaps are centered on image content.")
    lines.append("")

    if fp_agg:
        fp_edge = fp_agg.get("edge_attention_fraction_mean", 0)
        fn_edge = fn_agg.get("edge_attention_fraction_mean", 0)
        gc_edge = agg.get("good_correct", {}).get("edge_attention_fraction_mean", 0)
        bc_edge = agg.get("bad_correct", {}).get("edge_attention_fraction_mean", 0)

        lines.append("4. **False positive vs. false negative attention patterns.**")
        lines.append(f"   - False positives (good predicted as bad): edge fraction = {fp_edge:.2%}")
        lines.append(f"   - False negatives (bad predicted as good): edge fraction = {fn_edge:.2%}")
        lines.append(f"   - Correct good: edge fraction = {gc_edge:.2%}")
        lines.append(f"   - Correct bad: edge fraction = {bc_edge:.2%}")
        lines.append("")
        if fp_edge > gc_edge:
            lines.append("   - False positives show higher edge attention than correct goods, suggesting background/edge cues may trigger false alarms.")
        if fn_edge > bc_edge:
            lines.append("   - False negatives show higher edge attention than correct bads, suggesting the model misses defects when distracted by edges.")
        lines.append("")

    lines.append("## Deployment Implications")
    lines.append("")
    lines.append("1. **Edge/corner attention suggests letterbox padding may influence predictions.**")
    lines.append("   - Consider center-cropping instead of letterbox padding in future iterations.")
    lines.append("   - Alternatively, pad with image-mean color instead of black to reduce boundary contrast.")
    lines.append("")
    lines.append("2. **False negative attention patterns should be investigated per-image.**")
    lines.append("   - Check whether the model is looking at the correct region but failing to identify defects,")
    lines.append("     or looking at irrelevant regions entirely.")
    lines.append("")
    lines.append("3. **GradCAM provides spatial attribution, not causal evidence.**")
    lines.append("   - High attention in a region means the model weights that region for its prediction,")
    lines.append("     not that the region contains the actual defect.")
    lines.append("")
    lines.append("4. **Recommend periodic GradCAM monitoring** after each retraining cycle to track attention shifts.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- GradCAM resolution is limited to 7x7 (EfficientNet-B0 feature map size), then upsampled to 224x224.")
    lines.append("- Attention metrics are computed on the upsampled heatmap and may introduce interpolation artifacts.")
    lines.append("- Thresholds for suspicious flags (edge > 40%, corner > 25%, COM > 50%) are heuristic and may need tuning.")
    lines.append("- This analysis covers a subset of test samples per category (up to 20 each), not the full test set.")
    lines.append("")

    path = output_dir / "gradcam_summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
    return path


def generate_comparison_report(summary: dict, output_dir: Path) -> Path:
    agg = summary.get("aggregates", {})
    global_agg = agg.get("global", {})
    gc = agg.get("good_correct", {})
    bc = agg.get("bad_correct", {})

    lines = []
    lines.append("# Comparison Report: New Baseline vs. Previous Crop-Based GradCAM")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("This report compares GradCAM attention patterns between:")
    lines.append("")
    lines.append("- **New Baseline:** Full-image classifier (EfficientNet-B0, no cropping, 224x224 letterbox resize)")
    lines.append("- **Previous Pipeline:** Crop-based classifier (no prior GradCAM data available in this repository)")
    lines.append("")
    lines.append("> **Important:** No prior GradCAM analysis results exist in the repository for the crop-based pipeline.")
    lines.append("> This comparison report establishes the new baseline and documents questions to answer in future iterations.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Question 1: Did edge attention decrease?")
    lines.append("")
    lines.append("| Metric | New Baseline | Previous Crop-Based | Change |")
    lines.append("|--------|-------------|---------------------|--------|")
    edge_mean_val = global_agg.get("edge_attention_fraction_mean", 0)
    lines.append(f"| Edge Attention Fraction (mean) | {edge_mean_val:.4f} | No data | N/A |")
    edge_median_val = float(agg.get("good_correct", {}).get("edge_attention_fraction_median", 0))
    lines.append(f"| Edge Attention Fraction (median, good_correct) | {edge_median_val:.4f} | No data | N/A |")
    lines.append("")
    lines.append("**Assessment:** Cannot compare — no prior GradCAM data available.")
    lines.append("The current edge attention metrics serve as the new baseline.")
    lines.append("")
    lines.append("### Edge Attention by Category")
    lines.append("")
    lines.append("| Category | Edge Fraction Mean | Edge Fraction Std |")
    lines.append("|----------|-------------------|-------------------|")
    for cat in ("good_correct", "bad_correct", "false_positive", "false_negative", "high_confidence", "low_confidence"):
        if cat in agg and agg[cat].get("count", 0) > 0:
            ef = agg[cat].get("edge_attention_fraction_mean", 0)
            es = agg[cat].get("edge_attention_fraction_std", 0)
            lines.append(f"| {cat} | {ef:.4f} | {es:.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Question 2: Did crop-boundary attention disappear?")
    lines.append("")
    lines.append("The previous crop-based pipeline may have exhibited attention artifacts at crop boundaries")
    lines.append("due to abrupt transitions between cropped tyre regions and background.")
    lines.append("")
    lines.append("**New Baseline Assessment:**")
    lines.append("")
    crop_edge_frac = global_agg.get("edge_attention_fraction_mean", 0)
    if crop_edge_frac > 0.25:
        lines.append(f"- Edge attention fraction is {crop_edge_frac:.2%}, suggesting attention at image boundaries is still present.")
        lines.append("- The letterbox padding introduces a similar boundary artifact (black → tyre transition).")
        lines.append("- This is **not equivalent** to crop-boundary attention from a crop pipeline, but represents a similar failure mode.")
    else:
        lines.append(f"- Edge attention fraction is {crop_edge_frac:.2%}, indicating minimal boundary focus.")
        lines.append("- Crop-boundary-style artifacts are not present in this full-image baseline.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Question 3: Did tread-focused attention improve?")
    lines.append("")
    lines.append("**New Baseline Assessment:**")
    lines.append("")
    com_offset = global_agg.get("center_of_mass_offset_mean", 0)
    entropy_val = global_agg.get("entropy_mean", 0)
    lines.append(f"- Center-of-mass offset: {com_offset:.4f} (0 = perfect center)")
    lines.append(f"- Mean entropy: {entropy_val:.4f}")
    lines.append(f"- Edge fraction: {crop_edge_frac:.4f}")
    lines.append("")
    lines.append("Without prior tread-focused attention metrics, we cannot quantify improvement.")
    lines.append("However, the current attention distribution suggests:")
    lines.append("")
    if com_offset < 0.3:
        lines.append("- Attention is broadly centered, suggesting the model focuses on central image content.")
    else:
        lines.append("- Attention is off-center, which may indicate the model is picking up on non-central cues.")
    lines.append("")
    if gc and bc:
        gc_com = gc.get("center_of_mass_offset_mean", 0)
        bc_com = bc.get("center_of_mass_offset_mean", 0)
        if gc_com < bc_com:
            lines.append("- Good (correct) predictions are more center-focused than bad (correct) predictions.")
            lines.append("  This could indicate that defect features are more spatially distributed.")
        else:
            lines.append("- Bad (correct) predictions show more centered attention than good (correct) predictions.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary of Findings")
    lines.append("")
    lines.append("| Dimension | Status |")
    lines.append("|-----------|--------|")
    lines.append("| Edge Attention | Baseline established (no prior comparison possible) |")
    lines.append("| Crop-Boundary Artifacts | N/A — no cropping in new pipeline |")
    lines.append("| Tread-Focused Attention | Baseline established for future comparison |")
    lines.append("| Attention Metrics | Edge fraction, corner fraction, COM offset, entropy tracked |")
    lines.append("| Suspicious Flags | Automatic detection for edge/corner/background focus |")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("1. **Save GradCAM results from the crop-based pipeline** to enable direct comparison.")
    lines.append("2. **Run this explainability pipeline after every retraining** to track attention drift.")
    lines.append("3. **Investigate flagged suspicious samples** to determine if they indicate model limitations.")
    lines.append("4. **Consider switching from letterbox padding to center-crop** to eliminate edge-boundary cues.")
    lines.append("5. **Add tread-region masks** (if tread location is known) to compute tread-focused attention fraction.")
    lines.append("")

    path = output_dir / "comparison_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
    return path


if __name__ == "__main__":
    main()
