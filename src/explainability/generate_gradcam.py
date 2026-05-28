from __future__ import annotations
import json
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.classification_dataset import (  # noqa: E402
    CLASS_NAMES,
    TyreClassificationDataset,
    build_classification_manifests,
    build_eval_transforms,
)
from src.explainability.gradcam_utils import (  # noqa: E402
    CLASS_NAMES_TUPLE,
    GradCAMConfig,
    compute_attention_metrics,
    generate_gradcam_for_dataset,
)
from src.models.classifiers import load_classifier_checkpoint  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification"
SPLITS_ROOT = PROJECT_ROOT / "datasets" / "splits"
DEDUPLICATED_MANIFEST = PROJECT_ROOT / "outputs" / "metrics" / "deduplicated_dataset.csv"
CROPS_ROOT = PROJECT_ROOT / "outputs" / "crops"
GRADCAM_ROOT = PROJECT_ROOT / "outputs" / "gradcam"

@dataclass
class AggregateAnalysis:
    total_samples: int = 0
    suspicious_count: int = 0
    avg_corner_fraction: float = 0.0
    avg_edge_fraction: float = 0.0
    avg_com_offset: float = 0.0
    avg_activation_spread: float = 0.0
    per_category: dict[str, dict[str, float]] = None
    def __post_init__(self):
        if self.per_category is None:
            self.per_category = {}
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate GradCAM explainability visualizations for tyre wear classifier.")
    parser.add_argument("--weights", required=True, help="Path to classifier checkpoint (best_model.pt).")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples processed.")
    parser.add_argument("--max-per-category", type=int, default=30, help="Max visualizations per category.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay transparency (0-1).")
    parser.add_argument("--smooth-sigma", type=float, default=0.0, help="Heatmap Gaussian blur sigma (0 = no smoothing).")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--low-conf-margin", type=float, default=0.20, help="Margin from threshold for low confidence.")
    return parser.parse_args()
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
def _aggregate_results(results: list) -> AggregateAnalysis:
    if not results:
        return AggregateAnalysis()
    analysis = AggregateAnalysis(total_samples=len(results))
    total_corner = 0.0
    total_edge = 0.0
    total_offset = 0.0
    total_spread = 0.0
    category_data: dict[str, list[dict[str, float]]] = {}
    for r in results:
        m = r.attention_metrics
        if m is None:
            continue
        if m.is_suspicious:
            analysis.suspicious_count += 1
        total_corner += m.corner_activation_fraction
        total_edge += m.edge_activation_fraction
        total_offset += m.com_offset_from_center
        total_spread += m.activation_spread
        cat = r.category
        if cat not in category_data:
            category_data[cat] = []
        category_data[cat].append({
            "corner": m.corner_activation_fraction,
            "edge": m.edge_activation_fraction,
            "offset": m.com_offset_from_center,
            "spread": m.activation_spread,
        })
    n = len(results)
    analysis.avg_corner_fraction = total_corner / n
    analysis.avg_edge_fraction = total_edge / n
    analysis.avg_com_offset = total_offset / n
    analysis.avg_activation_spread = total_spread / n
    for cat, metrics_list in category_data.items():
        arr = np.array([[v["corner"], v["edge"], v["offset"], v["spread"]] for v in metrics_list])
        analysis.per_category[cat] = {
            "count": len(metrics_list),
            "avg_corner_fraction": float(arr[:, 0].mean()),
            "avg_edge_fraction": float(arr[:, 1].mean()),
            "avg_com_offset": float(arr[:, 2].mean()),
            "avg_spread": float(arr[:, 3].mean()),
            "suspicious_count": int(sum(1 for v in metrics_list if v["corner"] > 0.25 or v["edge"] > 0.50)),
        }
    return analysis

def run(args: Namespace | None = None) -> list:
    args = args or parse_args()
    device = _device()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    print(f"[GradCAM] Loading checkpoint: {weights_path}")
    model, checkpoint = load_classifier_checkpoint(weights_path=weights_path, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    model.eval()
    print(f"[GradCAM] Model: {model.model_name}, image_size: {image_size}, device: {device}")
    output_root = Path(args.output_dir) if args.output_dir else GRADCAM_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[GradCAM] Building {args.split} manifest...")
    manifests, summaries = build_classification_manifests(
        splits_root=SPLITS_ROOT,
        deduplicated_manifest_path=DEDUPLICATED_MANIFEST,
        crops_root=CROPS_ROOT,
    )
    manifest = manifests[args.split]
    print(f"[GradCAM] Test samples: {len(manifest)}")
    dataset = TyreClassificationDataset(
        manifest=manifest,
        transform=build_eval_transforms(image_size=image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    config = GradCAMConfig(
        alpha=args.alpha,
        smooth_sigma=args.smooth_sigma,
        max_per_category=args.max_per_category,
    )
    print(f"[GradCAM] Generating GradCAM visualizations...")
    results = generate_gradcam_for_dataset(
        model=model,
        loader=loader,
        output_root=output_root,
        device=device,
        config=config,
        max_samples=args.max_samples,
    )
    print(f"[GradCAM] Processed {len(results)} samples.")
    results_json_path = output_root / "gradcam_results.json"
    serializable = [
        {
            "image_path": r.image_path,
            "true_label": r.true_label,
            "predicted_label": r.predicted_label,
            "bad_class_confidence": r.bad_class_confidence,
            "category": r.category,
            "attention_metrics": {
                "center_of_mass_x": r.attention_metrics.center_of_mass_x if r.attention_metrics else None,
                "center_of_mass_y": r.attention_metrics.center_of_mass_y if r.attention_metrics else None,
                "com_offset_from_center": r.attention_metrics.com_offset_from_center if r.attention_metrics else None,
                "corner_activation_fraction": r.attention_metrics.corner_activation_fraction if r.attention_metrics else None,
                "edge_activation_fraction": r.attention_metrics.edge_activation_fraction if r.attention_metrics else None,
                "activation_spread": r.attention_metrics.activation_spread if r.attention_metrics else None,
                "is_suspicious": r.attention_metrics.is_suspicious if r.attention_metrics else False,
                "suspicious_reasons": r.attention_metrics.suspicious_reasons if r.attention_metrics else [],
            }
            if r.attention_metrics
            else None,
        }
        for r in results
    ]
    _save_json(serializable, results_json_path)
    print(f"[GradCAM] Results saved to {results_json_path}")
    susp_count = sum(1 for r in results if r.attention_metrics and r.attention_metrics.is_suspicious)
    print(f"[GradCAM] Done. {susp_count} suspicious attention patterns detected.")
    return results

if __name__ == "__main__":
    run()