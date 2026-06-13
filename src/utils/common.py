from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ["good", "bad"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets"
PROCESSED_ROOT = DATASET_ROOT / "processed"
ANNOTATED_ROOT = DATASET_ROOT / "annotated"
NEGATIVE_ROOT = DATASET_ROOT / "negative"
SPLITS_ROOT = DATASET_ROOT / "splits"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINTS_ROOT = OUTPUTS_ROOT / "checkpoints"
METRICS_ROOT = OUTPUTS_ROOT / "metrics"
FIGURES_ROOT = OUTPUTS_ROOT / "figures"
PREDICTIONS_ROOT = OUTPUTS_ROOT / "predictions"
VERIFICATION_ROOT = OUTPUTS_ROOT / "verification"
VERIFICATION_CHECKPOINTS_ROOT = VERIFICATION_ROOT / "checkpoints"
VERIFICATION_METRICS_ROOT = VERIFICATION_ROOT / "metrics"
VERIFICATION_FIGURES_ROOT = VERIFICATION_ROOT / "figures"
VERIFICATION_PREDICTIONS_ROOT = VERIFICATION_ROOT / "predictions"
VERIFICATION_SPLITS_ROOT = DATASET_ROOT / "verification_splits"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs() -> None:
    for path in [
        SPLITS_ROOT,
        NEGATIVE_ROOT,
        CHECKPOINTS_ROOT,
        METRICS_ROOT,
        FIGURES_ROOT,
        PREDICTIONS_ROOT,
        VERIFICATION_ROOT,
        VERIFICATION_CHECKPOINTS_ROOT,
        VERIFICATION_METRICS_ROOT,
        VERIFICATION_FIGURES_ROOT,
        VERIFICATION_PREDICTIONS_ROOT,
        VERIFICATION_SPLITS_ROOT,
    ]:
        ensure_dir(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
