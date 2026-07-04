import io
import sys
import os
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GATEKEEPER_CKPT = PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_frcnn.pt"
GATEKEEPER_V3_CKPT = PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_frcnn_v3.pt"
WEAR_CKPT = PROJECT_ROOT / "outputs" / "wear_detection" / "checkpoints" / "best_wear_frcnn.pt"
WEAR_V2_CKPT = PROJECT_ROOT / "outputs" / "wear_detection_v2" / "checkpoints" / "best_frcnn_v2.pt"

GATEKEEPER_CLASSES = {1: "Tire", 2: "Non-Tire"}
WEAR_CLASSES = {1: "Good-Tire", 2: "Bad-Tire", 3: "Non-Tire"}
WEAR_V2_CLASSES = {1: "Tire", 2: "Cut", 3: "Non-Tire"}

GATEKEEPER_METRICS = {
    "Precision": 0.9887,
    "Recall": 0.9972,
    "F1": 0.9929,
    "Accuracy": 0.9974,
}

WEAR_METRICS_V2 = {
    "OverallAccuracy": 0.8404,
    "Tire": {"Precision": 0.9361, "Recall": 0.9812, "F1": 0.9581},
    "Cut": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
    "Non-Tire": {"Precision": 0.8788, "Recall": 0.9355, "F1": 0.9062},
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(file_or_path):
    if isinstance(file_or_path, (str, Path)):
        return Image.open(file_or_path).convert("RGB")
    return Image.open(io.BytesIO(file_or_path.read())).convert("RGB")


def check_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
