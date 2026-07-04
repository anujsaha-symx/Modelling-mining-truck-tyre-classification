from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GATEKEEPER_MODEL = "v3"

GATEKEEPER_CHECKPOINTS = {
    "old": str(PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_frcnn_old.pt"),
    "v2":  str(PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_frcnn.pt"),
    "v3":  str(PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_frcnn_v3.pt"),
}
