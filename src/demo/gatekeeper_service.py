import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.demo.utils import GATEKEEPER_CKPT, GATEKEEPER_CLASSES, get_device, check_checkpoint

def _get_frcnn_model(num_classes=3):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def _load_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

class GatekeeperService:
    def __init__(self, checkpoint_path=None, score_threshold=0.5, margin=0.2):
        self.device = get_device()
        self.score_threshold = score_threshold
        self.margin = margin
        ckpt = checkpoint_path or GATEKEEPER_CKPT
        check_checkpoint(ckpt)

        self.model = _get_frcnn_model(num_classes=3).to(self.device)
        _load_checkpoint(self.model, str(ckpt), device=self.device)
        self.model.eval()

        self.transform = T.Compose([T.ToTensor()])

    @torch.no_grad()
    def predict(self, image):
        img_tensor = self.transform(image).to(self.device)
        output = self.model([img_tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        mask = scores >= self.score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            detections.append({
                "class": GATEKEEPER_CLASSES.get(int(label), f"Class_{int(label)}"),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            })

        return detections

    def decide(self, detections):
        tire_dets = [d for d in detections if d["class"] == "Tire"]
        non_tire_dets = [d for d in detections if d["class"] == "Non-Tire"]

        if not tire_dets:
            return {"is_tire": False, "reason": "No tyre detection", "confidence": 0.0}

        best_tire = max(tire_dets, key=lambda d: d["confidence"])
        best_non_tire = max(non_tire_dets, key=lambda d: d["confidence"]) if non_tire_dets else None

        if best_tire["confidence"] < self.score_threshold:
            return {"is_tire": False, "reason": "Below confidence threshold", "confidence": best_tire["confidence"]}

        if best_non_tire and best_non_tire["confidence"] > best_tire["confidence"] - self.margin:
            return {
                "is_tire": False,
                "reason": "Non-Tire competes with Tire detection",
                "confidence": best_tire["confidence"],
                "non_tire_conf": best_non_tire["confidence"],
            }

        return {"is_tire": True, "confidence": best_tire["confidence"], "reason": "Tyre verified"}