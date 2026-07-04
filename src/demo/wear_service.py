import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.demo.utils import WEAR_CKPT, get_device, check_checkpoint


CLASS_NAMES = {1: "Good-Tire", 2: "Bad-Tire", 3: "Non-Tire"}
DISPLAY_NAMES = {"Good-Tire": "Tire", "Bad-Tire": "Cut", "Non-Tire": "Non-Tire"}


def _get_wear_model(num_classes=4):
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


class WearService:
    def __init__(self, checkpoint_path=None, score_threshold=0.15):
        self.device = get_device()
        self.score_threshold = score_threshold
        ckpt = checkpoint_path or WEAR_CKPT
        check_checkpoint(ckpt)

        self.model = _get_wear_model(num_classes=4).to(self.device)
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

        return boxes, scores, labels

    def classify_output(self, boxes, scores, labels):
        has_cut = False
        has_tire = False
        has_nontire = False
        detections = []

        for box, score, label in zip(boxes, scores, labels):
            internal_name = CLASS_NAMES.get(int(label), "Non-Tire")
            display_name = DISPLAY_NAMES.get(internal_name, internal_name)
            x1, y1, x2, y2 = box
            detections.append({
                "class": display_name,
                "internal_class": internal_name,
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            })
            if internal_name == "Bad-Tire":
                has_cut = True
            elif internal_name == "Good-Tire":
                has_tire = True
            elif internal_name == "Non-Tire":
                has_nontire = True

        if has_cut:
            final_class = "Bad-Tire"
            reason = "Cut detected on tyre surface"
        elif has_tire:
            final_class = "Good-Tire"
            reason = "Tyre detected, no cuts found"
        elif has_nontire:
            final_class = "Non-Tire"
            reason = "Non-Tyre content detected"
        else:
            final_class = "Non-Tire"
            reason = "No objects detected above threshold"
            detections = [{"class": "Non-Tire", "confidence": 1.0, "bbox": [0, 0, 0, 0]}]

        return {
            "final_class": final_class,
            "reason": reason,
            "detections": detections,
        }
