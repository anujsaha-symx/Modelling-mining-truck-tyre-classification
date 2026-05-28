from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES_TUPLE = ("good", "bad")

@dataclass
class GradCAMConfig:
    alpha: float = 0.5
    colormap: int = cv2.COLORMAP_JET
    figure_size: tuple[int, int] = (16, 5)
    figure_dpi: int = 150
    smooth_sigma: float = 0.0
    max_per_category: int = 30
    suspicious_corner_threshold: float = 0.25
    suspicious_edge_threshold: float = 0.50

@dataclass
class AttentionMetrics:
    center_of_mass_x: float
    center_of_mass_y: float
    com_offset_from_center: float
    corner_activation_fraction: float
    edge_activation_fraction: float
    activation_spread: float
    activation_entropy: float
    max_activation_value: float
    mean_activation_value: float
    is_suspicious: bool = False
    suspicious_reasons: list[str] = field(default_factory=list)

@dataclass
class GradCAMResult:
    image_path: str
    true_label: str
    predicted_label: str
    bad_class_confidence: float
    category: str
    heatmap_normalized: np.ndarray | None = None
    attention_metrics: AttentionMetrics | None = None

class _ActivationGradientStore:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
    def forward_hook(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()
    def backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0].detach()

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._store = _ActivationGradientStore()
        self._handles: list[Any] = []
        self._register_hooks()
    def _register_hooks(self) -> None:
        self._handles.append(
            self.target_layer.register_forward_hook(self._store.forward_hook)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._store.backward_hook)
        )
    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)
        activations = self._store.activations
        gradients = self._store.gradients
        if activations is None or gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations or gradients.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy().astype(np.float32)
    def generate_for_class(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        return self.generate(input_tensor, target_class=target_class)
    def generate_batch(
        self,
        input_tensors: torch.Tensor,
        target_classes: list[int] | None = None,
        batch_size: int = 1,
    ) -> list[np.ndarray]:
        cams: list[np.ndarray] = []
        effective_bs = min(batch_size, input_tensors.size(0))
        for start in range(0, input_tensors.size(0), effective_bs):
            end = min(start + effective_bs, input_tensors.size(0))
            batch = input_tensors[start:end]
            self.model.zero_grad()
            output = self.model(batch)
            if target_classes is None:
                batch_targets = output.argmax(dim=1).cpu().tolist()
            else:
                batch_targets = target_classes[start:end]
            for i in range(len(batch)):
                one_hot = torch.zeros_like(output)
                one_hot[i, batch_targets[i]] = 1.0
                retain = i < len(batch) - 1
                output.backward(gradient=one_hot, retain_graph=retain)
                act = self._store.activations[i : i + 1]
                grad = self._store.gradients
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam_map = (weights * act).sum(dim=1, keepdim=True)
                cam_map = F.relu(cam_map)
                c_min = float(cam_map.min())
                c_max = float(cam_map.max())
                if c_max - c_min > 1e-7:
                    cam_map = (cam_map - c_min) / (c_max - c_min)
                else:
                    cam_map = torch.zeros_like(cam_map)
                cam_map = F.interpolate(
                    cam_map,
                    size=input_tensors.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                cams.append(cam_map.squeeze().cpu().numpy().astype(np.float32))
        return cams

class GuidedBackprop:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[Any] = []
    def _guided_relu_hook(self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
        if isinstance(module, nn.ReLU):
            return (F.relu(grad_output[0]),)
    def enable(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self._handles.append(
                    module.register_full_backward_hook(self._guided_relu_hook)
                )
    def disable(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    def generate_saliency(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)
        saliency = input_tensor.grad.data
        saliency = saliency.abs().max(dim=1, keepdim=True)[0]
        s_min = float(saliency.min())
        s_max = float(saliency.max())
        if s_max - s_min > 1e-7:
            saliency = (saliency - s_min) / (s_max - s_min)
        return saliency.squeeze().cpu().numpy().astype(np.float32)

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    h_min = float(heatmap.min())
    h_max = float(heatmap.max())
    if h_max - h_min > 1e-7:
        return ((heatmap - h_min) / (h_max - h_min)).astype(np.float32)
    return np.zeros_like(heatmap, dtype=np.float32)

def smooth_heatmap(heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return heatmap
    ksize = int(2 * round(sigma * 3) + 1)
    if ksize < 3:
        return heatmap
    return cv2.GaussianBlur(heatmap, (ksize, ksize), sigma)

def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    heatmap_uint8 = (normalize_heatmap(heatmap) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)

def compute_attention_metrics(heatmap: np.ndarray) -> AttentionMetrics:
    h, w = heatmap.shape
    total_energy = float(heatmap.sum())
    if total_energy < 1e-7:
        return AttentionMetrics(
            center_of_mass_x=w / 2.0,
            center_of_mass_y=h / 2.0,
            com_offset_from_center=0.0,
            corner_activation_fraction=0.0,
            edge_activation_fraction=0.0,
            activation_spread=0.0,
            activation_entropy=0.0,
            max_activation_value=0.0,
            mean_activation_value=0.0,
        )
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    com_x = float((xs * heatmap).sum() / total_energy)
    com_y = float((ys * heatmap).sum() / total_energy)
    center_x, center_y = w / 2.0, h / 2.0
    com_offset = float(np.sqrt((com_x - center_x) ** 2 + (com_y - center_y) ** 2) / np.sqrt(center_x**2 + center_y**2))
    border_frac = 0.1
    bh = int(h * border_frac)
    bw = int(w * border_frac)
    bh = max(bh, 1)
    bw = max(bw, 1)
    corner_regions = [
        heatmap[:bh, :bw],
        heatmap[:bh, -bw:],
        heatmap[-bh:, :bw],
        heatmap[-bh:, -bw:],
    ]
    corner_energy = sum(float(r.sum()) for r in corner_regions)
    corner_fraction = corner_energy / total_energy
    top_edge = heatmap[:bh, :]
    bottom_edge = heatmap[-bh:, :]
    left_edge = heatmap[:, :bw]
    right_edge = heatmap[:, -bw:]
    edge_mask = np.zeros_like(heatmap, dtype=bool)
    edge_mask[:bh, :] = True
    edge_mask[-bh:, :] = True
    edge_mask[:, :bw] = True
    edge_mask[:, -bw:] = True
    edge_energy = float(heatmap[edge_mask].sum())
    edge_fraction = edge_energy / total_energy
    norm_h = heatmap / total_energy
    spread = float(np.std(heatmap))
    eps = 1e-10
    entropy = -float((norm_h[norm_h > 0] * np.log(norm_h[norm_h > 0] + eps)).sum())
    reasons: list[str] = []
    is_susp = False
    if corner_fraction > 0.25:
        reasons.append(f"High corner attention ({corner_fraction:.1%})")
        is_susp = True
    if edge_fraction > 0.50:
        reasons.append(f"High edge attention ({edge_fraction:.1%})")
        is_susp = True
    if com_offset > 0.40:
        reasons.append(f"Attention off-center (offset={com_offset:.2f})")
        is_susp = True
    return AttentionMetrics(
        center_of_mass_x=com_x,
        center_of_mass_y=com_y,
        com_offset_from_center=com_offset,
        corner_activation_fraction=corner_fraction,
        edge_activation_fraction=edge_fraction,
        activation_spread=spread,
        activation_entropy=entropy,
        max_activation_value=float(heatmap.max()),
        mean_activation_value=float(heatmap.mean()),
        is_suspicious=is_susp,
        suspicious_reasons=reasons,
    )

def create_side_by_side_figure(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay_bgr: np.ndarray,
    title: str = "",
    config: GradCAMConfig = GradCAMConfig(),
) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=config.figure_size, dpi=config.figure_dpi)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")
    heatmap_display = normalize_heatmap(heatmap)
    axes[1].imshow(heatmap_display, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("GradCAM Heatmap", fontsize=10)
    axes[1].axis("off")
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig

def save_gradcam_visualization(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    true_label: str,
    predicted_label: str,
    confidence: float,
    category: str,
    config: GradCAMConfig = GradCAMConfig(),
) -> None:
    if config.smooth_sigma > 0:
        heatmap = smooth_heatmap(heatmap, sigma=config.smooth_sigma)

    image_resized = image_pil.resize((224, 224), Image.LANCZOS)
    image_rgb = np.array(image_resized)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(image_bgr, heatmap, alpha=config.alpha, colormap=config.colormap)
    color = "green" if predicted_label == true_label else "red"
    title = (
        f"True: {true_label} | Pred: {predicted_label} | "
        f"Bad-conf: {confidence:.4f} | {category}"
    )
    fig = create_side_by_side_figure(
        image_rgb=image_rgb,
        heatmap=heatmap,
        overlay_bgr=overlay,
        title=title,
        config=config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0.1, dpi=config.figure_dpi)
    plt.close(fig)

def categorize_prediction(
    true_label: str,
    predicted_label: str,
    confidence: float,
    threshold: float = 0.5,
    low_conf_margin: float = 0.20,
) -> str:
    if true_label != predicted_label:
        if predicted_label == "bad":
            return "false_positive"
        return "false_negative"
    if abs(confidence - threshold) < low_conf_margin:
        return "low_confidence"
    if confidence > (1.0 - low_conf_margin) or confidence < low_conf_margin:
        return "high_confidence"
    if predicted_label == "good":
        return "good_correct"
    return "bad_correct"

def build_filename(
    true_label: str,
    predicted_label: str,
    confidence: float,
    index: int = 0,
) -> str:
    conf_str = f"{confidence:.4f}"
    safe_conf = conf_str.replace(".", "_")
    return f"{true_label}_to_{predicted_label}_c{safe_conf}_idx{index:04d}.png"

def generate_gradcam_for_dataset(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    output_root: Path,
    device: torch.device,
    config: GradCAMConfig = GradCAMConfig(),
    max_samples: int | None = None,
) -> list[GradCAMResult]:
    output_root.mkdir(parents=True, exist_ok=True)
    category_dirs = {
        "good_correct": output_root / "good_correct",
        "bad_correct": output_root / "bad_correct",
        "false_positive": output_root / "false_positive",
        "false_negative": output_root / "false_negative",
        "low_confidence": output_root / "low_confidence",
        "high_confidence": output_root / "high_confidence",
    }
    for d in category_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    suspicious_dir = output_root / "suspicious_attention"
    suspicious_dir.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model, target_layer=model.gradcam_target_layer)
    guided_bp = GuidedBackprop(model) if config.smooth_sigma > 0 else None
    if guided_bp is not None:
        guided_bp.enable()
    all_results: list[GradCAMResult] = []
    category_counts: dict[str, int] = {k: 0 for k in category_dirs}
    suspicious_examples: list[GradCAMResult] = []
    total_batches = len(loader)
    processed = 0
    try:
        for inputs, targets, metadata in tqdm(loader, desc="GradCAM", total=total_batches):
            if max_samples is not None and processed >= max_samples:
                break
            inputs = inputs.to(device, non_blocking=True)
            batch_size_actual = inputs.size(0)
            with torch.no_grad():
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                predicted_classes = logits.argmax(dim=1)
            bad_confidences = probs[:, 1].cpu().numpy()
            for i in range(batch_size_actual):
                if max_samples is not None and processed >= max_samples:
                    break
                true_label = str(metadata["label"][i]) if isinstance(metadata, dict) else metadata[i]["label"]
                pred_idx = int(predicted_classes[i].item())
                predicted_label = CLASS_NAMES_TUPLE[pred_idx]
                confidence = float(bad_confidences[i])
                image_path = str(metadata["image_path"][i]) if isinstance(metadata, dict) else metadata[i]["image_path"]
                category = categorize_prediction(true_label, predicted_label, confidence)
                single_input = inputs[i : i + 1]
                try:
                    heatmap = gradcam.generate(single_input, target_class=pred_idx)
                except RuntimeError:
                    continue
                attention_metrics = compute_attention_metrics(heatmap)
                result = GradCAMResult(
                    image_path=image_path,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    bad_class_confidence=confidence,
                    category=category,
                    heatmap_normalized=heatmap,
                    attention_metrics=attention_metrics,
                )
                all_results.append(result)
                cat_dir = category_dirs.get(category)
                if cat_dir is not None and category_counts[category] < config.max_per_category:
                    filename = build_filename(true_label, predicted_label, confidence, processed)
                    img_pil = Image.open(image_path).convert("RGB")
                    save_gradcam_visualization(
                        image_pil=img_pil,
                        heatmap=heatmap,
                        output_path=cat_dir / filename,
                        true_label=true_label,
                        predicted_label=predicted_label,
                        confidence=confidence,
                        category=category,
                        config=config,
                    )
                    category_counts[category] += 1
                if attention_metrics.is_suspicious:
                    suspicious_examples.append(result)
                    susp_filename = f"suspicious_{build_filename(true_label, predicted_label, confidence, len(suspicious_examples))}"
                    img_pil = Image.open(image_path).convert("RGB")
                    save_gradcam_visualization(
                        image_pil=img_pil,
                        heatmap=heatmap,
                        output_path=suspicious_dir / susp_filename,
                        true_label=true_label,
                        predicted_label=predicted_label,
                        confidence=confidence,
                        category=f"suspicious_{category}",
                        config=config,
                    )
                    json_path = suspicious_dir / susp_filename.replace(".png", ".json")
                    import json
                    json_path.write_text(
                        json.dumps(
                            {
                                "image_path": image_path,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "bad_class_confidence": confidence,
                                "category": category,
                                "suspicious_reasons": attention_metrics.suspicious_reasons,
                                "attention_metrics": {
                                    "com_offset_from_center": attention_metrics.com_offset_from_center,
                                    "corner_activation_fraction": attention_metrics.corner_activation_fraction,
                                    "edge_activation_fraction": attention_metrics.edge_activation_fraction,
                                    "activation_spread": float(attention_metrics.activation_spread),
                                },
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                processed += 1
    finally:
        gradcam.remove_hooks()
        if guided_bp is not None:
            guided_bp.disable()
    return all_results