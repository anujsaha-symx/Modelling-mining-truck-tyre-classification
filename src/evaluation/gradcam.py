from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_maps: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._target_layer = self._find_last_conv(model)
        self._forward_handle = self._target_layer.register_forward_hook(self._save_features)
        self._backward_handle = self._target_layer.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Conv2d:
        last_conv: nn.Conv2d | None = None
        for module in model.features.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.features")
        return last_conv

    def _save_features(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.feature_maps = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        logit = output[:, class_idx]
        logit.backward(retain_graph=False)

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Gradients or feature maps not captured")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def cleanup(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


class AttentionMetrics:
    """Computes spatial attention metrics from GradCAM heatmaps."""

    @staticmethod
    def compute_all(heatmap: np.ndarray, image_size: int = 224) -> dict:
        edge_frac = AttentionMetrics.edge_attention_fraction(heatmap, image_size)
        corner_frac = AttentionMetrics.corner_attention_fraction(heatmap, image_size)
        com_off = AttentionMetrics.center_of_mass_offset(heatmap, image_size)
        ent = AttentionMetrics.entropy(heatmap)

        flags: list[str] = []
        if edge_frac > 0.40:
            flags.append("edge-focused")
        if corner_frac > 0.25:
            flags.append("corner-focused")
        if com_off > 0.50 and edge_frac > 0.30:
            flags.append("background-focused")

        return {
            "edge_attention_fraction": round(float(edge_frac), 6),
            "corner_attention_fraction": round(float(corner_frac), 6),
            "center_of_mass_offset": round(float(com_off), 6),
            "entropy": round(float(ent), 6),
            "suspicious_flags": flags,
        }

    @staticmethod
    def edge_attention_fraction(heatmap: np.ndarray, image_size: int = 224) -> float:
        h, w = heatmap.shape[:2]
        border = max(1, int(0.08 * image_size))
        scale = image_size / max(h, w)
        border_scaled = max(1, int(border / scale))

        mask = np.ones((h, w), dtype=bool)
        mask[border_scaled:-border_scaled, border_scaled:-border_scaled] = False

        total = heatmap.sum()
        if total < 1e-10:
            return 0.0
        return float(heatmap[mask].sum() / total)

    @staticmethod
    def corner_attention_fraction(heatmap: np.ndarray, image_size: int = 224) -> float:
        h, w = heatmap.shape[:2]
        corner_size = max(1, int(0.15 * image_size))
        scale = image_size / max(h, w)
        cs = max(1, int(corner_size / scale))

        mask = np.zeros((h, w), dtype=bool)
        mask[:cs, :cs] = True
        mask[:cs, -cs:] = True
        mask[-cs:, :cs] = True
        mask[-cs:, -cs:] = True

        total = heatmap.sum()
        if total < 1e-10:
            return 0.0
        return float(heatmap[mask].sum() / total)

    @staticmethod
    def center_of_mass_offset(heatmap: np.ndarray, image_size: int = 224) -> float:
        h, w = heatmap.shape[:2]
        total = heatmap.sum()
        if total < 1e-10:
            return 1.0

        ys, xs = np.mgrid[0:h, 0:w]
        cy = float((ys * heatmap).sum() / total)
        cx = float((xs * heatmap).sum() / total)

        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0
        max_dist = np.sqrt(center_y**2 + center_x**2)

        if max_dist < 1e-10:
            return 0.0
        return float(np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2) / max_dist)

    @staticmethod
    def entropy(heatmap: np.ndarray) -> float:
        flat = heatmap.flatten()
        total = flat.sum()
        if total < 1e-10:
            return 0.0
        p = flat / total
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return float(-(p * np.log2(p)).sum())
