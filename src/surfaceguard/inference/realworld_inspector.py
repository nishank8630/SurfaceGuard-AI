import time
from typing import Dict, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from surfaceguard.data.transforms import get_transforms
from surfaceguard.inference.heatmap import generate_heatmap_overlay, array_to_base64_jpeg


class RealWorldInspector:
    """Mode 2: Zero-Shot Real-World Object Inspector.
    
    Allows uploading ANY product image (any color, any bottle, any shape).
    Compares the inspection image against a golden clean standard or inspects
    local structural surface consistency without requiring pre-trained dataset categories.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.transform = get_transforms(image_size=(256, 256))

        # Lightweight frozen ResNet feature backbone
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.backbone = backbone.to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._features = []
        self.backbone.layer2.register_forward_hook(lambda m, i, o: self._features.append(o))

    def _extract_feature_map(self, img_pil: Image.Image) -> torch.Tensor:
        self._features.clear()
        t = self.transform(img_pil.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.backbone(t)
        return self._features[-1]  # Shape: (1, 128, 32, 32)

    def inspect_against_reference(
        self,
        reference_image: Image.Image,
        test_image: Image.Image,
        threshold: float = 3.5,
    ) -> Dict:
        """Compares test image against a clean golden standard reference image."""
        start_t = time.perf_counter()

        # Extract multi-scale deep features
        ref_feat = self._extract_feature_map(reference_image)
        test_feat = self._extract_feature_map(test_image)

        # Patch-wise Euclidean feature discrepancy
        diff = torch.norm(ref_feat - test_feat, dim=1).unsqueeze(1)  # (1, 1, 32, 32)

        # Upsample to full image resolution (H, W)
        W, H = test_image.size
        upsampled = F.interpolate(diff, size=(H, W), mode="bilinear", align_corners=False)
        raw_map = upsampled.squeeze().cpu().numpy()

        # Gaussian smoothing
        smoothed_map = cv2.GaussianBlur(raw_map.astype(np.float32), (21, 21), 4.0)

        # Anomaly score: 98th percentile of patch distances
        score = float(np.percentile(smoothed_map, 98))
        is_anomalous = bool(score >= threshold)

        # Normalize map for visualization
        min_v, max_v = smoothed_map.min(), smoothed_map.max()
        norm_map = (smoothed_map - min_v) / (max_v - min_v + 1e-6) if max_v > min_v else np.zeros_like(smoothed_map)

        # Generate overlays
        heatmap_rgb, overlay_rgb = generate_heatmap_overlay(test_image, norm_map)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "mode": "Reference Comparison",
            "is_anomalous": is_anomalous,
            "anomaly_score": round(score, 4),
            "threshold": round(threshold, 4),
            "inference_time_ms": round(elapsed_ms, 1),
            "heatmap_base64": array_to_base64_jpeg(heatmap_rgb),
            "overlay_base64": array_to_base64_jpeg(overlay_rgb),
        }
