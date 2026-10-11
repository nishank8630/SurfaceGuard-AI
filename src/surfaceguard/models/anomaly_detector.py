from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from surfaceguard.models.base import BaseAnomalyDetector
from surfaceguard.models.feature_extractor import ResNetFeatureExtractor
from surfaceguard.models.feature_bank import subsample_memory_bank, compute_min_distances_chunked
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PatchCoreDetector(BaseAnomalyDetector):
    """Method C: Pretrained CNN Feature Memory-Bank Anomaly Detector (PatchCore-style).
    
    1. Extracts multi-scale spatial patches from frozen pretrained ResNet.
    2. Constructs reference memory bank from normal training samples.
    3. Reduces memory bank size via coreset sampling.
    4. Computes patch anomaly distances via chunked nearest-neighbor retrieval.
    5. Interpolates and smooths spatial anomaly maps and derives image-level score.
    """

    def __init__(
        self,
        category: str,
        backbone_name: str = "resnet18",
        layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        sampling_ratio: float = 0.10,
        sampling_method: str = "coreset",
        blur_sigma: float = 4.0,
    ):
        super().__init__(category, device)
        self.backbone_name = backbone_name
        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        self.blur_sigma = blur_sigma

        self.feature_extractor = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            layers=layers or ["layer2", "layer3"],
            device=self.device,
        )
        self.memory_bank: Optional[torch.Tensor] = None

    def fit(self, train_loader: DataLoader) -> torch.Tensor:
        """Extract features from all normal training images and build memory bank."""
        logger.info(f"Building feature memory bank for '{self.category}' using {self.backbone_name}...")
        all_patches: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in train_loader:
                imgs = batch["image"].to(self.device)
                patches, _ = self.feature_extractor(imgs)
                all_patches.append(patches.cpu())

        stacked_patches = torch.cat(all_patches, dim=0)  # (Total_patches, D)
        logger.info(f"Extracted {stacked_patches.shape[0]} total patches (dim={stacked_patches.shape[1]})")

        # Subsample memory bank
        self.memory_bank = subsample_memory_bank(
            stacked_patches,
            sampling_ratio=self.sampling_ratio,
            method=self.sampling_method,
        ).to(self.device)

        self.is_fitted = True
        return self.memory_bank

    def predict_sample(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Compute pixel-level anomaly map and image-level anomaly score."""
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() or load_memory_bank() first.")

        H, W = image_tensor.shape[1], image_tensor.shape[2]
        img_batch = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            patches, (H_p, W_p) = self.feature_extractor(img_batch)
            min_dists = compute_min_distances_chunked(patches, self.memory_bank)

            # Spatial distance grid: (1, 1, H_p, W_p)
            score_grid = min_dists.reshape(1, 1, H_p, W_p)

            # Upsample bilinearly to original image resolution (H, W)
            upsampled = F.interpolate(score_grid, size=(H, W), mode="bilinear", align_corners=False)
            anomaly_map_raw = upsampled.squeeze().cpu().numpy()

        # Gaussian smoothing
        ksize = int(6 * self.blur_sigma + 1) | 1
        smoothed_map = cv2.GaussianBlur(anomaly_map_raw.astype(np.float32), (ksize, ksize), self.blur_sigma)

        # Image-level score: top patch distance
        image_score = float(np.max(min_dists.cpu().numpy()))

        # Normalize map locally for visualization [0, 1]
        min_v, max_v = smoothed_map.min(), smoothed_map.max()
        norm_map = (smoothed_map - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(smoothed_map)

        return image_score, norm_map

    def save_memory_bank(self, path: Path) -> None:
        if self.memory_bank is None:
            raise ValueError("No memory bank to save.")
        torch.save(self.memory_bank.cpu(), path)

    def load_memory_bank(self, path: Path) -> None:
        self.memory_bank = torch.load(path, map_location=self.device, weights_only=False).to(self.device)
        self.is_fitted = True
