from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from surfaceguard.models.base import BaseAnomalyDetector
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ClassicalBaseline(BaseAnomalyDetector):
    """Method A: Classical Computer Vision Statistical Difference Baseline.
    
    Builds a normal pixel/patch mean and variance model from defect-free training images.
    Identifies deviations based on Mahalanobis/Z-score distance followed by Gaussian smoothing.
    """

    def __init__(self, category: str, device: Optional[torch.device] = None, blur_sigma: float = 4.0):
        super().__init__(category, device)
        self.blur_sigma = blur_sigma
        self.mean_image: Optional[np.ndarray] = None
        self.std_image: Optional[np.ndarray] = None

    def fit(self, dataloader: DataLoader) -> "ClassicalBaseline":
        """Compute pixel-wise mean and variance from normal training images."""
        images_list: List[np.ndarray] = []
        for batch in dataloader:
            imgs = batch["image"]  # (B, 3, H, W)
            for i in range(imgs.shape[0]):
                img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
                images_list.append(img_np)

        if not images_list:
            raise ValueError(f"No training images provided to fit ClassicalBaseline for {self.category}")

        stacked = np.stack(images_list, axis=0)  # (N, H, W, 3)
        self.mean_image = np.mean(stacked, axis=0)
        self.std_image = np.std(stacked, axis=0) + 1e-5  # Add epsilon for numerical stability
        self.is_fitted = True
        logger.info(f"Classical baseline fitted on {len(images_list)} normal samples for '{self.category}'")
        return self

    def predict_sample(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Compute structural z-score deviation map."""
        if not self.is_fitted or self.mean_image is None or self.std_image is None:
            # Fallback if not fitted: compute internal gradient/edge deviation
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            diff = np.abs(img_np - np.mean(img_np))
        else:
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            diff = np.abs(img_np - self.mean_image) / self.std_image

        # Channel mean
        diff_2d = np.mean(diff, axis=-1)

        # Smooth with Gaussian blur
        ksize = int(6 * self.blur_sigma + 1) | 1
        anomaly_map = cv2.GaussianBlur(diff_2d.astype(np.float32), (ksize, ksize), self.blur_sigma)

        # Normalize map to [0, 1] range locally
        min_val, max_val = anomaly_map.min(), anomaly_map.max()
        if max_val > min_val:
            norm_map = (anomaly_map - min_val) / (max_val - min_val)
        else:
            norm_map = np.zeros_like(anomaly_map)

        # Image-level score: 98th percentile of anomaly map to be robust to single-pixel outliers
        image_score = float(np.percentile(anomaly_map, 98))
        return image_score, norm_map
