from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


class BaseAnomalyDetector(ABC):
    """Abstract interface for all anomaly detection models in SurfaceGuard AI."""

    def __init__(self, category: str, device: Optional[torch.device] = None):
        self.category = category
        self.device = device or torch.device("cpu")
        self.calibrated_threshold: float = 0.5
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """Fit the model or build reference memory bank on normal samples."""
        pass

    @abstractmethod
    def predict_sample(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Inference for a single normalized image tensor of shape (3, H, W).
        
        Returns:
            anomaly_score: float (image-level score)
            anomaly_map: np.ndarray of shape (H, W) with pixel anomaly values normalized [0, 1]
        """
        pass

    def calibrate_threshold(self, normal_scores: list, target_fpr: float = 0.05) -> float:
        """Calibrate decision threshold on held-out normal samples.
        
        Threshold is set at (1.0 - target_fpr) percentile of normal scores,
        ensuring controlled false-positive rate on normal inspection samples.
        """
        if not normal_scores:
            self.calibrated_threshold = 0.5
            return self.calibrated_threshold

        scores_arr = np.array(normal_scores)
        percentile = 100.0 * (1.0 - target_fpr)
        self.calibrated_threshold = float(np.percentile(scores_arr, percentile))
        return self.calibrated_threshold
