import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from surfaceguard.config import ALL_CATEGORIES
from surfaceguard.data.transforms import get_transforms
from surfaceguard.models.base import BaseAnomalyDetector
from surfaceguard.models.classical_baseline import ClassicalBaseline
from surfaceguard.models.autoencoder import AutoencoderDetector
from surfaceguard.models.anomaly_detector import PatchCoreDetector
from surfaceguard.inference.heatmap import generate_heatmap_overlay, array_to_base64_jpeg
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SurfaceGuardPredictor:
    """Unified inference engine for SurfaceGuard AI serving FastAPI and Streamlit."""

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        device_preference: str = "auto",
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.device = get_device(device_preference)
        self.artifacts = ArtifactManager(artifacts_dir)
        self.image_size = image_size
        self.transform = get_transforms(image_size=image_size)

        # In-memory model cache: {(category, model_type): detector_instance}
        self._model_cache: Dict[Tuple[str, str], BaseAnomalyDetector] = {}

    def get_detector(self, category: str, model_type: str = "patchcore") -> BaseAnomalyDetector:
        """Loads and caches the requested anomaly detector."""
        cat = category.lower().strip()
        m_type = model_type.lower().strip()

        if cat not in ALL_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Must be one of: {ALL_CATEGORIES}")

        cache_key = (cat, m_type)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        logger.info(f"Loading {m_type} model for category '{cat}' onto {self.device}...")

        if m_type == "patchcore":
            detector = PatchCoreDetector(category=cat, device=self.device)
            # Load memory bank
            bank_path = self.artifacts.get_feature_bank_path(cat)
            if not bank_path.exists():
                raise FileNotFoundError(
                    f"No feature memory bank found for category '{cat}' at {bank_path}. "
                    f"Run 'python scripts/build_feature_bank.py --category {cat}' to create it."
                )
            detector.load_memory_bank(bank_path)

        elif m_type == "autoencoder":
            detector = AutoencoderDetector(category=cat, device=self.device)
            ckpt_path = self.artifacts.get_checkpoint_path(cat, "autoencoder")
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"No autoencoder checkpoint found for category '{cat}' at {ckpt_path}. "
                    f"Run 'python scripts/train_autoencoder.py --category {cat}' to train it."
                )
            detector.load(ckpt_path)

        elif m_type in ["classical", "baseline"]:
            detector = ClassicalBaseline(category=cat, device=self.device)
            # Classical baseline will initialize zero/diff template if no precomputed template
        else:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose 'patchcore', 'autoencoder', or 'classical'.")

        # Load calibrated threshold if saved
        loaded_thresh = self.artifacts.load_threshold(cat, m_type)
        if loaded_thresh is not None:
            detector.calibrated_threshold = loaded_thresh
        else:
            detector.calibrated_threshold = 0.5  # Sensible default

        self._model_cache[cache_key] = detector
        return detector

    def predict_image(
        self,
        image: Image.Image,
        category: str,
        model_type: str = "patchcore",
        auto_segment: bool = False,
    ) -> Dict[str, Any]:
        """Runs end-to-end inference on a single PIL Image with optional auto-segmentation."""
        start_time = time.perf_counter()

        orig_w, orig_h = image.size
        detector = self.get_detector(category, model_type)

        # Preprocess: If real photo, isolate foreground to avoid background clutter penalties
        img_rgb = image.convert("RGB")
        if auto_segment:
            try:
                from surfaceguard.data.auto_segment import segment_and_normalize_object
                proc_img, _ = segment_and_normalize_object(img_rgb, target_size=self.image_size)
                img_tensor = self.transform(proc_img)
            except Exception:
                img_tensor = self.transform(img_rgb)
        else:
            img_tensor = self.transform(img_rgb)  # (3, H, W)

        # Model inference
        score, amap = detector.predict_sample(img_tensor)

        # Threshold decision
        threshold = detector.calibrated_threshold
        is_anomalous = bool(score >= threshold)

        # Generate overlays on original image
        heatmap_rgb, overlay_rgb = generate_heatmap_overlay(img_rgb, amap)

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "category": category,
            "model_type": model_type,
            "is_anomalous": is_anomalous,
            "anomaly_score": round(float(score), 4),
            "threshold": round(float(threshold), 4),
            "inference_time_ms": round(inference_time_ms, 2),
            "heatmap_base64": array_to_base64_jpeg(heatmap_rgb),
            "overlay_base64": array_to_base64_jpeg(overlay_rgb),
            "original_size": [orig_w, orig_h],
        }
