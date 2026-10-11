import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from surfaceguard.models.base import BaseAnomalyDetector
from surfaceguard.evaluation.metrics import compute_image_metrics, compute_pixel_metrics
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluates an anomaly detector on a dataset split, measuring metrics and latency."""

    def __init__(self, detector: BaseAnomalyDetector):
        self.detector = detector

    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: Optional[float] = None,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Run evaluation with latency measurement and metric extraction."""
        logger.info(f"Starting evaluation on '{self.detector.category}' ({len(test_loader.dataset)} samples)...")

        labels: List[int] = []
        image_scores: List[float] = []
        masks_list: List[np.ndarray] = []
        maps_list: List[np.ndarray] = []
        latencies_ms: List[float] = []

        is_warmed_up = False
        warmup_counter = 0

        for batch_idx, batch in enumerate(test_loader):
            imgs = batch["image"]      # (B, 3, H, W)
            msks = batch["mask"]       # (B, 1, H, W)
            lbls = batch["label"]      # (B,)

            for i in range(imgs.shape[0]):
                img_single = imgs[i]
                mask_single = msks[i].squeeze(0).cpu().numpy()
                lbl_single = int(lbls[i].item())

                # Warmup
                if not is_warmed_up:
                    self.detector.predict_sample(img_single)
                    warmup_counter += 1
                    if warmup_counter >= warmup_runs:
                        is_warmed_up = True
                    continue

                # Timed inference
                start_time = time.perf_counter()
                score, amap = self.detector.predict_sample(img_single)
                duration_ms = (time.perf_counter() - start_time) * 1000.0

                latencies_ms.append(duration_ms)
                image_scores.append(score)
                maps_list.append(amap)
                masks_list.append(mask_single)
                labels.append(lbl_single)

        if not labels:
            logger.warning("No samples evaluated after warmup.")
            return {"error": "No samples evaluated"}

        # Use passed threshold or calibrated threshold on detector
        effective_thresh = threshold if threshold is not None else self.detector.calibrated_threshold

        # Compute metrics
        img_metrics = compute_image_metrics(labels, image_scores, threshold=effective_thresh)
        pixel_metrics = compute_pixel_metrics(masks_list, maps_list)

        # Latency statistics
        latency_arr = np.array(latencies_ms)
        latency_summary = {
            "mean_ms": float(np.mean(latency_arr)),
            "median_ms": float(np.median(latency_arr)),
            "p95_ms": float(np.percentile(latency_arr, 95)),
            "throughput_fps": float(1000.0 / np.mean(latency_arr)) if len(latency_arr) > 0 else 0.0,
            "sample_count": len(labels),
        }

        results = {
            "category": self.detector.category,
            "model_type": self.detector.__class__.__name__,
            "latency": latency_summary,
            "image_metrics": img_metrics,
            "pixel_metrics": pixel_metrics,
        }

        logger.info(
            f"Results for '{self.detector.category}' | "
            f"Image AUROC: {img_metrics['auroc']} | "
            f"Pixel AUROC: {pixel_metrics['pixel_auroc']} | "
            f"Mean Latency: {latency_summary['mean_ms']:.1f}ms"
        )
        return results
