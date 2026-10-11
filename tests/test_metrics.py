import numpy as np
from surfaceguard.evaluation.metrics import compute_image_metrics, compute_pixel_metrics


def test_image_metrics_perfect_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    metrics = compute_image_metrics(y_true, y_scores, threshold=0.5)

    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["true_positives"] == 3
    assert metrics["true_negatives"] == 3
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_image_metrics_single_class_edge_case():
    # Only normal samples
    y_true = [0, 0, 0]
    y_scores = [0.1, 0.2, 0.3]
    metrics = compute_image_metrics(y_true, y_scores)
    # When single class, auroc and AP are undefined (None) without crashing
    assert metrics["auroc"] is None
    assert metrics["average_precision"] is None


def test_pixel_metrics():
    # 2 sample images of 16x16
    mask1 = np.zeros((16, 16), dtype=np.uint8)
    mask2 = np.zeros((16, 16), dtype=np.uint8)
    mask2[4:8, 4:8] = 1  # 16 anomalous pixels

    map1 = np.full((16, 16), 0.1, dtype=np.float32)
    map2 = np.full((16, 16), 0.1, dtype=np.float32)
    map2[4:8, 4:8] = 0.9

    metrics = compute_pixel_metrics([mask1, mask2], [map1, map2], subsample_factor=1)
    assert metrics["pixel_auroc"] == 1.0
    assert metrics["pixel_ap"] == 1.0
    assert "pixel_iou" in metrics
    assert "pixel_dice" in metrics
