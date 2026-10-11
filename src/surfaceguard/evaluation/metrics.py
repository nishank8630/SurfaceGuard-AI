from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix


def compute_image_metrics(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute image-level evaluation metrics: AUROC, AP, Precision, Recall, F1."""
    y_true = np.asarray(labels, dtype=int)
    y_scores = np.asarray(scores, dtype=float)

    unique_labels = np.unique(y_true)
    has_both_classes = len(unique_labels) > 1

    auroc = float(roc_auc_score(y_true, y_scores)) if has_both_classes else None
    ap = float(average_precision_score(y_true, y_scores)) if has_both_classes else None

    # If threshold not specified, default to 50th percentile or 0.5
    thresh = threshold if threshold is not None else float(np.median(y_scores))
    y_pred = (y_scores >= thresh).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "auroc": auroc,
        "average_precision": ap,
        "threshold": thresh,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def compute_pixel_metrics(
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    threshold: Optional[float] = None,
    subsample_factor: int = 4,  # Subsample pixels for fast, memory-safe pixel AUROC calculation
) -> Dict[str, Any]:
    """Compute pixel-level localization metrics: Pixel AUROC, Pixel AP, Pixel IoU, Dice."""
    flat_masks_list = []
    flat_maps_list = []

    for mask, amap in zip(masks, maps):
        # Subsample spatially for speed and memory efficiency
        m_sub = mask[::subsample_factor, ::subsample_factor].flatten()
        a_sub = amap[::subsample_factor, ::subsample_factor].flatten()
        flat_masks_list.append(m_sub)
        flat_maps_list.append(a_sub)

    y_true = np.concatenate(flat_masks_list).astype(int)
    y_scores = np.concatenate(flat_maps_list).astype(float)

    has_both_classes = len(np.unique(y_true)) > 1
    pixel_auroc = float(roc_auc_score(y_true, y_scores)) if has_both_classes else None
    pixel_ap = float(average_precision_score(y_true, y_scores)) if has_both_classes else None

    # Threshold for pixel-level IoU and Dice
    thresh = threshold if threshold is not None else float(np.mean(y_scores) + np.std(y_scores))
    y_pred = (y_scores >= thresh).astype(int)

    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    iou = float(intersection / union) if union > 0 else (1.0 if np.sum(y_true) == 0 else 0.0)
    dice = float((2 * intersection) / (np.sum(y_true) + np.sum(y_pred))) if (np.sum(y_true) + np.sum(y_pred)) > 0 else 1.0

    return {
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "pixel_threshold": thresh,
        "pixel_iou": iou,
        "pixel_dice": dice,
    }
