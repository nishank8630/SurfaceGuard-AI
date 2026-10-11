# SurfaceGuard AI — Evaluation Methodology & Metrics

## Unsupervised Evaluation Protocol

In accordance with standard industrial inspection benchmarks:
1. **Training Phase**:
   - The model learns only from defect-free normal samples (`train/good`).
   - Defect samples are never seen during training.
2. **Threshold Calibration**:
   - Calibrated exclusively on a held-out split of normal training samples (`val` split).
   - Set to the $(1 - \text{FPR})$ percentile of normal anomaly scores (default: 95th percentile, yielding a target FPR of 5% on defect-free production units).
3. **Test Evaluation**:
   - Evaluated on normal test items (`test/good`, label=0) and all defect classes (`test/<defect_type>`, label=1).

## Metric Definitions

### Image-Level Metrics
- **AUROC (Area Under ROC Curve)**: Ranking capability across all classification thresholds, invariant to threshold selection.
- **Average Precision (PR-AUC)**: Evaluates defect recall vs. precision, especially critical under class imbalance.
- **Precision / Recall / F1-Score**: Evaluated at the calibrated operating threshold.
- **Confusion Matrix**: Quantitative breakdown of True Positives, False Positives, True Negatives, and False Negatives.

### Pixel-Level Localization Metrics
- **Pixel AUROC**: Anomaly localization quality across all pixels. Evaluates whether anomalous pixels receive higher scores than background pixels.
- **Pixel AP**: Precision-Recall AUC computed over all test mask pixels.
- **Pixel IoU (Intersection over Union)**: Jaccard index between predicted binarized mask and ground-truth mask at calibrated pixel threshold:
  $$\text{IoU} = \frac{|P \cap GT|}{|P \cup GT|}$$
- **Dice Coefficient**:
  $$\text{Dice} = \frac{2 |P \cap GT|}{|P| + |GT|}$$

### Inference Latency & Profiling
- **Warm-up**: 3 unmeasured forward passes to warm caches and eliminate JIT overhead.
- **Statistics**: Mean, Median, and 95th Percentile latency in milliseconds.
- **Throughput**: Measured in Frames Per Second (FPS).
