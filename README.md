# SurfaceGuard AI
### Deep Learning-Based Industrial Defect Detection & Anomaly Localization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview & Motivation

**SurfaceGuard AI** is an industrial visual quality inspection system engineered to detect and localize structural and cosmetic surface anomalies across all 15 categories of the benchmark **MVTec Anomaly Detection (MVTec AD)** dataset.

In modern manufacturing (semiconductors, pharmaceuticals, precision machining, textiles), collecting tens of thousands of defective samples is prohibitively expensive because defect occurrences are rare and unpredictable. Consequently, supervised binary classifiers fail due to severe class imbalance.

SurfaceGuard AI implements an **unsupervised, one-class anomaly detection** paradigm:
- **Learns exclusively from defect-free normal samples**.
- Detects surface deviations without requiring prior knowledge of specific defect shapes.
- Outputs **image-level anomaly scores**, calibrated **Pass/Fail decisions**, and high-resolution **pixel-level anomaly heatmaps**.

---

## 2. System Architecture

```
                                [ INSPECTION IMAGE ]
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │  Preprocessing & Normalization │ (256x256, ImageNet Tensor)
                         └───────────────┬───────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  ▼                      ▼                      ▼
        [ Method A: Classical ] [ Method B: Autoencoder ] [ Method C: PatchCore CNN ]
        - Structural Z-Score    - 8-layer Conv Autoenc  - Pretrained ResNet-18/50
        - Pixel Mean/Std Diff   - Reconstruction Error  - Layer2 & Layer3 Embeddings
        - Fast Zero-Shot        - Gaussian Residual Map - Coreset Memory Bank (5-10%)
                  │                      │              - Chunked L2 Nearest-Neighbor
                  └──────────────────────┼──────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │ Anomaly Map & Score Generator │
                         │ - Bilinear Upsampling (H, W)  │
                         │ - Gaussian Kernel Smoothing   │
                         │ - Score: Top Patch Distance   │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │ Calibrated Decision Engine    │
                         │ - Normal Val Split Threshold  │
                         │ - Target 5% FPR Calibration   │
                         │ - JET Heatmap & Alpha Overlay │
                         └───────────────┬───────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                [ FastAPI REST Backend ]      [ Streamlit Web Dashboard ]
                - GET /health                 - Interactive drag-and-drop
                - GET /categories             - Heatmap overlay viewer
                - POST /predict               - Benchmark & artifact browser
```

---

## 3. Supported Categories (MVTec AD)

SurfaceGuard AI dynamically discovers and supports **all 15 MVTec AD categories**:

| Category Type | Categories |
| :--- | :--- |
| **Objects (10)** | `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper` |
| **Textures (5)** | `carpet`, `grid`, `leather`, `tile`, `wood` |

---

## 4. Implemented Anomaly Detection Methods

### Method A: Classical Computer Vision Baseline
- Builds a normal statistical template (mean $\mu(x,y)$ and variance $\sigma(x,y)$) from defect-free training images.
- Evaluates test images via structural $Z$-score deviation with Gaussian filtering.
- Requires no precomputed weights or neural backbones; useful as a fast, zero-shot comparative baseline.

### Method B: Convolutional Autoencoder (PyTorch)
- Compact encoder-decoder architecture: 4 downsampling convolutional blocks, latent representation bottleneck ($d=128$), and 4 transposed convolutional blocks.
- Trained strictly on normal images using MSE reconstruction loss.
- Anomaly map derived from pixel-level residual error: $M(x,y) = \|I(x,y) - \hat{I}(x,y)\|_1$ smoothed with a Gaussian kernel.

### Method C: Pretrained CNN Feature-Bank Detector (PatchCore-style)
- **Feature Backbone**: Frozen ImageNet-pretrained ResNet-18 or ResNet-50.
- **Multi-Scale Patches**: Extracts intermediate feature maps from `layer2` and `layer3`, aligning spatial resolutions via adaptive pooling.
- **Memory Bank Coreset**: Subsamples reference normal patch embeddings using greedy $k$-center coreset reduction (configurable $5\% - 20\%$) to keep RAM and inference latency low.
- **Chunked Nearest Neighbor**: Computes minimum Euclidean distances in safe 1024-patch chunks via norm expansion:
  $$\|q - m\|^2 = \|q\|^2 - 2qm^T + \|m\|^2$$
  guaranteeing zero out-of-memory errors on both CPU and GPU.
- **Spatial Heatmap**: Bilinearly interpolated back to image resolution $(H, W)$ and smoothed with Gaussian blur ($\sigma=4$).

---

## 5. Evaluation Methodology & Leakage Prevention

1. **No Data Leakage**: Training images are defect-free (`train/good`). Official test defect images are never seen during training or parameter tuning.
2. **Threshold Calibration**: Decision thresholds are calibrated on held-out normal validation images (`val` split from `train/good`) targeting a specified false positive rate (e.g. 5% FPR).
3. **Dual-Level Metrics**:
   - **Image-Level**: AUROC, Average Precision (AP), Precision, Recall, F1-score, Confusion Matrix.
   - **Pixel-Level**: Pixel AUROC, Pixel AP, Pixel IoU, Dice Score.
   - **Latency Profiling**: Mean, Median, 95th percentile latency (ms), and Frames Per Second (FPS).

---

## 6. Local Setup Instructions (Windows PowerShell)

### Step 1: Clone and Set Up Environment
```powershell
# Navigate to project directory
cd SurfaceGuard-AI

# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install package and dependencies
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### Step 2: Dataset Setup
The official MVTec AD dataset is stored under `data/mvtec_ad/`.
If you have downloaded `mvtec_anomaly_detection.tar.xz`, extract it:
```powershell
python scripts/validate_dataset.py --extract "C:\path\to\mvtec_anomaly_detection.tar.xz"
```
Or validate existing folders:
```powershell
python scripts/validate_dataset.py
python scripts/inspect_dataset.py
```

### Step 3: Run Automated Test Suite
Run unit tests with synthetic data fallback (no real dataset download required for unit testing):
```powershell
pytest tests/ -v
```

---

## 7. Command-Line Workflows

### A. Build Feature Memory Bank (PatchCore)
```powershell
# Build for a single category
python scripts/build_feature_bank.py --category bottle --sampling-ratio 0.05

# Build for all available categories
python scripts/build_feature_bank.py --category all --sampling-ratio 0.05
```

### B. Train Convolutional Autoencoder
```powershell
python scripts/train_autoencoder.py --category bottle --epochs 25 --batch-size 16
```

### C. Evaluate Model on Category
```powershell
# Evaluate PatchCore
python scripts/evaluate.py --category bottle --model-type patchcore

# Evaluate Autoencoder
python scripts/evaluate.py --category bottle --model-type autoencoder

# Evaluate Classical Baseline
python scripts/evaluate.py --category bottle --model-type classical
```

### D. Run Multi-Category Benchmark
```powershell
python scripts/run_benchmark.py --categories all --model-type patchcore
```

### E. Start FastAPI Backend
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger docs available at: `http://127.0.0.1:8000/docs`

### F. Start Streamlit Web Dashboard
```powershell
streamlit run app/dashboard.py
```
Opens interactive UI at: `http://localhost:8501`

---

## 8. Google Colab GPU Workflow

For full 15-category training and benchmarking on NVIDIA GPUs:
1. Open [`notebooks/SurfaceGuard_AI_Colab.ipynb`](notebooks/SurfaceGuard_AI_Colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Follow the guided cells to mount Google Drive, extract `mvtec_ad`, and execute multi-category experiments.
4. Exported memory banks and results are saved directly to Google Drive.

---

## 9. Measured Results vs. Pending Experiments

All experimental numbers are dynamically generated from real execution runs.

| Category | Model | Image AUROC | Image AP | Pixel AUROC | Mean Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | PatchCore (ResNet-18) | *Measured dynamically in results/* | *Measured* | *Measured* | *~150 ms (Intel i5 CPU)* |
| Other 14 Categories | PatchCore / Autoencoder | *Pending execution* | *Pending* | *Pending* | *Pending* |

> *Note*: To refresh measured tables, run `python scripts/run_benchmark.py` followed by `python scripts/generate_report.py`.

---

## 10. Dataset Attribution & License

- **Dataset**: MVTec Anomaly Detection Dataset (MVTec AD)
- **Creator**: MVTec Software GmbH (Munich, Germany)
- **Citation**: P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **SurfaceGuard AI Code**: Released under the MIT License.
