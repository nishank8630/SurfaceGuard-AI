# SurfaceGuard AI — System Architecture

SurfaceGuard AI is engineered for industrial visual inspection using an **unsupervised, one-class anomaly detection and pixel localization paradigm**.

```
                           [ INDUSTRIAL INPUT IMAGE ]
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Preprocessing & Normalization │ (256x256, ImageNet Mean/Std)
                       └───────────────┬───────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
      [ Classical Baseline ]  [ Conv-Autoencoder ]   [ PatchCore Feature Bank ]
      - Pixel Mean/Std Diff   - Reconstructs sample  - Pretrained ResNet-18
      - Structural Z-Score    - Residual L1/MSE diff - Multi-scale layers 2 & 3
                │                      │             - Memory Bank Coreset (10%)
                │                      │             - Chunked L2 Nearest-Neighbor
                └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Anomaly Map & Score Generator │
                       │ - Bilinear Upsampling (H, W)  │
                       │ - Gaussian Kernel Smoothing   │
                       │ - Image Score: Top Percentile │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Decision & Visualization      │
                       │ - Calibrated Val Threshold    │
                       │ - Binary Pass/Fail Decision   │
                       │ - Jet Heatmap Blended Overlay │
                       └───────────────┬───────────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
              [ FastAPI Backend ]           [ Streamlit Dashboard ]
              - REST API /predict           - Interactive UI
              - Base64 Overlays             - Real-time Inspection
              - Sub-second CPU latency      - Benchmark Browser
```

## Architectural Design Highlights

1. **Strict Data Leakage Prevention**:
   - The system is trained strictly on defect-free samples (`train/good`).
   - Normal validation samples (`val` split sampled exclusively from `train/good`) calibrate operating thresholds targeting a specified false positive rate (e.g. 5% FPR on normal production samples).
   - Test defect samples and ground truth masks are strictly used during final evaluation.

2. **Chunked Nearest Neighbor Search**:
   - Rather than storing and querying large pairwise distance matrices in RAM, distances are computed in small chunks (1024 query vectors) using norm decomposition: $\|q - m\|^2 = \|q\|^2 - 2qm^T + \|m\|^2$.
   - Prevents memory exhaustion on 16GB CPU systems and resource-constrained environments.

3. **Multi-Scale Spatial Embedding**:
   - Leverages intermediate feature hierarchies from `layer2` and `layer3` of ResNet backbones.
   - Low-level textures and mid-level structural features are combined into local neighborhood representations.
