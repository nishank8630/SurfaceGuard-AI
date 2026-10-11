# MVTec Anomaly Detection (MVTec AD) Dataset Specification

The MVTec AD dataset is the comprehensive benchmark for industrial visual inspection and unsupervised anomaly detection.

## Categories Overview

MVTec AD comprises **15 categories**:
- **10 Object Categories**: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`
- **5 Texture Categories**: `carpet`, `grid`, `leather`, `tile`, `wood`

Total images: 5,354 high-resolution real-world images.

## Structure & Pairing Convention

```
data/mvtec_ad/
└── <category>/
    ├── train/
    │   └── good/                          <-- Defect-free training samples
    ├── test/
    │   ├── good/                          <-- Defect-free test samples (mask = 0)
    │   ├── <defect_type_1>/               <-- Defective test samples
    │   └── <defect_type_2>/
    └── ground_truth/
        ├── <defect_type_1>/
        │   └── <stem>_mask.png            <-- Pixel-level binary masks
        └── <defect_type_2>/
            └── <stem>_mask.png
```

## Attribution & License

- **Dataset**: MVTec Anomaly Detection Dataset (MVTec AD)
- **Author**: MVTec Software GmbH
- **Citation**: P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Usage Notice**: Strictly non-commercial research and educational portfolio use.
