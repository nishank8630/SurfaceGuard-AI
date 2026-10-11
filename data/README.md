# MVTec AD Dataset Storage

Place the extracted MVTec Anomaly Detection dataset folders here.
Expected directory hierarchy:

```
data/mvtec_ad/
├── bottle/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/broken_large/*.png
│   └── ground_truth/broken_large/*_mask.png
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

To extract the official archive (`mvtec_anomaly_detection.tar.xz`), run:
```bash
python scripts/validate_dataset.py --extract "C:\path\to\mvtec_anomaly_detection.tar.xz"
```
Or use the Windows tar command:
```powershell
tar -xf "C:\path\to\mvtec_anomaly_detection.tar.xz" -C data/mvtec_ad/
```
