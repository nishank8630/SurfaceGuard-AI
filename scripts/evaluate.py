import argparse
from datetime import datetime
import json
from pathlib import Path
from torch.utils.data import DataLoader

from surfaceguard.config import resolve_path
from surfaceguard.data.dataset import MVTecDataset
from surfaceguard.models.classical_baseline import ClassicalBaseline
from surfaceguard.models.autoencoder import AutoencoderDetector
from surfaceguard.models.anomaly_detector import PatchCoreDetector
from surfaceguard.evaluation.evaluator import Evaluator
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model on test dataset.")
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g. bottle)")
    parser.add_argument("--model-type", type=str, default="patchcore", choices=["patchcore", "autoencoder", "classical"])
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Dataset directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    args = parser.parse_args()

    dev = get_device(args.device)
    artifacts = ArtifactManager()
    runs_dir = resolve_path("results/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    if args.model_type == "patchcore":
        detector = PatchCoreDetector(category=args.category, device=dev)
        bank_path = artifacts.get_feature_bank_path(args.category)
        detector.load_memory_bank(bank_path)
    elif args.model_type == "autoencoder":
        detector = AutoencoderDetector(category=args.category, device=dev)
        ckpt_path = artifacts.get_checkpoint_path(args.category, "autoencoder")
        detector.load(ckpt_path)
    elif args.model_type == "classical":
        detector = ClassicalBaseline(category=args.category, device=dev)
        # Fit statistical template on normal train images
        train_ds = MVTecDataset(category=args.category, split="train", data_dir=args.data_dir)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
        detector.fit(train_loader)

    # Load calibrated threshold
    calibrated_th = artifacts.load_threshold(args.category, args.model_type)
    if calibrated_th is not None:
        detector.calibrated_threshold = calibrated_th

    # Load test dataset
    test_ds = MVTecDataset(category=args.category, split="test", data_dir=args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    evaluator = Evaluator(detector)
    results = evaluator.evaluate(test_loader)

    # Save run record
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = runs_dir / f"{args.category}_{args.model_type}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {args.category.upper()} [{args.model_type.upper()}]")
    print("=" * 60)
    img_m = results["image_metrics"]
    pix_m = results["pixel_metrics"]
    lat_m = results["latency"]

    print(f"* Image AUROC         : {img_m['auroc']}")
    print(f"* Image AP            : {img_m['average_precision']}")
    print(f"* F1 Score            : {img_m['f1_score']:.4f} (at threshold {img_m['threshold']:.4f})")
    print(f"* Confusion Matrix    : TP={img_m['true_positives']} | FP={img_m['false_positives']} | TN={img_m['true_negatives']} | FN={img_m['false_negatives']}")
    print("-" * 60)
    print(f"* Pixel AUROC         : {pix_m['pixel_auroc']}")
    print(f"* Pixel AP            : {pix_m['pixel_ap']}")
    print(f"* Pixel IoU           : {pix_m['pixel_iou']:.4f}")
    print("-" * 60)
    print(f"* Mean Latency        : {lat_m['mean_ms']:.2f} ms")
    print(f"* Median Latency      : {lat_m['median_ms']:.2f} ms")
    print(f"* 95th Percentile     : {lat_m['p95_ms']:.2f} ms")
    print(f"* Throughput          : {lat_m['throughput_fps']:.1f} FPS")
    print("=" * 60)
    print(f"Detailed run saved to: {out_file}\n")


if __name__ == "__main__":
    main()
