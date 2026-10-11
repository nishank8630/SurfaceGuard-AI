import argparse
import time
from torch.utils.data import DataLoader

from surfaceguard.config import ALL_CATEGORIES, resolve_path
from surfaceguard.data.dataset import MVTecDataset
from surfaceguard.models.anomaly_detector import PatchCoreDetector
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("build_feature_bank")


def process_category(
    category: str,
    data_dir: str,
    backbone: str = "resnet18",
    sampling_ratio: float = 0.05,
    sampling_method: str = "random",
    target_fpr: float = 0.01,
    device: str = "auto",
):
    dev = get_device(device)
    artifacts = ArtifactManager()

    logger.info(f"=== Building Feature Bank for '{category}' ===")
    train_dataset = MVTecDataset(category=category, split="train", data_dir=data_dir)
    val_dataset = MVTecDataset(category=category, split="val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False) if len(val_dataset) > 0 else None

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    detector = PatchCoreDetector(
        category=category,
        backbone_name=backbone,
        device=dev,
        sampling_ratio=sampling_ratio,
        sampling_method=sampling_method,
    )

    start_t = time.perf_counter()
    memory_bank = detector.fit(train_loader)
    elapsed = time.perf_counter() - start_t
    logger.info(f"Memory bank constructed in {elapsed:.1f}s. Bank shape: {memory_bank.shape}")

    # Save feature memory bank
    bank_path = artifacts.get_feature_bank_path(category, backbone)
    detector.save_memory_bank(bank_path)

    # Calibrate decision threshold on normal validation split
    val_scores = []
    if val_loader:
        logger.info("Calibrating decision threshold on normal validation samples...")
        for batch in val_loader:
            for img in batch["image"]:
                score, _ = detector.predict_sample(img)
                val_scores.append(score)

    threshold = detector.calibrate_threshold(val_scores, target_fpr=target_fpr)
    artifacts.save_threshold(
        category,
        "patchcore",
        threshold,
        {
            "backbone": backbone,
            "sampling_ratio": sampling_ratio,
            "sampling_method": sampling_method,
            "bank_vectors": memory_bank.shape[0],
            "feature_dim": memory_bank.shape[1],
            "target_fpr": target_fpr,
            "elapsed_seconds": elapsed,
        },
    )
    logger.info(f"Threshold calibrated: {threshold:.4f} (target FPR: {target_fpr*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Construct and save reference feature memory banks.")
    parser.add_argument("--category", type=str, default="bottle", help="Category name or 'all'")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Dataset directory")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone (resnet18, resnet50)")
    parser.add_argument("--sampling-ratio", type=float, default=0.05, help="Memory bank subsampling ratio (e.g. 0.05 - 0.20)")
    parser.add_argument("--sampling-method", type=str, default="random", choices=["coreset", "random"])
    parser.add_argument("--target-fpr", type=float, default=0.01, help="Target false positive rate on normal images (e.g. 0.01 for 1%)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    args = parser.parse_args()

    data_dir_path = resolve_path(args.data_dir)

    if args.category.lower() == "all":
        categories_to_process = [c for c in ALL_CATEGORIES if (data_dir_path / c).exists()]
        logger.info(f"Processing all available categories ({len(categories_to_process)}): {categories_to_process}")
    else:
        categories_to_process = [args.category]

    for cat in categories_to_process:
        try:
            process_category(
                category=cat,
                data_dir=args.data_dir,
                backbone=args.backbone,
                sampling_ratio=args.sampling_ratio,
                sampling_method=args.sampling_method,
                target_fpr=args.target_fpr,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"Failed processing category '{cat}': {e}", exc_info=True)


if __name__ == "__main__":
    main()
