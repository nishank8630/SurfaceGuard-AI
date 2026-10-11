import argparse
import time
from torch.utils.data import DataLoader

from surfaceguard.config import resolve_path
from surfaceguard.data.dataset import MVTecDataset
from surfaceguard.models.autoencoder import AutoencoderDetector
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("train_autoencoder")


def main():
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder on normal samples.")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name (e.g. bottle)")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    args = parser.parse_args()

    dev = get_device(args.device)
    artifacts = ArtifactManager()

    logger.info(f"Loading normal training and validation datasets for '{args.category}'...")
    train_dataset = MVTecDataset(category=args.category, split="train", data_dir=args.data_dir)
    val_dataset = MVTecDataset(category=args.category, split="val", data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if len(val_dataset) > 0 else None

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    detector = AutoencoderDetector(category=args.category, device=dev)
    start_t = time.perf_counter()

    history = detector.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
    )
    elapsed = time.perf_counter() - start_t
    logger.info(f"Autoencoder training finished in {elapsed:.1f}s.")

    # Save checkpoint
    ckpt_path = artifacts.get_checkpoint_path(args.category, "autoencoder")
    detector.save(ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")

    # Threshold calibration on validation normal samples
    val_scores = []
    if val_loader:
        for batch in val_loader:
            for img in batch["image"]:
                score, _ = detector.predict_sample(img)
                val_scores.append(score)

    threshold = detector.calibrate_threshold(val_scores, target_fpr=0.05)
    artifacts.save_threshold(args.category, "autoencoder", threshold, {"epochs": args.epochs, "training_time_s": elapsed})
    logger.info(f"Calibrated threshold: {threshold:.4f} (target FPR: 5%)")


if __name__ == "__main__":
    main()
