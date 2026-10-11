import argparse
from datetime import datetime
import json
from pathlib import Path
from torch.utils.data import DataLoader

from surfaceguard.config import ALL_CATEGORIES, resolve_path
from surfaceguard.data.dataset import MVTecDataset
from surfaceguard.models.classical_baseline import ClassicalBaseline
from surfaceguard.models.autoencoder import AutoencoderDetector
from surfaceguard.models.anomaly_detector import PatchCoreDetector
from surfaceguard.evaluation.evaluator import Evaluator
from surfaceguard.evaluation.aggregation import aggregate_benchmark_results, generate_macro_summary
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("run_benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run multi-category benchmark across MVTec AD.")
    parser.add_argument("--model-type", type=str, default="patchcore", choices=["patchcore", "autoencoder", "classical"])
    parser.add_argument("--categories", nargs="+", default=["all"], help="List of categories or 'all'")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Dataset directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    args = parser.parse_args()

    dev = get_device(args.device)
    artifacts = ArtifactManager()
    data_dir_path = resolve_path(args.data_dir)
    reports_dir = resolve_path("results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.categories == ["all"]:
        candidate_cats = [c for c in ALL_CATEGORIES if (data_dir_path / c).exists()]
    else:
        candidate_cats = args.categories

    if not candidate_cats:
        logger.error(f"No categories available to benchmark in {data_dir_path}")
        return

    logger.info(f"Starting benchmark for {len(candidate_cats)} categories: {candidate_cats}")

    benchmark_results = []
    for cat in candidate_cats:
        logger.info(f"\n---> Evaluating Category: {cat}")
        try:
            if args.model_type == "patchcore":
                bank_path = artifacts.get_feature_bank_path(cat)
                if not bank_path.exists():
                    logger.warning(f"Memory bank missing for '{cat}'. Skipping. (Run scripts/build_feature_bank.py --category {cat})")
                    continue
                detector = PatchCoreDetector(category=cat, device=dev)
                detector.load_memory_bank(bank_path)

            elif args.model_type == "autoencoder":
                ckpt_path = artifacts.get_checkpoint_path(cat, "autoencoder")
                if not ckpt_path.exists():
                    logger.warning(f"Checkpoint missing for '{cat}'. Skipping. (Run scripts/train_autoencoder.py --category {cat})")
                    continue
                detector = AutoencoderDetector(category=cat, device=dev)
                detector.load(ckpt_path)

            elif args.model_type == "classical":
                detector = ClassicalBaseline(category=cat, device=dev)
                train_ds = MVTecDataset(category=cat, split="train", data_dir=args.data_dir)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
                detector.fit(train_loader)

            # Load threshold if present
            thresh = artifacts.load_threshold(cat, args.model_type)
            if thresh is not None:
                detector.calibrated_threshold = thresh

            # Evaluate
            test_ds = MVTecDataset(category=cat, split="test", data_dir=args.data_dir)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
            evaluator = Evaluator(detector)
            res = evaluator.evaluate(test_loader)
            benchmark_results.append(res)

        except Exception as e:
            logger.error(f"Failed benchmark for category '{cat}': {e}", exc_info=True)

    if not benchmark_results:
        logger.warning("No benchmark results generated.")
        return

    df = aggregate_benchmark_results(benchmark_results)
    macro = generate_macro_summary(df)

    # Save summary CSV
    csv_path = reports_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark summary CSV to {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SURFACEGUARD AI — MULTI-CATEGORY BENCHMARK SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    if "macro_average" in macro:
        print("MACRO AVERAGES:")
        for k, v in macro["macro_average"].items():
            if v is not None:
                print(f"  * {k:<25} : {v:.4f}")
        obj_val = macro.get("object_average_auroc")
        tex_val = macro.get("texture_average_auroc")
        print(f"  * Object Mean AUROC       : {obj_val:.4f}" if obj_val is not None else "  * Object Mean AUROC       : N/A")
        print(f"  * Texture Mean AUROC      : {tex_val:.4f}" if tex_val is not None else "  * Texture Mean AUROC      : N/A")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
