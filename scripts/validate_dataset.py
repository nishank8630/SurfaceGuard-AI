import argparse
import os
from pathlib import Path
import subprocess
import tarfile
from surfaceguard.config import resolve_path, ALL_CATEGORIES
from surfaceguard.data.validation import validate_mvtec_dataset
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("validate_dataset")


def extract_archive(archive_path: Path, dest_dir: Path, category_filter: str = None) -> None:
    """Extracts tar.xz archive to target destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting '{archive_path}' to '{dest_dir}'...")

    # On Windows 10/11, bsdtar is significantly faster than python tarfile for multi-GB xz archives
    try:
        cmd = ["tar", "-xf", str(archive_path), "-C", str(dest_dir)]
        if category_filter:
            cmd.append(category_filter)
        logger.info(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("Extraction complete via system tar.")
            return
        else:
            logger.warning(f"System tar returned code {res.returncode}: {res.stderr}. Trying Python tarfile...")
    except Exception as e:
        logger.warning(f"Could not use system tar ({e}). Falling back to python tarfile.")

    # Fallback to Python tarfile
    with tarfile.open(archive_path, "r:xz") as tar:
        if category_filter:
            members = [m for m in tar.getmembers() if m.name.startswith(category_filter)]
            tar.extractall(path=dest_dir, members=members)
        else:
            tar.extractall(path=dest_dir)
    logger.info("Extraction complete via python tarfile.")


def main():
    parser = argparse.ArgumentParser(description="Validate MVTec AD dataset directory integrity.")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Root dataset directory")
    parser.add_argument("--extract", type=str, default=None, help="Optional path to mvtec_anomaly_detection.tar.xz to extract")
    parser.add_argument("--extract-category", type=str, default=None, help="Extract only a specific category from the archive")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)

    if args.extract:
        archive_path = Path(args.extract)
        if not archive_path.exists():
            logger.error(f"Archive file not found: {archive_path}")
            return
        extract_archive(archive_path, data_dir, args.extract_category)

    print("\n" + "=" * 70)
    print("SURFACEGUARD AI — DATASET INTEGRITY VALIDATION")
    print(f"Dataset Root: {data_dir.resolve()}")
    print("=" * 70)

    summary = validate_mvtec_dataset(str(data_dir))

    if not summary["root_exists"]:
        print(f"\n❌ Dataset root does not exist: {data_dir}")
        print("Please download mvtec_anomaly_detection.tar.xz from https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
        print(f"and extract it into '{data_dir}'.")
        return

    print(f"\nDiscovered Categories ({len(summary['available_categories'])} / {len(ALL_CATEGORIES)}):")
    for cat in summary["available_categories"]:
        res = summary["category_results"][cat]
        status_icon = "[VALID]" if res["valid"] else "[INVALID]"
        print(f"  * {cat:<12} : {status_icon} | Train: {res['train_good_count']} | Test Good: {res['test_good_count']} | Test Defect: {res['test_defective_count']} | Masks: {res['mask_count']}")
        if res["missing_masks"]:
            print(f"      Missing masks ({len(res['missing_masks'])}): {res['missing_masks'][:2]}...")

    if summary["missing_categories"]:
        print(f"\nMissing Categories ({len(summary['missing_categories'])}):")
        print(f"  {', '.join(summary['missing_categories'])}")

    print("\n" + "=" * 70)
    if summary["all_valid"]:
        print("ALL 15 CATEGORIES VALID AND READY FOR BENCHMARKING!")
    elif len(summary["available_categories"]) > 0:
        print(f"[OK] Ready to evaluate {len(summary['available_categories'])} available categories.")
    else:
        print("[FAIL] No valid categories found.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
