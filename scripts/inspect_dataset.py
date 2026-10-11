import argparse
from pathlib import Path
from surfaceguard.config import ALL_CATEGORIES, resolve_path
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger("inspect_dataset")


def main():
    parser = argparse.ArgumentParser(description="Inspect MVTec AD dataset directory and category counts.")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Root dataset directory")
    args = parser.parse_args()

    root = resolve_path(args.data_dir)
    print("=" * 70)
    print("SURFACEGUARD AI - DATASET INSPECTION")
    print(f"Target Directory: {root.resolve()}")
    print("=" * 70)

    if not root.exists():
        print(f"Directory '{root}' does not exist.")
        print(f"Please extract MVTec AD to '{root}' or run 'python scripts/validate_dataset.py --extract <archive>'.")
        return

    found_cats = []
    missing_cats = []

    print(f"{'Category':<15} | {'Train Good':<10} | {'Test Good':<10} | {'Test Defects':<12} | {'Defect Types'}")
    print("-" * 70)

    for cat in ALL_CATEGORIES:
        cat_dir = root / cat
        if not cat_dir.exists():
            missing_cats.append(cat)
            continue

        found_cats.append(cat)
        train_good = len(list((cat_dir / "train" / "good").glob("*.*"))) if (cat_dir / "train" / "good").exists() else 0
        test_good = len(list((cat_dir / "test" / "good").glob("*.*"))) if (cat_dir / "test" / "good").exists() else 0

        defect_types = []
        test_defect_count = 0
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for d in sorted(test_dir.iterdir()):
                if d.is_dir() and d.name != "good":
                    count = len(list(d.glob("*.*")))
                    defect_types.append(f"{d.name} ({count})")
                    test_defect_count += count

        defects_str = ", ".join(defect_types) if defect_types else "None"
        if len(defects_str) > 35:
            defects_str = defects_str[:32] + "..."

        print(f"{cat:<15} | {train_good:<10} | {test_good:<10} | {test_defect_count:<12} | {defects_str}")

    print("-" * 70)
    print(f"Total Categories Present: {len(found_cats)} / {len(ALL_CATEGORIES)}")
    if missing_cats:
        print(f"Missing Categories ({len(missing_cats)}): {', '.join(missing_cats)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
