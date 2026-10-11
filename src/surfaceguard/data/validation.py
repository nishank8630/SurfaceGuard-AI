from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from surfaceguard.config import ALL_CATEGORIES, resolve_path
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_category(category_dir: Path) -> Dict[str, Any]:
    """Inspect and validate a single MVTec category directory."""
    result: Dict[str, Any] = {
        "category": category_dir.name,
        "valid": False,
        "train_good_count": 0,
        "test_good_count": 0,
        "test_defective_count": 0,
        "mask_count": 0,
        "defect_types": [],
        "missing_masks": [],
        "unreadable_images": [],
        "errors": [],
    }

    train_good_dir = category_dir / "train" / "good"
    if not train_good_dir.exists():
        result["errors"].append(f"Missing train/good directory: {train_good_dir}")
        return result

    # Validate train images
    train_files = list(train_good_dir.glob("*.png")) + list(train_good_dir.glob("*.jpg"))
    result["train_good_count"] = len(train_files)
    for p in train_files[:5]:  # Spot check readability
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as e:
            result["unreadable_images"].append(f"{p}: {e}")

    # Validate test images
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    if not test_dir.exists():
        result["errors"].append(f"Missing test directory: {test_dir}")
        return result

    defect_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    for d in defect_dirs:
        defect_name = d.name
        img_files = list(d.glob("*.png")) + list(d.glob("*.jpg"))
        if defect_name == "good":
            result["test_good_count"] += len(img_files)
        else:
            result["defect_types"].append(defect_name)
            result["test_defective_count"] += len(img_files)
            for img_path in img_files:
                expected_mask = gt_dir / defect_name / f"{img_path.stem}_mask.png"
                fallback_mask = gt_dir / defect_name / f"{img_path.stem}.png"
                if expected_mask.exists() or fallback_mask.exists():
                    result["mask_count"] += 1
                else:
                    result["missing_masks"].append(str(img_path))

    result["valid"] = (
        result["train_good_count"] > 0
        and result["test_good_count"] > 0
        and result["test_defective_count"] > 0
        and len(result["missing_masks"]) == 0
        and len(result["unreadable_images"]) == 0
    )
    return result


def validate_mvtec_dataset(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Validate all 15 categories within the MVTec AD root directory."""
    root = resolve_path(data_dir or "data/mvtec_ad")
    summary: Dict[str, Any] = {
        "dataset_root": str(root),
        "root_exists": root.exists(),
        "available_categories": [],
        "missing_categories": [],
        "category_results": {},
        "all_valid": False,
    }

    if not root.exists():
        summary["missing_categories"] = ALL_CATEGORIES.copy()
        return summary

    for cat in ALL_CATEGORIES:
        cat_dir = root / cat
        if cat_dir.exists() and cat_dir.is_dir():
            cat_summary = validate_category(cat_dir)
            summary["category_results"][cat] = cat_summary
            summary["available_categories"].append(cat)
        else:
            summary["missing_categories"].append(cat)

    summary["all_valid"] = (
        len(summary["missing_categories"]) == 0
        and all(res.get("valid", False) for res in summary["category_results"].values())
    )
    return summary
