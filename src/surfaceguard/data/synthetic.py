import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw

from surfaceguard.config import resolve_path


def create_synthetic_mvtec_dataset(
    output_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    image_size: int = 128,
    train_count: int = 10,
    test_good_count: int = 5,
    test_defect_count: int = 5,
) -> Path:
    """Creates a miniature synthetic MVTec-like dataset structure for fast offline testing."""
    root = resolve_path(output_dir or "data/synthetic_mvtec")
    cats = categories or ["bottle", "tile"]

    for cat in cats:
        cat_dir = root / cat
        train_good = cat_dir / "train" / "good"
        test_good = cat_dir / "test" / "good"
        test_defect = cat_dir / "test" / "scratch"
        gt_defect = cat_dir / "ground_truth" / "scratch"

        for p in [train_good, test_good, test_defect, gt_defect]:
            p.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)

        # Generate normal training images
        for i in range(train_count):
            base_pattern = (np.random.rand(image_size, image_size, 3) * 50 + 100).astype(np.uint8)
            img = Image.fromarray(base_pattern)
            img.save(train_good / f"{i:03d}.png")

        # Generate normal test images
        for i in range(test_good_count):
            base_pattern = (np.random.rand(image_size, image_size, 3) * 50 + 100).astype(np.uint8)
            img = Image.fromarray(base_pattern)
            img.save(test_good / f"{i:03d}.png")

        # Generate defective test images and masks
        for i in range(test_defect_count):
            base_pattern = (np.random.rand(image_size, image_size, 3) * 50 + 100).astype(np.uint8)
            img = Image.fromarray(base_pattern)
            mask = Image.new("L", (image_size, image_size), 0)

            # Draw a bright anomaly line/polygon
            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            x1 = np.random.randint(20, image_size - 40)
            y1 = np.random.randint(20, image_size - 40)
            x2 = x1 + np.random.randint(10, 30)
            y2 = y1 + np.random.randint(10, 30)

            draw_img.rectangle([x1, y1, x2, y2], fill=(255, 0, 0))
            draw_mask.rectangle([x1, y1, x2, y2], fill=255)

            img.save(test_defect / f"{i:03d}.png")
            mask.save(gt_defect / f"{i:03d}_mask.png")

    return root
