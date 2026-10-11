from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from PIL import Image

from surfaceguard.config import resolve_path, ALL_CATEGORIES
from surfaceguard.data.transforms import get_transforms, get_mask_transforms
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MVTecDataset(Dataset):
    """PyTorch Dataset for MVTec AD supporting all 15 categories.
    
    Structure:
        data_dir/
            <category>/
                train/
                    good/
                test/
                    good/
                    <defect_type_1>/
                    ...
                ground_truth/
                    <defect_type_1>/
                        <stem>_mask.png
    """

    def __init__(
        self,
        category: str,
        split: str = "test",  # "train", "val", or "test"
        data_dir: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        self.category = category.lower().strip()
        self.split = split.lower().strip()
        self.data_dir = resolve_path(data_dir or "data/mvtec_ad")
        self.category_dir = self.data_dir / self.category
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: '{self.category_dir}'. "
                f"Verify MVTec AD is extracted under '{self.data_dir}'."
            )

        self.transform = transform or get_transforms(image_size=image_size)
        self.mask_transform = mask_transform or get_mask_transforms(image_size=image_size)

        self.samples: List[Dict[str, Union[str, int]]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory and index samples according to the requested split."""
        train_good_dir = self.category_dir / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Missing train directory: {train_good_dir}")

        train_images = sorted(
            [p for p in train_good_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
        )

        if self.split in ["train", "val"]:
            # Deterministic train/validation split of normal training samples
            rng = torch.Generator().manual_seed(self.random_seed)
            perm = torch.randperm(len(train_images), generator=rng).tolist()
            val_count = max(1, int(len(train_images) * self.val_ratio)) if len(train_images) > 5 else 0

            if self.split == "val":
                selected_indices = perm[:val_count]
            else:  # "train"
                selected_indices = perm[val_count:]

            for idx in selected_indices:
                img_path = train_images[idx]
                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path": "",
                    "label": 0,  # 0: normal
                    "defect_type": "good",
                    "category": self.category,
                })

        elif self.split == "test":
            test_dir = self.category_dir / "test"
            gt_dir = self.category_dir / "ground_truth"

            if not test_dir.exists():
                raise FileNotFoundError(f"Missing test directory: {test_dir}")

            defect_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
            for d in defect_dirs:
                defect_name = d.name
                is_anomalous = (defect_name != "good")
                img_files = sorted(
                    [p for p in d.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
                )

                for img_path in img_files:
                    mask_path = ""
                    if is_anomalous:
                        # Match mask path: ground_truth/<defect_name>/<stem>_mask.png
                        expected_mask = gt_dir / defect_name / f"{img_path.stem}_mask.png"
                        if not expected_mask.exists():
                            # Alternate fallback: <stem>.png
                            expected_mask = gt_dir / defect_name / f"{img_path.stem}.png"
                        if expected_mask.exists():
                            mask_path = str(expected_mask)
                        else:
                            logger.warning(f"Ground-truth mask missing for {img_path}")

                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": mask_path,
                        "label": 1 if is_anomalous else 0,
                        "defect_type": defect_name,
                        "category": self.category,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)

        # Handle mask
        if sample["mask_path"]:
            with Image.open(sample["mask_path"]) as mask_img:
                mask_img = mask_img.convert("L")
                mask_tensor = self.mask_transform(mask_img)
                # Binarize mask (0.0 or 1.0)
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Normal sample: zero mask of shape (1, H, W)
            H, W = self.image_size
            mask_tensor = torch.zeros((1, H, W), dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": sample["label"],
            "category": sample["category"],
            "defect_type": sample["defect_type"],
            "image_path": img_path,
        }
