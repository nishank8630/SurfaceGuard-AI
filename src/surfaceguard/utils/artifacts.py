import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from surfaceguard.config import resolve_path, ALL_CATEGORIES
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """Manages serialization, loading, and availability discovery of model artifacts."""

    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = resolve_path(artifacts_dir or "artifacts")
        self.banks_dir = self.artifacts_dir / "feature_banks"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.thresholds_dir = self.artifacts_dir / "thresholds"

        self.banks_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds_dir.mkdir(parents=True, exist_ok=True)

    def get_feature_bank_path(self, category: str, backbone: str = "resnet18") -> Path:
        return self.banks_dir / f"{category}_{backbone}_bank.pt"

    def get_checkpoint_path(self, category: str, model_type: str = "autoencoder") -> Path:
        return self.checkpoints_dir / f"{category}_{model_type}.pt"

    def get_threshold_path(self, category: str, model_type: str = "patchcore") -> Path:
        return self.thresholds_dir / f"{category}_{model_type}_threshold.json"

    def save_feature_bank(self, category: str, memory_bank: torch.Tensor, metadata: Dict[str, Any], backbone: str = "resnet18") -> Path:
        out_path = self.get_feature_bank_path(category, backbone)
        payload = {
            "category": category,
            "backbone": backbone,
            "memory_bank": memory_bank.cpu(),
            "metadata": metadata,
        }
        torch.save(payload, out_path)
        logger.info(f"Saved feature memory bank for '{category}' to {out_path} ({memory_bank.shape})")
        return out_path

    def load_feature_bank(self, category: str, backbone: str = "resnet18") -> Dict[str, Any]:
        bank_path = self.get_feature_bank_path(category, backbone)
        if not bank_path.exists():
            raise FileNotFoundError(
                f"Feature memory bank for category '{category}' not found at {bank_path}. "
                f"Run scripts/build_feature_bank.py --category {category} first."
            )
        data = torch.load(bank_path, map_location="cpu", weights_only=False)
        return data

    def save_checkpoint(self, category: str, model_state_dict: dict, metadata: Dict[str, Any], model_type: str = "autoencoder") -> Path:
        out_path = self.get_checkpoint_path(category, model_type)
        payload = {
            "category": category,
            "model_type": model_type,
            "state_dict": model_state_dict,
            "metadata": metadata,
        }
        torch.save(payload, out_path)
        logger.info(f"Saved checkpoint for '{category}' ({model_type}) to {out_path}")
        return out_path

    def load_checkpoint(self, category: str, model_type: str = "autoencoder") -> Dict[str, Any]:
        ckpt_path = self.get_checkpoint_path(category, model_type)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for category '{category}' ({model_type}) not found at {ckpt_path}. "
                f"Train the model using scripts/train_{model_type}.py first."
            )
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def save_threshold(self, category: str, model_type: str, threshold: float, metadata: Optional[Dict[str, Any]] = None) -> Path:
        out_path = self.get_threshold_path(category, model_type)
        payload = {
            "category": category,
            "model_type": model_type,
            "threshold": float(threshold),
            "metadata": metadata or {},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved calibrated threshold {threshold:.4f} for '{category}' ({model_type}) to {out_path}")
        return out_path

    def load_threshold(self, category: str, model_type: str) -> Optional[float]:
        thresh_path = self.get_threshold_path(category, model_type)
        if not thresh_path.exists():
            return None
        with open(thresh_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("threshold", 0.5))

    def list_available_categories(self, model_type: str = "patchcore") -> Dict[str, bool]:
        """Check which of the 15 categories have artifacts ready."""
        availability = {}
        for cat in ALL_CATEGORIES:
            if model_type == "patchcore":
                avail = self.get_feature_bank_path(cat).exists()
            elif model_type == "autoencoder":
                avail = self.get_checkpoint_path(cat, "autoencoder").exists()
            else:
                avail = True  # Classical baseline requires no precomputed weights
            availability[cat] = avail
        return availability
