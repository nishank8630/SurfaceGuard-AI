import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

OBJECT_CATEGORIES: List[str] = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

TEXTURE_CATEGORIES: List[str] = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]

ALL_CATEGORIES: List[str] = sorted(OBJECT_CATEGORIES + TEXTURE_CATEGORIES)


def find_project_root() -> Path:
    """Locate the project root directory containing configs/ or setup.py."""
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (curr / "configs").exists() or (curr / "setup.py").exists() or (curr / ".git").exists():
            return curr
        curr = curr.parent
    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = find_project_root()


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict(dict1)
    for k, v in dict2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load default configuration and optionally overlay a specific config file."""
    default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config: Dict[str, Any] = {}

    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if config_path:
        custom_path = Path(config_path)
        if not custom_path.is_absolute():
            custom_path = PROJECT_ROOT / custom_path
        if custom_path.exists():
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_config = yaml.safe_load(f) or {}
                config = deep_merge(config, custom_config)

    # Environment variable overrides
    if "SURFACEGUARD_DATA_DIR" in os.environ:
        config.setdefault("paths", {})["data_dir"] = os.environ["SURFACEGUARD_DATA_DIR"]
    if "SURFACEGUARD_ARTIFACTS_DIR" in os.environ:
        config.setdefault("paths", {})["artifacts_dir"] = os.environ["SURFACEGUARD_ARTIFACTS_DIR"]
    if "SURFACEGUARD_RESULTS_DIR" in os.environ:
        config.setdefault("paths", {})["results_dir"] = os.environ["SURFACEGUARD_RESULTS_DIR"]

    return config


def resolve_path(rel_or_abs_path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    p = Path(rel_or_abs_path)
    if not p.is_absolute():
        return PROJECT_ROOT / p
    return p
