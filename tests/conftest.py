import sys
from pathlib import Path
import pytest

# Ensure src/ is on sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from surfaceguard.data.synthetic import create_synthetic_mvtec_dataset


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Session fixture creating a miniature synthetic MVTec dataset structure."""
    temp_dir = tmp_path_factory.mktemp("synthetic_mvtec")
    created_path = create_synthetic_mvtec_dataset(
        output_dir=str(temp_dir),
        categories=["bottle", "tile"],
        image_size=64,
        train_count=6,
        test_good_count=3,
        test_defect_count=3,
    )
    return created_path
