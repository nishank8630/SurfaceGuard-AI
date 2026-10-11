import pytest
import torch
from surfaceguard.data.dataset import MVTecDataset


def test_dataset_train_split(synthetic_data_dir):
    ds = MVTecDataset(category="bottle", split="train", data_dir=synthetic_data_dir, image_size=(64, 64))
    assert len(ds) > 0
    sample = ds[0]
    assert "image" in sample
    assert "mask" in sample
    assert "label" in sample
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert sample["label"] == 0
    # Normal masks must be all zeros
    assert torch.sum(sample["mask"]) == 0.0


def test_dataset_val_split(synthetic_data_dir):
    ds = MVTecDataset(category="bottle", split="val", data_dir=synthetic_data_dir, image_size=(64, 64))
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["label"] == 0


def test_dataset_test_split(synthetic_data_dir):
    ds = MVTecDataset(category="bottle", split="test", data_dir=synthetic_data_dir, image_size=(64, 64))
    # 3 good + 3 scratch = 6 test samples
    assert len(ds) == 6

    # Verify both normal and anomalous items exist
    labels = [s["label"] for s in ds]
    assert 0 in labels
    assert 1 in labels

    # Defect samples should have non-zero masks
    defect_sample = [ds[i] for i in range(len(ds)) if ds[i]["label"] == 1][0]
    assert torch.sum(defect_sample["mask"]) > 0.0


def test_missing_category_raises(synthetic_data_dir):
    with pytest.raises(FileNotFoundError):
        MVTecDataset(category="nonexistent_product", data_dir=synthetic_data_dir)
