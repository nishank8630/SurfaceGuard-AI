import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from surfaceguard.data.dataset import MVTecDataset
from surfaceguard.models.classical_baseline import ClassicalBaseline
from surfaceguard.models.autoencoder import ConvAutoencoderNet, AutoencoderDetector
from surfaceguard.models.feature_extractor import ResNetFeatureExtractor
from surfaceguard.models.feature_bank import subsample_memory_bank, compute_min_distances_chunked
from surfaceguard.models.anomaly_detector import PatchCoreDetector


def test_classical_baseline(synthetic_data_dir):
    ds = MVTecDataset(category="bottle", split="train", data_dir=synthetic_data_dir, image_size=(64, 64))
    loader = DataLoader(ds, batch_size=2)
    model = ClassicalBaseline(category="bottle")
    model.fit(loader)
    assert model.is_fitted

    test_img = torch.randn(3, 64, 64)
    score, amap = model.predict_sample(test_img)
    assert isinstance(score, float)
    assert amap.shape == (64, 64)
    assert amap.min() >= 0.0 and amap.max() <= 1.0


def test_conv_autoencoder_forward():
    net = ConvAutoencoderNet(base_channels=16, latent_dim=32)
    x = torch.randn(2, 3, 64, 64)
    out = net(x)
    assert out.shape == (2, 3, 64, 64)


def test_autoencoder_detector(synthetic_data_dir):
    train_ds = MVTecDataset(category="bottle", split="train", data_dir=synthetic_data_dir, image_size=(64, 64))
    loader = DataLoader(train_ds, batch_size=2)
    model = AutoencoderDetector(category="bottle", base_channels=16, latent_dim=32)
    history = model.fit(loader, epochs=1)
    assert len(history["train_loss"]) == 1

    test_img = torch.randn(3, 64, 64)
    score, amap = model.predict_sample(test_img)
    assert isinstance(score, float)
    assert amap.shape == (64, 64)


def test_feature_extractor_and_chunked_distances():
    extractor = ResNetFeatureExtractor(backbone_name="resnet18")
    x = torch.randn(2, 3, 64, 64)
    patches, (Hp, Wp) = extractor(x)
    assert patches.dim() == 2
    assert patches.shape[0] == 2 * Hp * Wp

    # Test coreset subsampling
    reduced = subsample_memory_bank(patches, sampling_ratio=0.5, method="coreset")
    assert reduced.shape[0] == max(10, int(patches.shape[0] * 0.5))

    # Test chunked min distance computation
    query = torch.randn(15, patches.shape[1])
    dists = compute_min_distances_chunked(query, reduced, chunk_size=5)
    assert dists.shape == (15,)
    assert torch.all(dists >= 0.0)


def test_patchcore_detector(synthetic_data_dir):
    train_ds = MVTecDataset(category="bottle", split="train", data_dir=synthetic_data_dir, image_size=(64, 64))
    loader = DataLoader(train_ds, batch_size=2)
    model = PatchCoreDetector(category="bottle", backbone_name="resnet18", sampling_ratio=0.2)
    bank = model.fit(loader)
    assert bank.shape[0] > 0

    test_img = torch.randn(3, 64, 64)
    score, amap = model.predict_sample(test_img)
    assert isinstance(score, float)
    assert amap.shape == (64, 64)
