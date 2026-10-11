from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from surfaceguard.models.base import BaseAnomalyDetector
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConvAutoencoderNet(nn.Module):
    """Convolutional Autoencoder architecture for reconstruction-based anomaly detection."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        # Encoder: (B, 3, 256, 256) -> (B, latent_dim, 16, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, latent_dim, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: (B, latent_dim, 16, 16) -> (B, 3, 256, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),  # -> 256x256
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderDetector(BaseAnomalyDetector):
    """Method B: Autoencoder Anomaly Detector.
    
    Trained strictly on defect-free samples. Anomalies exhibit higher reconstruction error.
    """

    def __init__(
        self,
        category: str,
        device: Optional[torch.device] = None,
        base_channels: int = 32,
        latent_dim: int = 128,
        blur_sigma: float = 4.0,
    ):
        super().__init__(category, device)
        self.net = ConvAutoencoderNet(base_channels=base_channels, latent_dim=latent_dim).to(self.device)
        self.blur_sigma = blur_sigma

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        lr: float = 1e-3,
        patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Train autoencoder on defect-free images only."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        self.net.train()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            total_train_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device)
                optimizer.zero_grad()
                recon = self.net(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * imgs.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            history["train_loss"].append(avg_train_loss)

            # Validation loss if provided
            avg_val_loss = avg_train_loss
            if val_loader and len(val_loader.dataset) > 0:
                self.net.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(self.device)
                        recon = self.net(imgs)
                        v_loss = criterion(recon, imgs)
                        total_val_loss += v_loss.item() * imgs.size(0)
                avg_val_loss = total_val_loss / len(val_loader.dataset)
                self.net.train()

            history["val_loss"].append(avg_val_loss)

            if epoch % 5 == 0 or epoch == epochs:
                logger.info(f"[{self.category}] Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= 10:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        self.is_fitted = True
        return history

    def predict_sample(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Compute pixel-level reconstruction residual map and anomaly score."""
        self.net.eval()
        with torch.no_grad():
            img_batch = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
            recon = self.net(img_batch)
            diff = torch.abs(img_batch - recon)  # (1, 3, H, W)
            diff_map = torch.mean(diff, dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # Smooth reconstruction residual
        ksize = int(6 * self.blur_sigma + 1) | 1
        smoothed_map = cv2.GaussianBlur(diff_map.astype(np.float32), (ksize, ksize), self.blur_sigma)

        # Normalized map for visualization [0, 1]
        min_v, max_v = smoothed_map.min(), smoothed_map.max()
        norm_map = (smoothed_map - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(smoothed_map)

        # Image-level anomaly score: top 98th percentile error
        image_score = float(np.percentile(smoothed_map, 98))
        return image_score, norm_map

    def save(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.is_fitted = True
