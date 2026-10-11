from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResNetFeatureExtractor(nn.Module):
    """Extracts and aggregates multi-scale patch features from intermediate ResNet layers."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.backbone_name = backbone_name.lower()
        self.selected_layers = layers or ["layer2", "layer3"]

        # Load pretrained backbone
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone.to(self.device)
        self.backbone.eval()

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_maps: List[torch.Tensor] = []
        self._register_hooks()

    def _hook(self, module, input, output):
        self.feature_maps.append(output)

    def _register_hooks(self):
        for name, module in self.backbone.named_children():
            if name in self.selected_layers:
                module.register_forward_hook(self._hook)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract multi-layer feature patches.
        
        Input: (B, 3, H, W)
        Output:
            patch_features: (B * H_patch * W_patch, D)
            spatial_shape: (H_patch, W_patch)
        """
        self.feature_maps.clear()
        _ = self.backbone(x.to(self.device))

        # Target spatial shape matches layer2 resolution
        target_h, target_w = self.feature_maps[0].shape[2], self.feature_maps[0].shape[3]
        pool = nn.AdaptiveAvgPool2d((target_h, target_w))

        aligned_features = []
        for feat in self.feature_maps:
            if feat.shape[2:] != (target_h, target_w):
                feat = pool(feat)
            aligned_features.append(feat)

        # Concatenate channels: (B, C1 + C2, H_p, W_p)
        concat_feats = torch.cat(aligned_features, dim=1)
        B, C, H_p, W_p = concat_feats.shape

        # Reshape to patch representations: (B, H_p, W_p, C) -> (B * H_p * W_p, C)
        patches = concat_feats.permute(0, 2, 3, 1).reshape(B * H_p * W_p, C)
        return patches, (H_p, W_p)
