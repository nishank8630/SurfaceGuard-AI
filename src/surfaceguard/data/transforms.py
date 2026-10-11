from typing import Tuple, Union
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: Union[int, Tuple[int, int]] = (256, 256),
    normalize: bool = True,
) -> T.Compose:
    """Standard image transforms for inference and evaluation."""
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    transforms_list = [
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if normalize:
        transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms_list)


def get_mask_transforms(
    image_size: Union[int, Tuple[int, int]] = (256, 256),
) -> T.Compose:
    """Transforms for ground-truth masks: resize with nearest neighbor and convert to binary float tensor."""
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Reverts ImageNet normalization for visualization. Expects (C, H, W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0.0, 1.0)
