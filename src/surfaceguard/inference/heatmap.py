import base64
import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def generate_heatmap_overlay(
    image: Union[np.ndarray, Image.Image],
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate colorized heatmap and blended overlay on the original image.
    
    Args:
        image: Original RGB image as PIL Image or uint8 np.ndarray (H, W, 3).
        anomaly_map: 2D float array (H, W) with values normalized [0.0, 1.0].
        alpha: Blending weight for heatmap overlay.
        colormap: OpenCV colormap enum (default cv2.COLORMAP_JET).
        
    Returns:
        heatmap_rgb: (H, W, 3) uint8 RGB heatmap.
        overlay_rgb: (H, W, 3) uint8 RGB blended overlay.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    H, W = img_np.shape[:2]

    # Ensure anomaly map matches image spatial dimensions
    if anomaly_map.shape[:2] != (H, W):
        anomaly_map = cv2.resize(anomaly_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255] uint8
    norm_map = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)

    # Apply color map: produces BGR
    heatmap_bgr = cv2.applyColorMap(norm_map, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blend overlay
    overlay_rgb = cv2.addWeighted(img_np, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return heatmap_rgb, overlay_rgb


def array_to_base64_jpeg(arr: np.ndarray, quality: int = 90) -> str:
    """Encode an RGB numpy array to base64 JPEG string."""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
