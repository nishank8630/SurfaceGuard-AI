from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def segment_and_normalize_object(
    image: Image.Image,
    target_size: Tuple[int, int] = (256, 256),
    bg_mode: str = "dark",  # "dark" to match MVTec black conveyor, or "neutral"
) -> Tuple[Image.Image, np.ndarray]:
    """Isolates the target foreground object, removes background clutter,
    and centers the object onto the canonical MVTec conveyor background.
    
    Returns:
        normalized_image: PIL Image of shape target_size ready for PatchCore
        foreground_mask: Binary mask (H, W) where 1 indicates object region
    """
    img_rgb = np.array(image.convert("RGB"))
    H, W = img_rgb.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Detect whether background is bright (e.g. white retail photo) or dark
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    avg_corner_intensity = float(np.mean(corners))
    is_bright_bg = avg_corner_intensity > 180

    if is_bright_bg:
        # Background is white/light: threshold out the bright background
        _, bin_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    else:
        # Dark or natural background: use Canny edge closing & Otsu threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bin_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find the largest connected foreground contour
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        # Create solid mask of the object
        solid_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(solid_mask, [c], -1, 255, thickness=cv2.FILLED)

        # Smooth edges
        solid_mask = cv2.GaussianBlur(solid_mask, (7, 7), 0)
        alpha = (solid_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # Crop to bounding box
        x, y, w, h = cv2.boundingRect(c)
        cropped_obj = img_rgb[y : y + h, x : x + w]
        cropped_alpha = alpha[y : y + h, x : x + w]

        # Canonical canvas (256x256) with dark MVTec-style background
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        if bg_mode == "neutral":
            canvas[:] = [30, 30, 30]

        # Resize object maintaining aspect ratio to fit inside canvas (with padding)
        scale = min((target_size[0] * 0.85) / h, (target_size[1] * 0.70) / w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized_obj = cv2.resize(cropped_obj, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_alpha = cv2.resize(cropped_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]

        # Center on canvas
        start_y = (target_size[0] - new_h) // 2
        start_x = (target_size[1] - new_w) // 2

        # Alpha blend object onto the canonical dark inspection background
        canvas_crop = canvas[start_y : start_y + new_h, start_x : start_x + new_w]
        blended = (resized_obj * resized_alpha + canvas_crop * (1.0 - resized_alpha)).astype(np.uint8)
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = blended

        norm_image = Image.fromarray(canvas)
        return norm_image, resized_alpha.squeeze()

    # Fallback if no clean contour found
    norm_image = image.resize(target_size, Image.BILINEAR)
    return norm_image, np.ones(target_size, dtype=np.float32)
