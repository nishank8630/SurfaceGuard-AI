import numpy as np
from PIL import Image
import torch
from surfaceguard.inference.heatmap import generate_heatmap_overlay, array_to_base64_jpeg
from surfaceguard.inference.predictor import SurfaceGuardPredictor
from surfaceguard.utils.artifacts import ArtifactManager


def test_heatmap_overlay_generation():
    img = Image.new("RGB", (64, 64), color="blue")
    amap = np.random.rand(64, 64).astype(np.float32)

    heatmap_rgb, overlay_rgb = generate_heatmap_overlay(img, amap)
    assert heatmap_rgb.shape == (64, 64, 3)
    assert overlay_rgb.shape == (64, 64, 3)

    b64 = array_to_base64_jpeg(overlay_rgb)
    assert b64.startswith("data:image/jpeg;base64,")


def test_predictor_classical_baseline(tmp_path):
    # Predictor with classical baseline requires no precomputed weights
    predictor = SurfaceGuardPredictor(artifacts_dir=str(tmp_path))
    test_img = Image.new("RGB", (64, 64), color="red")

    res = predictor.predict_image(image=test_img, category="bottle", model_type="classical")
    assert res["category"] == "bottle"
    assert "is_anomalous" in res
    assert "anomaly_score" in res
    assert "threshold" in res
    assert "heatmap_base64" in res
    assert "overlay_base64" in res
    assert res["inference_time_ms"] > 0
