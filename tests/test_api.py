import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pytorch_version" in data
    assert "device_name" in data


def test_categories_endpoint():
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data["all_categories"]) == 15
    assert "bottle" in data["all_categories"]
    assert "count_total" in data
    assert data["count_total"] == 15


def test_predict_endpoint_classical():
    img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        data={"category": "bottle", "model_type": "classical"},
        files={"file": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "bottle"
    assert "anomaly_score" in data
    assert "overlay_base64" in data


def test_predict_invalid_category():
    img = Image.new("RGB", (32, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        data={"category": "non_existent_category", "model_type": "classical"},
        files={"file": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 400
