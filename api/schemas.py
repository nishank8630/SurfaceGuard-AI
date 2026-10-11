from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "SurfaceGuard AI Inference Backend"
    pytorch_version: str
    cuda_available: bool
    device_name: str


class CategoriesResponse(BaseModel):
    all_categories: List[str]
    available_categories: Dict[str, bool]
    count_ready: int
    count_total: int


class PredictionResponse(BaseModel):
    category: str
    model_type: str
    is_anomalous: bool
    anomaly_score: float
    threshold: float
    inference_time_ms: float
    heatmap_base64: str = Field(..., description="Base64 encoded JPEG heatmap")
    overlay_base64: str = Field(..., description="Base64 encoded JPEG overlay")
    original_size: List[int]
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
