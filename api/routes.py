import io
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from surfaceguard.config import ALL_CATEGORIES
from surfaceguard.inference.predictor import SurfaceGuardPredictor
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device_info
from surfaceguard.utils.logging_utils import get_logger
from api.schemas import HealthResponse, CategoriesResponse, PredictionResponse

logger = get_logger(__name__)

router = APIRouter()
predictor = SurfaceGuardPredictor()
artifact_mgr = ArtifactManager()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service status and device configuration."""
    dev_info = get_device_info()
    return HealthResponse(
        status="healthy",
        service="SurfaceGuard AI Inference Backend",
        pytorch_version=dev_info["pytorch_version"],
        cuda_available=dev_info["cuda_available"],
        device_name=dev_info["device_name"],
    )


@router.get("/categories", response_model=CategoriesResponse)
def list_categories(model_type: str = "patchcore") -> CategoriesResponse:
    """Returns all 15 supported MVTec categories and indicates artifact availability."""
    available = artifact_mgr.list_available_categories(model_type=model_type)
    ready_count = sum(1 for v in available.values() if v)
    return CategoriesResponse(
        all_categories=ALL_CATEGORIES,
        available_categories=available,
        count_ready=ready_count,
        count_total=len(ALL_CATEGORIES),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Inspection image file (PNG, JPG, BMP)"),
    category: str = Form(..., description="MVTec category name (e.g. bottle, tile)"),
    model_type: str = Form("patchcore", description="Model method (patchcore, autoencoder, classical)"),
) -> PredictionResponse:
    """Performs visual anomaly detection and localization on the uploaded image."""
    cat = category.lower().strip()
    if cat not in ALL_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category '{category}'. Must be one of {ALL_CATEGORIES}",
        )

    # Validate file content type
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Must be PNG, JPEG, or BMP.",
        )

    # Read image contents
    try:
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file exceeds maximum allowable size (20 MB).",
            )
        img = Image.open(io.BytesIO(contents))
        img.verify()
        # Re-open after verify()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {str(e)}",
        )

    try:
        result = predictor.predict_image(image=img, category=cat, model_type=model_type)
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )
    except Exception as err:
        logger.error(f"Inference failure: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(err)}",
        )

    return PredictionResponse(**result)
