from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="SurfaceGuard AI",
    description="Deep Learning-Based Industrial Defect Detection and Anomaly Localization API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit or web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router)  # Also mount at root for direct endpoints /health, /categories, /predict


@app.get("/")
def root():
    return {
        "project": "SurfaceGuard AI",
        "description": "Industrial Defect Detection and Anomaly Localization API",
        "documentation": "/docs",
        "health": "/health",
    }
