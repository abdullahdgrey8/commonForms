# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.router import api_router
from app.services.model_cache import model_cache
from app.middleware.error_handler import add_error_handlers
import os
import torch

# 1. Silence pypdfium2 garbage collection AssertionError (super common, 100% safe)
warnings.filterwarnings("ignore", category=UserWarning, module="pypdfium2")
logging.getLogger("pypdfium2").setLevel(logging.CRITICAL)

# 2. Silence Ultralytics/YOLO NMS time limit warnings on dense PDFs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
os.environ["YOLO_NMS_TIME_LIMIT"] = "30.0"  # Give it plenty of breathing room

# 3. Silence PyTorch + CUDA harmless warnings (memory, deprecation, etc.)
logging.getLogger("torch").setLevel(logging.CRITICAL)
os.environ["TORCH_LOGS"] = "none"  # PyTorch 2.4+ feature

# 4. Optional: Silence other common noisy libs (if you use them)
logging.getLogger("PIL").setLevel(logging.WARNING)        # Pillow warnings
logging.getLogger("passlib").setLevel(logging.WARNING)   # Password hashing
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# 5. Expandable segments = better CUDA memory reuse (helps avoid OOM on edge cases)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    # os.environ["ORT_DISABLE_GPU"] = "1"
    # os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["YOLO_NO_AUTO_UPDATE"] = "1"
    # logger.info("ONNX Runtime locked to CPUExecutionProvider - no more Invalid device id!")
    model_cache.setup_cache_env()
    if settings.PRELOAD_MODELS:
        model_cache.preload_models()

    logger.info("Startup complete")
    yield
    logger.info("Shutting down")

# Initialize app
app = FastAPI(
    title=settings.APP_NAME,
    description="Make any PDF fillable using CommonForms AI",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup logging first
setup_logging()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handlers
add_error_handlers(app)

# API Router
app.include_router(api_router, prefix="/api/v1")

# Root
@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api": "/api/v1"
    }