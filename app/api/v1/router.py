# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import auth, health, pdf, benchmark, batch, performance

# Create API v1 router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health.router,
    tags=["Health"]
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    pdf.router,
    prefix="/pdf",
    tags=["PDF Processing"]
)

api_router.include_router(
    benchmark.router,
    prefix="/pdf",
    tags=["Benchmarking"]
)

api_router.include_router(
    batch.router,
    prefix="/pdf",
    tags=["Batch Processing"]
)

api_router.include_router(
    performance.router,
    prefix="/performance",
    tags=["Performance Testing"]
)