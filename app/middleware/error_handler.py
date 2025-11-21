# app/middleware/error_handler.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import get_logger

logger = get_logger(__name__)

def add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(f"HTTP {exc.status_code}: {exc.detail} | {request.method} {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = [f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()]
        logger.warning(f"Validation error: {errors} | {request.method} {request.url}")
        return JSONResponse(
            status_code=422,
            content={"error": "Validation Error", "detail": errors}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error: {exc} | {request.method} {request.url}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"}
        )

    logger.info("Global error handlers registered.")