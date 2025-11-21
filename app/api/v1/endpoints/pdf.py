# # app/api/v1/endpoints/pdf.py
# from fastapi import APIRouter, Depends, File, UploadFile, Form, Response
# from fastapi.responses import FileResponse
# from app.api.deps import get_current_active_user
# from app.models.auth import User
# from app.models.pdf import PDFProcessRequest
# from app.services.pdf_service import pdf_service
# from app.services.metrics_tracker import metrics_tracker
# from app.core.logging import get_logger

# logger = get_logger(__name__)

# router = APIRouter()


# @router.post("/make-fillable", response_class=FileResponse)
# async def make_fillable(
#     response: Response,
#     pdf: UploadFile = File(..., description="Original PDF to make fillable"),
#     model: str = Form("FFDNet-L", description="Model: FFDNet-L or FFDNet-S"),
#     keep_existing: bool = Form(False, description="Keep existing form fields"),
#     use_signature_fields: bool = Form(False, description="Use signature fields for signatures"),
#     device: str = Form("cpu", description="Processing device (cpu, cuda:0, etc.)"),
#     confidence: float = Form(0.3, ge=0.0, le=1.0, description="Detection confidence threshold"),
#     fast: bool = Form(False, description="Use faster ONNX model"),
#     multiline: bool = Form(False, description="Allow multiline text boxes"),
#     track_metrics: bool = Form(False, description="Track DPI and GPU metrics"),
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     Upload a PDF and get back a fillable version with form fields
    
#     **Authentication Required:** Include Bearer token in Authorization header
    
#     **Parameters:**
#     - `pdf`: PDF file to process (max 50MB)
#     - `model`: Detection model (FFDNet-L for better accuracy, FFDNet-S for speed)
#     - `keep_existing`: Preserve any existing form fields
#     - `use_signature_fields`: Create signature-type fields for signatures
#     - `device`: Processing device (use "cuda:0" for GPU acceleration)
#     - `confidence`: Detection threshold (0.0-1.0, higher = fewer but more confident fields)
#     - `fast`: Use ONNX for faster CPU processing
#     - `multiline`: Enable multiline text boxes
#     - `track_metrics`: Enable DPI and GPU metrics tracking (adds processing overhead)
    
#     **Returns:** Processed PDF file with form fields added
    
#     **Metrics:** If track_metrics=true, performance data is logged and stored.
#     Check response headers for X-Request-ID to query metrics later.
    
#     **Example using curl:**
#     ```bash
#     curl -X POST "http://localhost:8000/api/v1/pdf/make-fillable" \\
#       -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
#       -F "pdf=@input.pdf" \\
#       -F "model=FFDNet-L" \\
#       -F "confidence=0.3" \\
#       -F "track_metrics=true" \\
#       -o output.pdf
#     ```
#     """
#     logger.info(
#         f"User {current_user.username} processing PDF: {pdf.filename}, "
#         f"track_metrics={track_metrics}"
#     )
    
#     # Create request params
#     params = PDFProcessRequest(
#         model=model,
#         keep_existing=keep_existing,
#         use_signature_fields=use_signature_fields,
#         device=device,
#         confidence=confidence,
#         fast=fast,
#         multiline=multiline
#     )
    
#     # Generate request ID
#     request_id = metrics_tracker.generate_request_id()
    
#     # Process PDF
#     output_path, original_filename, metrics = await pdf_service.process_pdf(
#         pdf, params, track_metrics=track_metrics, request_id=request_id
#     )
    
#     # Add metrics to response headers if tracked
#     if metrics:
#         response.headers["X-Request-ID"] = request_id
#         response.headers["X-Processing-Time"] = str(round(metrics.total_processing_time, 2))
        
#         if metrics.dpi_metrics:
#             response.headers["X-Input-DPI"] = str(round(metrics.dpi_metrics.input_dpi, 2))
#             response.headers["X-Output-DPI"] = str(round(metrics.dpi_metrics.output_dpi, 2))
        
#         if metrics.gpu_metrics and metrics.gpu_metrics.gpu_available:
#             response.headers["X-GPU-Peak-Memory-MB"] = str(
#                 round(metrics.gpu_metrics.peak_memory_mb, 2)
#             )
#             if metrics.gpu_metrics.memory_increase_mb:
#                 response.headers["X-GPU-Memory-Increase-MB"] = str(
#                     round(metrics.gpu_metrics.memory_increase_mb, 2)
#                 )
    
#     # Return processed file
#     return FileResponse(
#         output_path,
#         media_type="application/pdf",
#         filename=f"fillable_{original_filename}",
#         background=None  # Keep file until response is sent
#     )


# @router.get("/metrics/summary")
# async def get_metrics_summary(
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     Get summary of processing metrics
    
#     Returns aggregated statistics from stored metrics
#     """
#     logger.info(f"User {current_user.username} requesting metrics summary")
#     summary = metrics_tracker.get_metrics_summary()
#     return summary


# @router.get("/metrics/recent")
# async def get_recent_metrics(
#     limit: int = 50,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     Get recent processing metrics
    
#     **Parameters:**
#     - `limit`: Number of recent records to return (max 100)
#     """
#     logger.info(f"User {current_user.username} requesting recent metrics (limit={limit})")
    
#     if limit > 100:
#         limit = 100
    
#     metrics = metrics_tracker.get_recent_metrics(limit=limit)
#     return {
#         "count": len(metrics),
#         "metrics": [m.model_dump() for m in metrics]
#     }
# app/api/v1/endpoints/pdf.py
from fastapi import APIRouter, Depends, File, UploadFile, Form, Response
from fastapi.responses import FileResponse
from app.api.deps import get_current_active_user
from app.models.auth import User
from app.models.pdf import PDFProcessRequest
from app.services.pdf_service import pdf_service
from app.services.metrics_tracker import metrics_tracker
from app.utils.device_validator import device_validator
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/devices")
async def get_available_devices(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get information about available compute devices
    
    Returns details about CPU and GPU availability
    """
    info = device_validator.get_device_info()
    info["recommended_device"] = device_validator.recommend_device()
    return info


@router.post("/make-fillable", response_class=FileResponse)
async def make_fillable(
    response: Response,
    pdf: UploadFile = File(..., description="Original PDF to make fillable"),
    model: str = Form("FFDNet-L", description="Model: FFDNet-L or FFDNet-S"),
    keep_existing: bool = Form(False, description="Keep existing form fields"),
    use_signature_fields: bool = Form(False, description="Use signature fields for signatures"),
    device: str = Form("cpu", description="Processing device (cpu, cuda:0, etc.)"),
    confidence: float = Form(0.3, ge=0.0, le=1.0, description="Detection confidence threshold"),
    fast: bool = Form(False, description="Use faster ONNX model"),
    multiline: bool = Form(False, description="Allow multiline text boxes"),
    track_metrics: bool = Form(False, description="Track DPI and GPU metrics"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload a PDF and get back a fillable version with form fields
    
    **Authentication Required:** Include Bearer token in Authorization header
    
    **Parameters:**
    - `pdf`: PDF file to process (max 50MB)
    - `model`: Detection model (FFDNet-L for better accuracy, FFDNet-S for speed)
    - `keep_existing`: Preserve any existing form fields
    - `use_signature_fields`: Create signature-type fields for signatures
    - `device`: Processing device (use "cuda:0" for GPU acceleration)
    - `confidence`: Detection threshold (0.0-1.0, higher = fewer but more confident fields)
    - `fast`: Use ONNX for faster CPU processing
    - `multiline`: Enable multiline text boxes
    - `track_metrics`: Enable DPI and GPU metrics tracking (adds processing overhead)
    
    **Returns:** Processed PDF file with form fields added
    
    **Metrics:** If track_metrics=true, performance data is logged and stored.
    Check response headers for X-Request-ID to query metrics later.
    
    **Example using curl:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/pdf/make-fillable" \\
      -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
      -F "pdf=@input.pdf" \\
      -F "model=FFDNet-L" \\
      -F "confidence=0.3" \\
      -F "track_metrics=true" \\
      -o output.pdf
    ```
    """
    logger.info(
        f"User {current_user.username} processing PDF: {pdf.filename}, "
        f"requested_device={device}, track_metrics={track_metrics}"
    )
    
    # Validate device and potentially fall back to CPU
    is_valid, validated_device, device_message = device_validator.validate_device(device)
    if not is_valid:
        logger.warning(f"Device validation failed: {device_message}")
        device = validated_device  # Use fallback device
    else:
        logger.info(f"Device validation passed: {device_message}")
    
    logger.info(f"Final device to be used: {device}")
    
    # Create request params with validated device
    params = PDFProcessRequest(
        model=model,
        keep_existing=keep_existing,
        use_signature_fields=use_signature_fields,
        device=device,  # Use validated device
        confidence=confidence,
        fast=fast,
        multiline=multiline
    )
    
    # Generate request ID
    request_id = metrics_tracker.generate_request_id()
    
    # Process PDF
    output_path, original_filename, metrics = await pdf_service.process_pdf(
        pdf, params, track_metrics=track_metrics, request_id=request_id
    )
    
    # Add metrics to response headers if tracked
    if metrics:
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(round(metrics.total_processing_time, 2))
        response.headers["X-Device-Used"] = metrics.device
        
        if not is_valid:
            response.headers["X-Device-Warning"] = device_message
        
        if metrics.dpi_metrics:
            response.headers["X-Input-DPI"] = str(round(metrics.dpi_metrics.input_dpi, 2))
            response.headers["X-Output-DPI"] = str(round(metrics.dpi_metrics.output_dpi, 2))
        
        if metrics.gpu_metrics and metrics.gpu_metrics.gpu_available:
            if metrics.gpu_metrics.peak_memory_mb is not None:
                response.headers["X-GPU-Peak-Memory-MB"] = str(
                    round(metrics.gpu_metrics.peak_memory_mb, 2)
                )
            if metrics.gpu_metrics.memory_increase_mb is not None:
                response.headers["X-GPU-Memory-Increase-MB"] = str(
                    round(metrics.gpu_metrics.memory_increase_mb, 2)
                )
    
    # Return processed file
    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"fillable_{original_filename}",
        background=None  # Keep file until response is sent
    )


@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get summary of processing metrics
    
    Returns aggregated statistics from stored metrics
    """
    logger.info(f"User {current_user.username} requesting metrics summary")
    summary = metrics_tracker.get_metrics_summary()
    return summary


@router.get("/metrics/recent")
async def get_recent_metrics(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get recent processing metrics
    
    **Parameters:**
    - `limit`: Number of recent records to return (max 100)
    """
    logger.info(f"User {current_user.username} requesting recent metrics (limit={limit})")
    
    if limit > 100:
        limit = 100
    
    metrics = metrics_tracker.get_recent_metrics(limit=limit)
    return {
        "count": len(metrics),
        "metrics": [m.model_dump() for m in metrics]
    }