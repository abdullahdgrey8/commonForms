# app/api/v1/endpoints/batch.py
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from app.api.deps import get_current_active_user
from app.models.auth import User
from app.models.pdf import PDFProcessRequest
from app.models.batch import BatchProcessResponse
from app.services.batch_services import batch_service
from app.utils.device_validator import device_validator
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/make-fillable-batch", response_model=BatchProcessResponse)
async def make_fillable_batch(
    pdfs: List[UploadFile] = File(..., description="Multiple PDF files to process"),
    model: str = Form("FFDNet-L", description="Model: FFDNet-L or FFDNet-S"),
    keep_existing: bool = Form(False, description="Keep existing form fields"),
    use_signature_fields: bool = Form(False, description="Use signature fields for signatures"),
    device: str = Form("cpu", description="Processing device (cpu, cuda:0, etc.)"),
    confidence: float = Form(0.3, ge=0.0, le=1.0, description="Detection confidence threshold"),
    fast: bool = Form(False, description="Use faster ONNX model"),
    multiline: bool = Form(False, description="Allow multiline text boxes"),
    track_metrics: bool = Form(False, description="Track DPI and GPU metrics"),
    parallel: bool = Form(True, description="Process files in parallel"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload multiple PDFs and get back fillable versions with form fields
    
    **Authentication Required:** Include Bearer token in Authorization header
    
    **Parameters:**
    - `pdfs`: Multiple PDF files to process (max 10 files, 50MB each)
    - `model`: Detection model (FFDNet-L or FFDNet-S)
    - `keep_existing`: Preserve any existing form fields
    - `use_signature_fields`: Create signature-type fields for signatures
    - `device`: Processing device (use "cuda:0" for GPU)
    - `confidence`: Detection threshold (0.0-1.0)
    - `fast`: Use ONNX for faster CPU processing
    - `multiline`: Enable multiline text boxes
    - `track_metrics`: Enable metrics tracking for all files
    - `parallel`: Process files concurrently (faster) or sequentially (safer)
    
    **Returns:** JSON with results for each file and overall statistics
    
    **Example using curl:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/pdf/make-fillable-batch" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -F "pdfs=@file1.pdf" \\
      -F "pdfs=@file2.pdf" \\
      -F "pdfs=@file3.pdf" \\
      -F "model=FFDNet-L" \\
      -F "parallel=true"
    ```
    
    **Example using Python:**
    ```python
    files = [
        ('pdfs', open('file1.pdf', 'rb')),
        ('pdfs', open('file2.pdf', 'rb')),
        ('pdfs', open('file3.pdf', 'rb'))
    ]
    data = {'model': 'FFDNet-L', 'parallel': True}
    response = requests.post(url, files=files, data=data, headers=headers)
    ```
    """
    logger.info(
        f"User {current_user.username} batch processing: {len(pdfs)} files, "
        f"parallel={parallel}, track_metrics={track_metrics}"
    )
    
    # Validate file count
    if len(pdfs) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch. Use multiple batches for more files."
        )
    
    # Validate all files are PDFs
    for pdf in pdfs:
        if not pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File {pdf.filename} is not a PDF"
            )
    
    # Validate device
    is_valid, validated_device, device_message = device_validator.validate_device(device)
    if not is_valid:
        logger.warning(f"Device validation failed: {device_message}")
        device = validated_device
    else:
        logger.info(f"Device validation passed: {device_message}")
    
    # Create processing parameters
    params = PDFProcessRequest(
        model=model,
        keep_existing=keep_existing,
        use_signature_fields=use_signature_fields,
        device=device,
        confidence=confidence,
        fast=fast,
        multiline=multiline
    )
    
    # Process batch
    try:
        result = await batch_service.process_batch(
            pdfs,
            params,
            track_metrics=track_metrics,
            parallel=parallel
        )
        
        logger.info(
            f"Batch {result.batch_id}: {result.successful}/{result.total_files} successful "
            f"in {result.total_time:.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/batch-info")
async def get_batch_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get information about batch processing capabilities
    
    Returns limits and recommendations for batch processing
    """
    return {
        "max_files_per_batch": 10,
        "max_file_size_mb": 50,
        "supported_formats": ["pdf"],
        "parallel_processing_available": True,
        "recommendations": {
            "small_files": "Use parallel=true for files < 5 pages",
            "large_files": "Use parallel=false for files > 20 pages to avoid memory issues",
            "gpu_usage": "With GPU, recommend 2-3 files max in parallel to avoid VRAM limits",
            "cpu_usage": "With CPU, can process 5-10 files in parallel depending on cores"
        }
    }