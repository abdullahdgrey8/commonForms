# app/api/v1/endpoints/benchmark.py
import asyncio
import time
from fastapi import APIRouter, Depends, File, UploadFile, Body
from app.api.deps import get_current_active_user
from app.models.auth import User
from app.models.pdf import PDFProcessRequest
from app.models.metrics import BenchmarkRequest, BenchmarkResult
from app.services.pdf_service import pdf_service
from app.services.metrics_tracker import metrics_tracker
from app.core.logging import get_logger
import tempfile
import shutil

logger = get_logger(__name__)

router = APIRouter()


async def process_single_request(
    pdf_data: bytes,
    filename: str,
    params: PDFProcessRequest,
    request_num: int
) -> dict:
    """Process a single PDF request for benchmarking"""
    try:
        # Create temporary file from bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_data)
            tmp_path = tmp.name
        
        # Create UploadFile-like object
        class FakeUploadFile:
            def __init__(self, path, name):
                self.filename = name
                self.file = open(path, 'rb')
        
        fake_upload = FakeUploadFile(tmp_path, filename)
        request_id = f"bench_{request_num}_{metrics_tracker.generate_request_id()}"
        
        start_time = time.time()
        
        # Process PDF
        output_path, original_filename, metrics = await pdf_service.process_pdf(
            fake_upload,
            params,
            track_metrics=True,
            request_id=request_id
        )
        
        processing_time = time.time() - start_time
        
        # Cleanup
        fake_upload.file.close()
        import os
        os.unlink(tmp_path)
        if output_path:
            os.unlink(output_path)
        
        return {
            "success": True,
            "processing_time": processing_time,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Benchmark request {request_num} failed: {e}")
        return {
            "success": False,
            "processing_time": 0,
            "error": str(e)
        }


@router.post("/benchmark", response_model=BenchmarkResult)
async def run_benchmark(
    pdf: UploadFile = File(..., description="PDF file to use for benchmarking"),
    num_concurrent: int = Body(5, ge=1, le=50, embed=True),
    model: str = Body("FFDNet-L", embed=True),
    device: str = Body("cpu", embed=True),
    confidence: float = Body(0.3, ge=0.0, le=1.0, embed=True),
    current_user: User = Depends(get_current_active_user)
):
    """
    Run benchmark test with concurrent PDF processing
    
    **Purpose:** Test system performance under load with multiple concurrent requests
    
    **Parameters:**
    - `pdf`: Test PDF file (will be processed multiple times concurrently)
    - `num_concurrent`: Number of concurrent requests (1-50)
    - `model`: Model to use for all requests
    - `device`: Processing device (cpu or cuda:0)
    - `confidence`: Detection confidence threshold
    
    **Returns:** Benchmark results with timing statistics
    
    **Warning:** This endpoint is resource-intensive. Use carefully in production.
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/pdf/benchmark" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -F "pdf=@test.pdf" \\
      -F "num_concurrent=10" \\
      -F "model=FFDNet-L" \\
      -F "device=cpu"
    ```
    """
    logger.info(
        f"User {current_user.username} starting benchmark: "
        f"{num_concurrent} concurrent requests"
    )
    
    # Read PDF data into memory
    pdf_data = await pdf.read()
    await pdf.seek(0)  # Reset file pointer
    
    # Create processing parameters
    params = PDFProcessRequest(
        model=model,
        keep_existing=False,
        use_signature_fields=False,
        device=device,
        confidence=confidence,
        fast=False,
        multiline=False
    )
    
    # Run concurrent requests
    total_start = time.time()
    
    tasks = [
        process_single_request(pdf_data, pdf.filename, params, i)
        for i in range(num_concurrent)
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - total_start
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    processing_times = [r["processing_time"] for r in successful]
    
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    
    requests_per_second = num_concurrent / total_time if total_time > 0 else 0
    
    # Extract metrics objects
    metrics_list = [r["metrics"] for r in successful if r.get("metrics")]
    
    benchmark_result = BenchmarkResult(
        total_requests=num_concurrent,
        successful_requests=len(successful),
        failed_requests=len(failed),
        average_processing_time=round(avg_time, 3),
        min_processing_time=round(min_time, 3),
        max_processing_time=round(max_time, 3),
        total_time=round(total_time, 3),
        requests_per_second=round(requests_per_second, 3),
        metrics=metrics_list
    )
    
    logger.info(
        f"Benchmark complete: {len(successful)}/{num_concurrent} successful, "
        f"avg time: {avg_time:.2f}s, throughput: {requests_per_second:.2f} req/s"
    )
    
    return benchmark_result


@router.post("/benchmark/sequential")
async def run_sequential_benchmark(
    pdf: UploadFile = File(..., description="PDF file to use for benchmarking"),
    num_requests: int = Body(5, ge=1, le=20, embed=True),
    model: str = Body("FFDNet-L", embed=True),
    device: str = Body("cpu", embed=True),
    confidence: float = Body(0.3, ge=0.0, le=1.0, embed=True),
    current_user: User = Depends(get_current_active_user)
):
    """
    Run sequential benchmark test (one request at a time)
    
    Useful for baseline performance measurement without concurrency overhead.
    
    **Parameters:**
    - `pdf`: Test PDF file
    - `num_requests`: Number of sequential requests (1-20)
    - `model`: Model to use
    - `device`: Processing device
    - `confidence`: Detection confidence
    
    **Returns:** Benchmark results
    """
    logger.info(
        f"User {current_user.username} starting sequential benchmark: "
        f"{num_requests} requests"
    )
    
    # Read PDF data
    pdf_data = await pdf.read()
    
    # Create parameters
    params = PDFProcessRequest(
        model=model,
        keep_existing=False,
        use_signature_fields=False,
        device=device,
        confidence=confidence,
        fast=False,
        multiline=False
    )
    
    # Run sequential requests
    total_start = time.time()
    results = []
    
    for i in range(num_requests):
        result = await process_single_request(pdf_data, pdf.filename, params, i)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    processing_times = [r["processing_time"] for r in successful]
    
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    
    requests_per_second = num_requests / total_time if total_time > 0 else 0
    
    metrics_list = [r["metrics"] for r in successful if r.get("metrics")]
    
    return BenchmarkResult(
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        average_processing_time=round(avg_time, 3),
        min_processing_time=round(min_time, 3),
        max_processing_time=round(max_time, 3),
        total_time=round(total_time, 3),
        requests_per_second=round(requests_per_second, 3),
        metrics=metrics_list
    )