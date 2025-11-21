# app/api/v1/endpoints/performance.py
import time
import tempfile
from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, Depends, File, UploadFile, Body
from pydantic import BaseModel, Field
from app.api.deps import get_current_active_user
from app.models.auth import User
from app.models.pdf import PDFProcessRequest
from app.services.pdf_service import pdf_service
from app.services.metrics_tracker import metrics_tracker
from app.core.logging import get_logger
import PyPDF2
import fitz  # PyMuPDF

logger = get_logger(__name__)
router = APIRouter()


class PageSizeResult(BaseModel):
    """Result for a single page size test"""
    page_size: str = Field(description="Page size name (e.g., A4, Letter)")
    width_pts: float = Field(description="Width in points")
    height_pts: float = Field(description="Height in points")
    width_inches: float = Field(description="Width in inches")
    height_inches: float = Field(description="Height in inches")
    processing_time: float = Field(description="Processing time in seconds")
    success: bool = Field(description="Whether processing succeeded")
    error_message: str | None = Field(None, description="Error message if failed")
    request_id: str = Field(description="Request ID for this test")
    gpu_peak_memory_mb: float | None = Field(None, description="Peak GPU memory if tracked")


class PageSizeTestResponse(BaseModel):
    """Response for page size testing"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_time: float
    results: List[PageSizeResult]
    fastest_size: str | None = None
    slowest_size: str | None = None


class LoadTestResult(BaseModel):
    """Result for load testing"""
    request_number: int
    processing_time: float
    success: bool
    error_message: str | None = None
    request_id: str
    gpu_peak_memory_mb: float | None = None


class LoadTestResponse(BaseModel):
    """Response for load testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    average_processing_time: float
    min_processing_time: float
    max_processing_time: float
    requests_per_second: float
    results: List[LoadTestResult]


# Standard page sizes in points (1 inch = 72 points)
PAGE_SIZES = {
    "A4": (595, 842),  # 8.27 x 11.69 inches
    "Letter": (612, 792),  # 8.5 x 11 inches
    "Legal": (612, 1008),  # 8.5 x 14 inches
    "A3": (842, 1191),  # 11.69 x 16.54 inches
    "A5": (420, 595),  # 5.83 x 8.27 inches
    "Tabloid": (792, 1224),  # 11 x 17 inches
}


def create_test_pdf(page_size: str, output_path: str) -> None:
    """Create a test PDF with specified page size"""
    width, height = PAGE_SIZES[page_size]
    
    # Create PDF using PyMuPDF
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    
    # Add some sample content
    text = f"Test PDF - {page_size} ({width}x{height} points)"
    page.insert_text((50, 50), text, fontsize=12)
    
    # Add some form-like elements (boxes that look like form fields)
    for i in range(5):
        y_pos = 100 + (i * 60)
        # Draw rectangle
        rect = fitz.Rect(50, y_pos, 300, y_pos + 30)
        page.draw_rect(rect, color=(0, 0, 0), width=1)
        # Add label
        page.insert_text((50, y_pos - 10), f"Field {i+1}:", fontsize=10)
    
    doc.save(output_path)
    doc.close()


@router.post("/test-page-sizes", response_model=PageSizeTestResponse)
async def test_page_sizes(
    page_sizes: List[str] = Body(
        default=["A4", "Letter", "Legal"],
        description="List of page sizes to test"
    ),
    model: str = Body("FFDNet-L", description="Model to use"),
    device: str = Body("cpu", description="Device (cpu or cuda:0)"),
    confidence: float = Body(0.3, ge=0.0, le=1.0),
    track_gpu: bool = Body(True, description="Track GPU memory usage"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Test processing performance across different PDF page sizes
    
    **Purpose:** Measure how page size affects processing time and memory usage
    
    **Parameters:**
    - `page_sizes`: List of page sizes to test (A4, Letter, Legal, A3, A5, Tabloid)
    - `model`: Model to use for all tests
    - `device`: Processing device
    - `confidence`: Detection confidence
    - `track_gpu`: Track GPU memory (only works with CUDA devices)
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/performance/test-page-sizes" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{
        "page_sizes": ["A4", "Letter", "Legal", "A3"],
        "model": "FFDNet-L",
        "device": "cuda:0",
        "track_gpu": true
      }'
    ```
    """
    logger.info(
        f"User {current_user.username} starting page size test: "
        f"{page_sizes}, device={device}"
    )
    
    # Validate page sizes
    valid_sizes = []
    for size in page_sizes:
        if size in PAGE_SIZES:
            valid_sizes.append(size)
        else:
            logger.warning(f"Unknown page size: {size}, skipping")
    
    if not valid_sizes:
        valid_sizes = ["A4", "Letter"]
        logger.info(f"No valid page sizes provided, using defaults: {valid_sizes}")
    
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
    
    results = []
    total_start = time.time()
    
    # Test each page size
    for page_size in valid_sizes:
        logger.info(f"Testing page size: {page_size}")
        request_id = f"pagesize_{page_size}_{metrics_tracker.generate_request_id()}"
        
        tmp_input = None
        try:
            # Create test PDF
            tmp_input = tempfile.mktemp(suffix=".pdf")
            create_test_pdf(page_size, tmp_input)
            
            # Create UploadFile-like object
            class FakeUploadFile:
                def __init__(self, path, name):
                    self.filename = name
                    self.file = open(path, 'rb')
            
            fake_upload = FakeUploadFile(tmp_input, f"test_{page_size}.pdf")
            
            start_time = time.time()
            
            # Process PDF
            output_path, original_filename, metrics = await pdf_service.process_pdf(
                fake_upload,
                params,
                track_metrics=track_gpu,
                request_id=request_id
            )
            
            processing_time = time.time() - start_time
            
            # Get dimensions
            width, height = PAGE_SIZES[page_size]
            
            # Extract GPU memory if available
            gpu_memory = None
            if metrics and metrics.gpu_metrics and metrics.gpu_metrics.peak_memory_mb:
                gpu_memory = metrics.gpu_metrics.peak_memory_mb
            
            result = PageSizeResult(
                page_size=page_size,
                width_pts=width,
                height_pts=height,
                width_inches=round(width / 72, 2),
                height_inches=round(height / 72, 2),
                processing_time=round(processing_time, 3),
                success=True,
                request_id=request_id,
                gpu_peak_memory_mb=gpu_memory
            )
            
            results.append(result)
            
            # Cleanup
            fake_upload.file.close()
            if output_path:
                Path(output_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to test page size {page_size}: {e}")
            width, height = PAGE_SIZES[page_size]
            results.append(PageSizeResult(
                page_size=page_size,
                width_pts=width,
                height_pts=height,
                width_inches=round(width / 72, 2),
                height_inches=round(height / 72, 2),
                processing_time=0,
                success=False,
                error_message=str(e),
                request_id=request_id
            ))
        finally:
            if tmp_input:
                Path(tmp_input).unlink(missing_ok=True)
    
    total_time = time.time() - total_start
    
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    fastest = None
    slowest = None
    if successful:
        fastest = min(successful, key=lambda x: x.processing_time)
        slowest = max(successful, key=lambda x: x.processing_time)
    
    response = PageSizeTestResponse(
        total_tests=len(results),
        successful_tests=len(successful),
        failed_tests=len(failed),
        total_time=round(total_time, 3),
        results=results,
        fastest_size=fastest.page_size if fastest else None,
        slowest_size=slowest.page_size if slowest else None
    )
    
    logger.info(
        f"Page size test complete: {len(successful)}/{len(results)} successful, "
        f"fastest: {response.fastest_size}, slowest: {response.slowest_size}"
    )
    
    return response


@router.post("/load-test", response_model=LoadTestResponse)
async def run_load_test(
    pdf: UploadFile = File(..., description="PDF file to use for testing"),
    num_requests: int = Body(10, ge=1, le=100, description="Number of requests to send"),
    concurrent: bool = Body(False, description="Send requests concurrently or sequentially"),
    model: str = Body("FFDNet-L"),
    device: str = Body("cpu"),
    confidence: float = Body(0.3, ge=0.0, le=1.0),
    track_gpu: bool = Body(True),
    current_user: User = Depends(get_current_active_user)
):
    """
    Run load test with multiple synchronous or asynchronous requests
    
    **Purpose:** Test system performance under load
    
    **Parameters:**
    - `pdf`: Test PDF file
    - `num_requests`: Number of requests to send (1-100)
    - `concurrent`: If true, sends all requests at once; if false, sends sequentially
    - `model`: Model to use
    - `device`: Processing device
    - `confidence`: Detection confidence
    - `track_gpu`: Track GPU memory usage
    
    **Sequential vs Concurrent:**
    - Sequential (concurrent=false): Measures single-request performance, one after another
    - Concurrent (concurrent=true): Measures multi-request handling capacity
    
    **Example:**
    ```bash
    # Sequential test (synchronous)
    curl -X POST "http://localhost:8000/api/v1/performance/load-test" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -F "pdf=@test.pdf" \\
      -F "num_requests=20" \\
      -F "concurrent=false" \\
      -F "device=cuda:0"
    
    # Concurrent test (async)
    curl -X POST "http://localhost:8000/api/v1/performance/load-test" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -F "pdf=@test.pdf" \\
      -F "num_requests=10" \\
      -F "concurrent=true" \\
      -F "device=cuda:0"
    ```
    """
    logger.info(
        f"User {current_user.username} starting load test: "
        f"{num_requests} requests, concurrent={concurrent}"
    )
    
    # Read PDF data
    pdf_data = await pdf.read()
    await pdf.seek(0)
    
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
    
    async def process_request(request_num: int) -> LoadTestResult:
        """Process a single request"""
        request_id = f"load_{request_num}_{metrics_tracker.generate_request_id()}"
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_data)
                tmp_path = tmp.name
            
            class FakeUploadFile:
                def __init__(self, path, name):
                    self.filename = name
                    self.file = open(path, 'rb')
            
            fake_upload = FakeUploadFile(tmp_path, pdf.filename)
            
            start_time = time.time()
            
            output_path, original_filename, metrics = await pdf_service.process_pdf(
                fake_upload,
                params,
                track_metrics=track_gpu,
                request_id=request_id
            )
            
            processing_time = time.time() - start_time
            
            # Get GPU memory
            gpu_memory = None
            if metrics and metrics.gpu_metrics and metrics.gpu_metrics.peak_memory_mb:
                gpu_memory = metrics.gpu_metrics.peak_memory_mb
            
            # Cleanup
            fake_upload.file.close()
            Path(tmp_path).unlink(missing_ok=True)
            if output_path:
                Path(output_path).unlink(missing_ok=True)
            
            return LoadTestResult(
                request_number=request_num,
                processing_time=round(processing_time, 3),
                success=True,
                request_id=request_id,
                gpu_peak_memory_mb=gpu_memory
            )
            
        except Exception as e:
            logger.error(f"Load test request {request_num} failed: {e}")
            return LoadTestResult(
                request_number=request_num,
                processing_time=0,
                success=False,
                error_message=str(e),
                request_id=request_id
            )
    
    # Run requests
    total_start = time.time()
    
    if concurrent:
        # Run all requests concurrently
        import asyncio
        tasks = [process_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    else:
        # Run sequentially
        results = []
        for i in range(num_requests):
            result = await process_request(i)
            results.append(result)
    
    total_time = time.time() - total_start
    
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    processing_times = [r.processing_time for r in successful]
    
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    requests_per_second = num_requests / total_time if total_time > 0 else 0
    
    response = LoadTestResponse(
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_time=round(total_time, 3),
        average_processing_time=round(avg_time, 3),
        min_processing_time=round(min_time, 3),
        max_processing_time=round(max_time, 3),
        requests_per_second=round(requests_per_second, 3),
        results=results
    )
    
    logger.info(
        f"Load test complete: {len(successful)}/{num_requests} successful, "
        f"avg time: {avg_time:.2f}s, throughput: {requests_per_second:.2f} req/s"
    )
    
    return response