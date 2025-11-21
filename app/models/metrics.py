# app/models/metrics.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DPIMetrics(BaseModel):
    """DPI-related metrics"""
    input_dpi: float = Field(description="Input PDF DPI")
    output_dpi: float = Field(description="Output PDF DPI")
    processing_time_seconds: float = Field(description="Total processing time")
    

class GPUMetrics(BaseModel):
    """GPU memory metrics"""
    gpu_available: bool = Field(description="Whether GPU was available")
    device_name: Optional[str] = Field(None, description="GPU device name")
    peak_memory_mb: Optional[float] = Field(None, description="Peak GPU memory in MB")
    initial_memory_mb: Optional[float] = Field(None, description="Initial GPU memory in MB")
    memory_increase_mb: Optional[float] = Field(None, description="Memory increase during processing")


class ProcessingMetrics(BaseModel):
    """Complete processing metrics"""
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filename: str = Field(description="Original filename")
    file_size_bytes: int = Field(description="Input file size")
    model_used: str = Field(description="Model name used")
    device: str = Field(description="Device used (cpu/cuda)")
    
    # DPI metrics
    dpi_metrics: Optional[DPIMetrics] = None
    
    # GPU metrics
    gpu_metrics: Optional[GPUMetrics] = None
    
    # Timing
    total_processing_time: float = Field(description="Total processing time in seconds")
    
    # Worker info
    concurrent_requests: int = Field(default=1, description="Number of concurrent requests")
    
    # Success status
    success: bool = Field(default=True)
    error_message: Optional[str] = None


class BenchmarkRequest(BaseModel):
    """Benchmark testing request"""
    num_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of concurrent requests to test"
    )
    test_file_path: Optional[str] = Field(
        None,
        description="Path to test PDF (if not provided, will use uploaded file)"
    )
    model: str = Field(default="FFDNet-L")
    device: str = Field(default="cpu")
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    """Benchmark test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time: float
    min_processing_time: float
    max_processing_time: float
    total_time: float
    requests_per_second: float
    metrics: list[ProcessingMetrics]