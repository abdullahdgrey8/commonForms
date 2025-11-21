# app/models/batch.py
from pydantic import BaseModel, Field
from typing import List, Optional


class BatchFileResult(BaseModel):
    """Result for a single file in batch processing"""
    filename: str = Field(description="Original filename")
    success: bool = Field(description="Whether processing succeeded")
    output_filename: Optional[str] = Field(None, description="Output filename if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(description="Processing time in seconds")
    request_id: Optional[str] = Field(None, description="Request ID for metrics")


class BatchProcessResponse(BaseModel):
    """Response for batch processing"""
    total_files: int = Field(description="Total number of files submitted")
    successful: int = Field(description="Number of successfully processed files")
    failed: int = Field(description="Number of failed files")
    total_time: float = Field(description="Total processing time in seconds")
    results: List[BatchFileResult] = Field(description="Individual file results")
    download_url: Optional[str] = Field(None, description="URL to download ZIP of all files")
    batch_id: str = Field(description="Batch processing ID")


class BatchStatusResponse(BaseModel):
    """Status of a batch processing job"""
    batch_id: str
    status: str  # "processing", "completed", "failed"
    total_files: int
    processed_files: int
    successful: int
    failed: int
    download_url: Optional[str] = None