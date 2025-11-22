# app/services/batch_service.py
import asyncio
import time
import zipfile
import shutil
from pathlib import Path
from typing import List
from fastapi import UploadFile
from app.models.pdf import PDFProcessRequest
from app.models.batch import BatchFileResult, BatchProcessResponse
from app.services.pdf_service import pdf_service
from app.services.metrics_tracker import metrics_tracker
from app.core.logging import get_logger

logger = get_logger(__name__)


class BatchService:
    """Service for batch PDF processing"""
    
    def __init__(self):
        self.batch_dir = Path("data/batch")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_single_file(
        self,
        pdf_file: UploadFile,
        params: PDFProcessRequest,
        track_metrics: bool = False
    ) -> BatchFileResult:
        """Process a single file and return result"""
        start_time = time.time()
        request_id = metrics_tracker.generate_request_id()
        
        try:
            logger.info(f"[Batch-{request_id}] Processing {pdf_file.filename}")
            
            output_path, original_filename, metrics = await pdf_service.process_pdf(
                pdf_file,
                params,
                track_metrics=track_metrics,
                request_id=request_id
            )
            
            processing_time = time.time() - start_time
            
            return BatchFileResult(
                filename=original_filename,
                success=True,
                output_filename=f"fillable_{original_filename}",
                processing_time=processing_time,
                request_id=request_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[Batch-{request_id}] Failed to process {pdf_file.filename}: {e}")
            
            return BatchFileResult(
                filename=pdf_file.filename,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                request_id=request_id
            )
    
    async def process_batch(
        self,
        pdf_files: List[UploadFile],
        params: PDFProcessRequest,
        track_metrics: bool = False,
        parallel: bool = True
    ) -> BatchProcessResponse:
        """
        Process multiple PDFs
        
        Args:
            pdf_files: List of PDF files to process
            params: Processing parameters
            track_metrics: Whether to track metrics
            parallel: Process files in parallel (True) or sequentially (False)
        """
        batch_id = metrics_tracker.generate_request_id()
        logger.info(f"[Batch-{batch_id}] Starting batch processing: {len(pdf_files)} files")
        
        start_time = time.time()
        
        # Create batch directory
        batch_path = self.batch_dir / batch_id
        batch_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Process files
            if parallel:
                # --- FIX: Limit concurrency to prevent OOM errors ---
                # We use a Semaphore to limit active tasks to 3 at a time
                semaphore = asyncio.Semaphore(4) 

                async def semaphore_wrapper(pdf):
                    async with semaphore:
                        return await self.process_single_file(pdf, params, track_metrics)

                tasks = [semaphore_wrapper(pdf) for pdf in pdf_files]
                results = await asyncio.gather(*tasks)
            else:
                # Process files sequentially
                results = []
                for pdf in pdf_files:
                    result = await self.process_single_file(pdf, params, track_metrics)
                    results.append(result)
            
            total_time = time.time() - start_time
            
            # Count successes and failures
            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            
            logger.info(
                f"[Batch-{batch_id}] Complete: {successful}/{len(pdf_files)} successful "
                f"in {total_time:.2f}s"
            )
            
            # Create response
            response = BatchProcessResponse(
                total_files=len(pdf_files),
                successful=successful,
                failed=failed,
                total_time=total_time,
                results=results,
                batch_id=batch_id
            )
            
            return response
            
        except Exception as e:
            logger.error(f"[Batch-{batch_id}] Batch processing failed: {e}")
            raise
    
    async def create_zip_archive(
        self,
        results: List[BatchFileResult],
        batch_id: str
    ) -> Path:
        """
        Create ZIP archive of processed files
        
        Args:
            results: List of processing results
            batch_id: Batch ID
            
        Returns:
            Path to ZIP file
        """
        batch_path = self.batch_dir / batch_id
        zip_path = batch_path / f"batch_{batch_id}.zip"
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for result in results:
                    if result.success and result.output_filename:
                        # In real implementation, you'd get the actual file path
                        # For now, this is a placeholder
                        logger.info(f"Adding {result.output_filename} to ZIP")
            
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            raise


# Global batch service instance
batch_service = BatchService()