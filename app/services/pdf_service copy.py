# app/services/pdf_service.py
import tempfile
import shutil
import time
import os
from pathlib import Path
from fastapi import UploadFile, HTTPException
import commonforms
import torch
from app.core.logging import get_logger
from app.models.pdf import PDFProcessRequest
from app.models.metrics import ProcessingMetrics, DPIMetrics
from app.services.model_cache import model_cache
from app.services.metrics_tracker import metrics_tracker
from app.utils.dpi_analyzer import dpi_analyzer
from app.utils.gpu_monitor import create_gpu_monitor

logger = get_logger(__name__)


class PDFService:
    """Service for processing PDF files"""
    
    @staticmethod
    def _setup_cuda_environment(device: str):
        """
        Setup CUDA environment variables before processing
        
        This is critical because commonforms may run in a subprocess
        and needs to see CUDA devices
        """
        if device.startswith("cuda"):
            # Ensure CUDA is visible
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                current = os.environ['CUDA_VISIBLE_DEVICES']
                if current == '-1':
                    logger.warning("CUDA_VISIBLE_DEVICES was set to -1, removing restriction")
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Extract device number
            if ":" in device:
                device_num = device.split(":")[1]
            else:
                device_num = "0"
            
            # Set CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = device_num
            
            # Verify CUDA is available
            if torch.cuda.is_available():
                logger.info(
                    f"CUDA setup successful: device={device}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                    f"torch.cuda.device_count()={torch.cuda.device_count()}"
                )
            else:
                logger.error(
                    f"CUDA setup failed: torch.cuda.is_available()=False, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
        else:
            # For CPU, ensure no CUDA restrictions
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                logger.info("Using CPU, removing CUDA restrictions")
                del os.environ['CUDA_VISIBLE_DEVICES']
    
    @staticmethod
    async def process_pdf(
        pdf_file: UploadFile,
        params: PDFProcessRequest,
        track_metrics: bool = False,
        request_id: str = None
    ) -> tuple[str, str, ProcessingMetrics | None]:
        """
        Process a PDF file to make it fillable
        
        Args:
            pdf_file: Uploaded PDF file
            params: Processing parameters
            track_metrics: Whether to track and store metrics
            request_id: Optional request ID for tracking
            
        Returns:
            Tuple of (output_path, original_filename, metrics)
            
        Raises:
            HTTPException: If processing fails
        """
        if request_id is None:
            request_id = metrics_tracker.generate_request_id()
        
        # Setup CUDA environment BEFORE any processing
        PDFService._setup_cuda_environment(params.device)
        
        logger.info(
            f"[{request_id}] Processing PDF: {pdf_file.filename} "
            f"with model: {params.model}, device: {params.device}, metrics: {track_metrics}"
        )
        
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF"
            )
        
        # Initialize metrics tracking
        metrics = None
        gpu_monitor = None
        start_time = time.time()
        
        if track_metrics:
            gpu_monitor = create_gpu_monitor()
            gpu_monitor.start_monitoring(params.device)
        
        # Create temporary files
        suffix = ".pdf"
        tmp_in_path = None
        tmp_out_path = None
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                shutil.copyfileobj(pdf_file.file, tmp_in)
                tmp_in_path = tmp_in.name
            
            # Get file size
            file_size = Path(tmp_in_path).stat().st_size
            
            # Analyze input DPI if tracking metrics
            input_dpi = None
            if track_metrics:
                logger.info(f"[{request_id}] Analyzing input PDF DPI...")
                input_dpi = dpi_analyzer.get_pdf_dpi(tmp_in_path)
                logger.info(f"[{request_id}] Input DPI: {input_dpi}")
            
            # Create output temporary file path
            tmp_out_path = tempfile.mktemp(suffix=suffix)
            
            # Get model path from cache
            model_path = model_cache.get_model_path(params.model)
            
            # Log CUDA status before processing
            logger.info(
                f"[{request_id}] Starting commonforms processing with device={params.device}"
            )
            if params.device.startswith("cuda"):
                logger.info(
                    f"[{request_id}] PyTorch CUDA status: "
                    f"available={torch.cuda.is_available()}, "
                    f"device_count={torch.cuda.device_count()}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
                )
            
            processing_start = time.time()
            
            # Process the PDF
            commonforms.prepare_form(
                tmp_in_path,
                tmp_out_path,
                model_or_path=model_path,
                keep_existing_fields=params.keep_existing,
                use_signature_fields=params.use_signature_fields,
                device=params.device,
                confidence=params.confidence,
                fast=params.fast,
                multiline=params.multiline,
            )
            
            processing_time = time.time() - processing_start
            logger.info(f"[{request_id}] Processing complete in {processing_time:.2f}s")
            
            # Analyze output DPI if tracking metrics
            output_dpi = None
            if track_metrics:
                logger.info(f"[{request_id}] Analyzing output PDF DPI...")
                output_dpi = dpi_analyzer.get_pdf_dpi(tmp_out_path)
                logger.info(f"[{request_id}] Output DPI: {output_dpi}")
            
            # Collect metrics
            if track_metrics:
                total_time = time.time() - start_time
                
                # Get GPU metrics
                gpu_metrics = None
                if gpu_monitor:
                    gpu_metrics = gpu_monitor.stop_monitoring()
                    logger.info(
                        f"[{request_id}] GPU metrics collected: "
                        f"available={gpu_metrics.gpu_available}, "
                        f"peak_memory={gpu_metrics.peak_memory_mb} MB"
                    )
                
                # Create DPI metrics
                dpi_metrics = None
                if input_dpi is not None and output_dpi is not None:
                    dpi_metrics = DPIMetrics(
                        input_dpi=input_dpi,
                        output_dpi=output_dpi,
                        processing_time_seconds=processing_time
                    )
                
                # Create complete metrics object
                metrics = ProcessingMetrics(
                    request_id=request_id,
                    filename=pdf_file.filename,
                    file_size_bytes=file_size,
                    model_used=params.model,
                    device=params.device,
                    dpi_metrics=dpi_metrics,
                    gpu_metrics=gpu_metrics,
                    total_processing_time=total_time,
                    success=True
                )
                
                # Store metrics
                metrics_tracker.store_metrics(metrics)
                logger.info(f"[{request_id}] Metrics stored successfully")
            
            return tmp_out_path, pdf_file.filename, metrics
            
        except Exception as exc:
            # Clean up on error
            if tmp_in_path:
                Path(tmp_in_path).unlink(missing_ok=True)
            if tmp_out_path:
                Path(tmp_out_path).unlink(missing_ok=True)
            
            # Store error metrics if tracking
            if track_metrics:
                error_metrics = ProcessingMetrics(
                    request_id=request_id,
                    filename=pdf_file.filename,
                    file_size_bytes=file_size if 'file_size' in locals() else 0,
                    model_used=params.model,
                    device=params.device,
                    total_processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(exc)
                )
                metrics_tracker.store_metrics(error_metrics)
            
            logger.error(f"[{request_id}] PDF processing failed: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {str(exc)}"
            )
        finally:
            # Clean up input file
            if tmp_in_path:
                Path(tmp_in_path).unlink(missing_ok=True)


# Global service instance
pdf_service = PDFService()