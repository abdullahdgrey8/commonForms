# app/services/pdf_service.py
import os
import time
import tempfile
import shutil
import asyncio  # <--- Added asyncio
from pathlib import Path
from fastapi import UploadFile, HTTPException
import torch

from app.core.logging import get_logger
from app.models.pdf import PDFProcessRequest
from app.models.metrics import ProcessingMetrics, DPIMetrics
from app.services.model_cache import model_cache
from app.services.metrics_tracker import metrics_tracker
from app.utils.dpi_analyzer import dpi_analyzer
from app.utils.gpu_monitor import create_gpu_monitor

# import commonforms at function-call time (after env vars set at startup)
import commonforms

logger = get_logger(__name__)

class PDFService:
    def _setup_cuda_environment(self, device: str):
        if device and device.startswith("cuda"):
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                del os.environ['CUDA_VISIBLE_DEVICES']
            # set visible device number (if specified like cuda:0)
            try:
                device_index = device.split(":")[1]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device_index)
            except Exception:
                # leave existing env if parsing failed
                pass

            logger.info(f"CUDA setup successful: device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, torch.cuda.device_count()={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    async def process_pdf(self, pdf_file: UploadFile, params: PDFProcessRequest, track_metrics: bool=False, request_id: str=None):
        if request_id is None:
            request_id = f"req_{int(time.time()*1000)}"
        logger.info(f"[{request_id}] PDFService.process_pdf start: {pdf_file.filename} model={params.model} device={params.device}")

        # ensure HF offline and model cache env (defensive)
        try:
            model_cache.setup_cache_env()
        except Exception as e:
            logger.warning(f"[{request_id}] model_cache.setup_cache_env() failed: {e}")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("ULTRALYTICS_HUB", "off")

        # Setup CUDA env if requested
        self._setup_cuda_environment(params.device)

        tmp_in_path = None
        tmp_out_path = None
        metrics = None
        gpu_monitor = None
        start_time = time.time()

        try:
            # Save file to a temp path
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                shutil.copyfileobj(pdf_file.file, tmp_in)
            tmp_in_path = tmp_in.name

            # file size for metrics
            try:
                file_size = Path(tmp_in_path).stat().st_size
            except Exception:
                file_size = 0

            # optional DPI analysis
            input_dpi = None
            if track_metrics:
                try:
                    input_dpi = dpi_analyzer.get_pdf_dpi(tmp_in_path)
                except Exception as e:
                    logger.debug(f"[{request_id}] DPI read failed: {e}")

            # decide model id/path based on device
            model_id_or_repo = model_cache.get_model_path(params.model, params.device, params.fast)

            # log status
            logger.info(f"[{request_id}] Starting commonforms.prepare_form with model={model_id_or_repo} device={params.device}")

            # Maybe start GPU monitor
            if track_metrics:
                try:
                    gpu_monitor = create_gpu_monitor()
                    gpu_monitor.start_monitoring(params.device)
                except Exception as e:
                    logger.warning(f"[{request_id}] GPU monitor start failed: {e}")
                    gpu_monitor = None

            # output file
            tmp_out_path = tempfile.mktemp(suffix=suffix)

            # Call existing CommonForms entrypoint.
            processing_start = time.time()
            
            # --- FIX: Run blocking synchronous call in a separate thread ---
            await asyncio.to_thread(
                commonforms.prepare_form,
                tmp_in_path,
                tmp_out_path,
                model_or_path=model_id_or_repo,
                keep_existing_fields=params.keep_existing,
                use_signature_fields=params.use_signature_fields,
                device=params.device,
                confidence=params.confidence,
                fast=params.fast,
                multiline=params.multiline,
            )
            # -------------------------------------------------------------

            processing_time = time.time() - processing_start
            logger.info(f"[{request_id}] Processing complete in {processing_time:.2f}s")

            # output DPI analysis
            output_dpi = None
            if track_metrics:
                try:
                    output_dpi = dpi_analyzer.get_pdf_dpi(tmp_out_path)
                except Exception:
                    pass

            # collect GPU metrics
            gpu_metrics = None
            if gpu_monitor:
                try:
                    gpu_metrics = gpu_monitor.stop_monitoring()
                except Exception:
                    gpu_metrics = None

            # build metrics objects
            dpi_metrics = None
            if input_dpi and output_dpi:
                dpi_metrics = DPIMetrics(input_dpi=input_dpi, output_dpi=output_dpi, processing_time_seconds=processing_time)

            total_time = time.time() - start_time
            metrics = ProcessingMetrics(
                request_id=request_id,
                filename=pdf_file.filename,
                file_size_bytes=file_size,
                model_used=params.model,
                device=params.device,
                dpi_metrics=dpi_metrics,
                gpu_metrics=gpu_metrics,
                total_processing_time=round(total_time, 3),
                success=True
            )

            if track_metrics:
                try:
                    metrics_tracker.store_metrics(metrics)
                except Exception:
                    logger.debug(f"[{request_id}] storing metrics failed")

            logger.info(f"[{request_id}] Finished processing in {total_time:.2f}s")
            return tmp_out_path, pdf_file.filename, metrics

        except Exception as exc:
            # cleanup
            if tmp_in_path:
                try: Path(tmp_in_path).unlink(missing_ok=True)
                except Exception: pass
            if tmp_out_path:
                try: Path(tmp_out_path).unlink(missing_ok=True)
                except Exception: pass

            logger.error(f"[{request_id}] PDF processing failed: {exc}", exc_info=True)

            if track_metrics:
                try:
                    fail_metrics = ProcessingMetrics(
                        request_id=request_id,
                        filename=pdf_file.filename,
                        file_size_bytes=file_size if 'file_size' in locals() else 0,
                        model_used=params.model,
                        device=params.device,
                        total_processing_time=round(time.time() - start_time, 3),
                        success=False,
                        error_message=str(exc)
                    )
                    metrics_tracker.store_metrics(fail_metrics)
                except Exception:
                    pass

            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(exc)}")

# global instance
pdf_service = PDFService()