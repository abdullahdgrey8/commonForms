# app/services/metrics_tracker.py
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading
from app.models.metrics import ProcessingMetrics, DPIMetrics, GPUMetrics
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class MetricsTracker:
    """Tracks and stores processing metrics"""
    
    def __init__(self):
        self.metrics_dir = Path("data/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / "processing_metrics.jsonl"
        self._lock = threading.Lock()
        
    def generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def store_metrics(self, metrics: ProcessingMetrics) -> None:
        """Store metrics to file"""
        try:
            with self._lock:
                with open(self.metrics_file, 'a') as f:
                    f.write(metrics.model_dump_json() + '\n')
            logger.info(f"Metrics stored for request: {metrics.request_id}")
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def get_recent_metrics(self, limit: int = 100) -> list[ProcessingMetrics]:
        """Retrieve recent metrics"""
        metrics = []
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        metrics.append(ProcessingMetrics.model_validate_json(line))
        except Exception as e:
            logger.error(f"Failed to read metrics: {e}")
        return metrics
    
    def get_metrics_summary(self) -> dict:
        """Get summary statistics of all metrics"""
        metrics = self.get_recent_metrics(limit=1000)
        
        if not metrics:
            return {"message": "No metrics available"}
        
        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        
        processing_times = [m.total_processing_time for m in metrics if m.success]
        
        gpu_usage = sum(1 for m in metrics if m.gpu_metrics and m.gpu_metrics.gpu_available)
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": total - successful,
            "gpu_usage_count": gpu_usage,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
        }


# Global metrics tracker instance
metrics_tracker = MetricsTracker()