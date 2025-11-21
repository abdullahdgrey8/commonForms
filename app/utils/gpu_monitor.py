# app/utils/gpu_monitor.py
from typing import Optional
import torch
import time
from app.models.metrics import GPUMetrics
from app.core.logging import get_logger

logger = get_logger(__name__)


class GPUMonitor:
    """Monitor GPU memory usage during processing"""
    
    def __init__(self):
        self.initial_memory: Optional[float] = None
        self.peak_memory: Optional[float] = None
        self.device_name: Optional[str] = None
        self.device_num: Optional[int] = None
        self.gpu_available = False
        self.memory_snapshots = []
        self.monitoring_active = False
        
    def start_monitoring(self, device: str = "cpu") -> None:
        """Start monitoring GPU memory"""
        # Reset state
        self.monitoring_active = False
        self.gpu_available = False
        self.initial_memory = None
        self.peak_memory = None
        self.memory_snapshots = []
        
        # Only monitor if CUDA device requested
        if not device.startswith("cuda"):
            logger.info(f"Using {device} - GPU monitoring disabled")
            return
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning(
                f"GPU requested ({device}) but CUDA not available. "
                f"Install PyTorch with CUDA support"
            )
            return
        
        try:
            # Extract device number
            if ":" in device:
                self.device_num = int(device.split(":")[1])
            else:
                self.device_num = 0
            
            # Verify device exists
            if self.device_num >= torch.cuda.device_count():
                logger.warning(
                    f"Device {device} not available. "
                    f"Only {torch.cuda.device_count()} GPU(s) detected"
                )
                return
            
            # Initialize monitoring
            self.device_name = torch.cuda.get_device_name(self.device_num)
            self.gpu_available = True
            self.monitoring_active = True
            
            # Reset peak memory stats for this device
            torch.cuda.reset_peak_memory_stats(self.device_num)
            torch.cuda.empty_cache()
            
            # Record initial memory
            self.initial_memory = torch.cuda.memory_allocated(self.device_num) / (1024 ** 2)
            self.take_memory_snapshot()
            
            logger.info(
                f"GPU monitoring started for {self.device_name} (cuda:{self.device_num}). "
                f"Initial memory: {self.initial_memory:.2f} MB"
            )
            
        except Exception as e:
            logger.error(f"Failed to start GPU monitoring: {e}", exc_info=True)
            self.monitoring_active = False
            self.gpu_available = False
    
    def get_current_memory(self) -> Optional[float]:
        """Get current GPU memory usage in MB"""
        if not self.monitoring_active:
            return None
        
        try:
            return torch.cuda.memory_allocated(self.device_num) / (1024 ** 2)
        except Exception as e:
            logger.error(f"Failed to get current GPU memory: {e}")
            return None
    
    def get_reserved_memory(self) -> Optional[float]:
        """Get reserved GPU memory in MB"""
        if not self.monitoring_active:
            return None
        
        try:
            return torch.cuda.memory_reserved(self.device_num) / (1024 ** 2)
        except Exception as e:
            logger.error(f"Failed to get reserved GPU memory: {e}")
            return None
    
    def take_memory_snapshot(self) -> None:
        """Take a snapshot of current memory usage"""
        if not self.monitoring_active:
            return
        
        try:
            snapshot = {
                "allocated": self.get_current_memory(),
                "reserved": self.get_reserved_memory(),
                "timestamp": time.time()
            }
            self.memory_snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
    
    def get_peak_memory(self) -> Optional[float]:
        """Get peak GPU memory usage in MB"""
        if not self.monitoring_active:
            return None
        
        try:
            return torch.cuda.max_memory_allocated(self.device_num) / (1024 ** 2)
        except Exception as e:
            logger.error(f"Failed to get peak GPU memory: {e}")
            return None
    
    def stop_monitoring(self) -> GPUMetrics:
        """Stop monitoring and return metrics"""
        if not self.monitoring_active:
            return GPUMetrics(
                gpu_available=False,
                device_name=None,
                peak_memory_mb=None,
                initial_memory_mb=None,
                memory_increase_mb=None
            )
        
        try:
            # Take final snapshot
            self.take_memory_snapshot()
            
            # Get peak memory
            self.peak_memory = self.get_peak_memory()
            final_memory = self.get_current_memory()
            reserved_memory = self.get_reserved_memory()
            
            # Calculate memory increase
            memory_increase = None
            if self.initial_memory is not None and self.peak_memory is not None:
                memory_increase = self.peak_memory - self.initial_memory
            
            # Calculate average memory from snapshots
            avg_allocated = None
            if self.memory_snapshots:
                allocated_values = [s["allocated"] for s in self.memory_snapshots if s["allocated"] is not None]
                if allocated_values:
                    avg_allocated = sum(allocated_values) / len(allocated_values)
            
            metrics = GPUMetrics(
                gpu_available=True,
                device_name=self.device_name,
                peak_memory_mb=round(self.peak_memory, 2) if self.peak_memory else None,
                initial_memory_mb=round(self.initial_memory, 2) if self.initial_memory else None,
                memory_increase_mb=round(memory_increase, 2) if memory_increase else None
            )
            
            logger.info(
                f"GPU monitoring stopped. Device: {self.device_name}, "
                f"Peak: {metrics.peak_memory_mb} MB, "
                f"Increase: {metrics.memory_increase_mb} MB"
            )
            
            if reserved_memory:
                logger.info(f"Reserved memory: {reserved_memory:.2f} MB")
            if avg_allocated:
                logger.info(f"Average allocated: {avg_allocated:.2f} MB")
            if final_memory:
                logger.info(f"Final allocated: {final_memory:.2f} MB")
            
            # Cleanup
            torch.cuda.empty_cache()
            self.monitoring_active = False
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to stop GPU monitoring: {e}", exc_info=True)
            return GPUMetrics(
                gpu_available=False,
                device_name=None,
                peak_memory_mb=None,
                initial_memory_mb=None,
                memory_increase_mb=None
            )
    
    def get_gpu_info(self) -> dict:
        """Get general GPU information"""
        if not torch.cuda.is_available():
            return {"gpu_available": False, "message": "CUDA not available"}
        
        try:
            device_num = self.device_num if self.device_num is not None else 0
            device_props = torch.cuda.get_device_properties(device_num)
            
            info = {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(device_num),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "total_memory_gb": round(device_props.total_memory / (1024 ** 3), 2),
                "total_memory_mb": round(device_props.total_memory / (1024 ** 2), 2),
                "current_allocated_mb": round(torch.cuda.memory_allocated(device_num) / (1024 ** 2), 2),
                "current_reserved_mb": round(torch.cuda.memory_reserved(device_num) / (1024 ** 2), 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(device_num) / (1024 ** 2), 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "multi_processor_count": device_props.multi_processor_count
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {"gpu_available": False, "error": str(e)}


def create_gpu_monitor() -> GPUMonitor:
    """Create a new GPU monitor instance"""
    return GPUMonitor()