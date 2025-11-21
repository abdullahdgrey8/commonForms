# app/utils/device_validator.py
"""
Device validation and recommendation utility
"""
import torch
from typing import Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


class DeviceValidator:
    """Validates and recommends compute devices"""
    
    @staticmethod
    def validate_device(device: str) -> Tuple[bool, str, str]:
        """
        Validate if requested device is available
        
        Args:
            device: Device string (e.g., 'cpu', 'cuda', 'cuda:0')
            
        Returns:
            Tuple of (is_valid, validated_device, message)
        """
        # CPU is always valid
        if device == "cpu":
            return True, "cpu", "Using CPU"
        
        # Check CUDA devices
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                message = (
                    "GPU requested but CUDA not available. "
                    "Falling back to CPU. "
                    "To enable GPU: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu118"
                )
                logger.warning(message)
                return False, "cpu", message
            
            # Extract device number
            try:
                if ":" in device:
                    device_num = int(device.split(":")[1])
                else:
                    device_num = 0
                    device = "cuda:0"  # Normalize
                
                # Check if device exists
                if device_num >= torch.cuda.device_count():
                    message = (
                        f"GPU device {device_num} not found. "
                        f"Available GPUs: {torch.cuda.device_count()}. "
                        f"Falling back to CPU."
                    )
                    logger.warning(message)
                    return False, "cpu", message
                
                gpu_name = torch.cuda.get_device_name(device_num)
                message = f"Using GPU: {gpu_name}"
                logger.info(message)
                return True, device, message
                
            except (ValueError, IndexError) as e:
                message = f"Invalid device format: {device}. Falling back to CPU."
                logger.warning(message)
                return False, "cpu", message
        
        # Unknown device format
        message = f"Unknown device: {device}. Falling back to CPU."
        logger.warning(message)
        return False, "cpu", message
    
    @staticmethod
    def get_device_info() -> dict:
        """Get information about available devices"""
        info = {
            "cpu_available": True,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "device_string": f"cuda:{i}",
                        "total_memory_gb": round(
                            torch.cuda.get_device_properties(i).total_memory / (1024**3), 
                            2
                        ),
                        "compute_capability": torch.cuda.get_device_capability(i)
                    }
                    info["gpus"].append(gpu_info)
                except Exception as e:
                    logger.error(f"Could not get info for GPU {i}: {e}")
        
        return info
    
    @staticmethod
    def recommend_device() -> str:
        """Recommend best available device"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"


# Global validator instance
device_validator = DeviceValidator()