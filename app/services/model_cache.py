# app/services/model_cache.py
import os
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

class ModelCache:
    def __init__(self):
        self.cache_dir = Path("data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def setup_cache_env(self):
        os.environ["COMMONFORMS_CACHE"] = str(self.cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["ULTRALYTICS_HUB"] = "off"
        logger.info("Offline mode enabled | Models must be in data/models/")

    def get_model_path(self, model_name: str, device: str = "cpu", fast: bool = False) -> str:
        if model_name != "FFDNet-L":
            raise ValueError("Only FFDNet-L is fully supported right now")

        if fast:
            path = self.cache_dir / "FFDNet-L.onnx"
        else:
            path = self.cache_dir / "FFDNet-L.pt"

        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}! Download it first.")

        return str(path)

    def preload_models(self):
        logger.info("Models ready in data/models/ (lazy load)")

model_cache = ModelCache()