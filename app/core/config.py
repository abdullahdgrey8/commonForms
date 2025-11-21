from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "CommonForms API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Model Cache
    MODEL_CACHE_DIR: str = "./data/models"
    PRELOAD_MODELS: bool = True
    
    # Processing
    DEFAULT_DEVICE: str = "cpu"
    DEFAULT_MODEL: str = "FFDNet-L"
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB
    
    # Hardcoded test users (use database in production)
    TEST_USER_USERNAME: str = "admin"
    TEST_USER_PASSWORD: str = "changeme123"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    @property
    def model_cache_path(self) -> Path:
        """Get Path object for model cache directory"""
        path = Path(self.MODEL_CACHE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()