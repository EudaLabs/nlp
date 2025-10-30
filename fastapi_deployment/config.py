"""Configuration for FastAPI deployment."""
import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "NLP Model API"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    
    # Model Settings
    model_path: str = "./models/bert-imdb"
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    max_length: int = 512
    batch_size: int = 32
    
    # Class names (optional)
    class_names: Optional[List[str]] = ["negative", "positive"]
    
    # CORS Settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # API Key (optional, for authentication)
    api_key: Optional[str] = None
    
    # Performance
    enable_caching: bool = True
    cache_size: int = 1000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Initialize settings
settings = Settings()
