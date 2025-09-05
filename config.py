"""
Configuration settings for the Whisper Inference Service.
"""

import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model configuration
    MODEL_SIZE: str = Field(
        default="medium",
        description="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)"
    )
    
    COMPUTE: Literal["cpu", "gpu"] = Field(
        default="cpu",
        description="Compute type for inference"
    )
    
    NUM_WORKERS: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of worker processes"
    )
    
    BEAM_SIZE: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Beam size for beam search"
    )
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Server port")
    
    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # File upload limits
    MAX_FILE_SIZE: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum file size in bytes"
    )
    
    ALLOWED_EXTENSIONS: set = Field(
        default={".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"},
        description="Allowed audio file extensions"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True
