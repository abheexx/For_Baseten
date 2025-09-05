"""
Whisper Inference Service

A production-ready FastAPI service for speech-to-text transcription using faster-whisper.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from whisper_service import WhisperService
from config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics - defined as functions to avoid duplicate registration
def get_transcription_requests_counter():
    try:
        return Counter(
            'transcription_requests_total', 
            'Total number of transcription requests',
            ['model_size', 'compute_type']
        )
    except ValueError:
        # Metric already exists, return existing one
        from prometheus_client import REGISTRY
        for collector in REGISTRY.collectors:
            if hasattr(collector, '_name') and collector._name == 'transcription_requests_total':
                return collector
        raise

def get_transcription_duration_histogram():
    try:
        return Histogram(
            'transcription_duration_seconds',
            'Time spent on transcription',
            ['model_size', 'compute_type']
        )
    except ValueError:
        # Metric already exists, return existing one
        from prometheus_client import REGISTRY
        for collector in REGISTRY.collectors:
            if hasattr(collector, '_name') and collector._name == 'transcription_duration_seconds':
                return collector
        raise

def get_transcription_errors_counter():
    try:
        return Counter(
            'transcription_errors_total',
            'Total number of transcription errors',
            ['error_type']
        )
    except ValueError:
        # Metric already exists, return existing one
        from prometheus_client import REGISTRY
        for collector in REGISTRY.collectors:
            if hasattr(collector, '_name') and collector._name == 'transcription_errors_total':
                return collector
        raise

# Global whisper service instance
whisper_service: Optional[WhisperService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global whisper_service
    
    # Startup
    logger.info("Starting Whisper Inference Service...")
    settings = Settings()
    
    try:
        whisper_service = WhisperService(
            model_size=settings.MODEL_SIZE,
            compute_type=settings.COMPUTE,
            num_workers=settings.NUM_WORKERS,
            beam_size=settings.BEAM_SIZE
        )
        await whisper_service.initialize()
        logger.info(f"Whisper service initialized with model: {settings.MODEL_SIZE}")
    except Exception as e:
        logger.error(f"Failed to initialize whisper service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Whisper Inference Service...")
    if whisper_service:
        await whisper_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Whisper Inference Service",
    description="Production-ready speech-to-text transcription service using faster-whisper",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health_check():
    """Health check endpoint for Kubernetes."""
    return {"status": "healthy", "service": "whisper-inference-service"}

@app.get("/readyz")
async def readiness_check():
    """Readiness check endpoint for Kubernetes."""
    if whisper_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if not whisper_service.is_ready():
        raise HTTPException(status_code=503, detail="Whisper service not ready")
    
    return {"status": "ready", "service": "whisper-inference-service"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    task: str = "transcribe"
):
    """
    Transcribe audio file to text.
    
    Args:
        file: Audio file to transcribe
        language: Optional language code (e.g., 'en', 'es', 'fr')
        task: Task type ('transcribe' or 'translate')
    
    Returns:
        JSON response with transcription results
    """
    if whisper_service is None:
        raise HTTPException(status_code=503, detail="Whisper service not available")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        get_transcription_errors_counter().labels(error_type="invalid_file_type").inc()
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    # Validate task parameter
    if task not in ["transcribe", "translate"]:
        get_transcription_errors_counter().labels(error_type="invalid_task").inc()
        raise HTTPException(
            status_code=400,
            detail="Task must be 'transcribe' or 'translate'"
        )
    
    try:
        # Read file content
        audio_content = await file.read()
        
        if len(audio_content) == 0:
            get_transcription_errors_counter().labels(error_type="empty_file").inc()
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Record metrics
        settings = Settings()
        get_transcription_requests_counter().labels(
            model_size=settings.MODEL_SIZE,
            compute_type=settings.COMPUTE
        ).inc()
        
        # Perform transcription
        with get_transcription_duration_histogram().labels(
            model_size=settings.MODEL_SIZE,
            compute_type=settings.COMPUTE
        ).time():
            result = await whisper_service.transcribe(
                audio_content=audio_content,
                filename=file.filename,
                language=language,
                task=task
            )
        
        # Log successful transcription
        logger.info(f"Successfully transcribed file: {file.filename}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        get_transcription_errors_counter().labels(error_type="transcription_failed").inc()
        logger.error(f"Transcription failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "whisper-inference-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/healthz",
        "ready": "/readyz",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Whisper models are not thread-safe
    )
