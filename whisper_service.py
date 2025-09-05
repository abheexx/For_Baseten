"""
Whisper service implementation using faster-whisper.
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

logger = logging.getLogger(__name__)


class WhisperService:
    """Whisper transcription service using faster-whisper."""
    
    def __init__(
        self,
        model_size: str = "medium",
        compute_type: str = "cpu",
        num_workers: int = 1,
        beam_size: int = 5
    ):
        """
        Initialize the Whisper service.
        
        Args:
            model_size: Size of the Whisper model
            compute_type: Compute type ('cpu' or 'gpu')
            num_workers: Number of worker processes
            beam_size: Beam size for beam search
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.num_workers = num_workers
        self.beam_size = beam_size
        self.model = None
        self._ready = False
        
    async def initialize(self):
        """Initialize the Whisper model asynchronously."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size} ({self.compute_type})")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model
            )
            
            self._ready = True
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _load_model(self):
        """Load the Whisper model (runs in thread pool)."""
        # Map compute type to valid faster-whisper compute types
        compute_type_map = {
            "cpu": "int8",  # Use int8 quantization for CPU
            "gpu": "float16" if torch.cuda.is_available() else "int8"
        }
        
        device = "cuda" if self.compute_type == "gpu" and torch.cuda.is_available() else "cpu"
        compute_type = compute_type_map.get(self.compute_type, "int8")
        
        return WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            num_workers=self.num_workers
        )
    
    def is_ready(self) -> bool:
        """Check if the service is ready to process requests."""
        return self._ready and self.model is not None
    
    async def transcribe(
        self,
        audio_content: bytes,
        filename: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio content to text.
        
        Args:
            audio_content: Raw audio file content
            filename: Original filename
            language: Optional language code
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.is_ready():
            raise RuntimeError("Whisper service not ready")
        
        # Create temporary file for audio content
        with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(filename)) as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                self._transcribe_file,
                temp_file_path,
                language,
                task
            )
            
            # Format results
            result = self._format_transcription_result(segments, info, filename)
            
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
    
    def _transcribe_file(
        self,
        file_path: str,
        language: Optional[str],
        task: str
    ) -> tuple[List[Segment], Dict[str, Any]]:
        """Transcribe audio file (runs in thread pool)."""
        return self.model.transcribe(
            file_path,
            language=language,
            task=task,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
    
    def _format_transcription_result(
        self,
        segments: List[Segment],
        info: Dict[str, Any],
        filename: str
    ) -> Dict[str, Any]:
        """Format transcription results into a structured response."""
        
        # Combine all text
        full_text = " ".join(segment.text for segment in segments)
        
        # Format segments with timestamps
        formatted_segments = []
        for segment in segments:
            formatted_segments.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability
                    }
                    for word in segment.words
                ] if segment.words else []
            })
        
        return {
            "filename": filename,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad,
            "all_language_probs": info.all_language_probs,
            "transcription": {
                "full_text": full_text,
                "segments": formatted_segments
            },
            "model_info": {
                "model_size": self.model_size,
                "compute_type": self.compute_type,
                "beam_size": self.beam_size
            }
        }
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename."""
        return Path(filename).suffix or ".wav"
    
    async def cleanup(self):
        """Clean up resources."""
        if self.model:
            # Clean up model resources
            del self.model
            self.model = None
        
        self._ready = False
        logger.info("Whisper service cleaned up")
