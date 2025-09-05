"""
Tests for the WhisperService class.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from whisper_service import WhisperService


class TestWhisperService:
    """Test WhisperService functionality."""
    
    def test_init(self):
        """Test service initialization."""
        service = WhisperService(
            model_size="tiny",
            compute_type="cpu",
            num_workers=2,
            beam_size=3
        )
        assert service.model_size == "tiny"
        assert service.compute_type == "cpu"
        assert service.num_workers == 2
        assert service.beam_size == 3
        assert service.model is None
        assert not service.is_ready()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful model initialization."""
        service = WhisperService(model_size="tiny", compute_type="cpu")
        
        # Mock the model loading
        mock_model = Mock()
        with patch("whisper_service.WhisperModel", return_value=mock_model):
            await service.initialize()
            
            assert service.model == mock_model
            assert service.is_ready()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test model initialization failure."""
        service = WhisperService(model_size="tiny", compute_type="cpu")
        
        # Mock model loading to raise an exception
        with patch("whisper_service.WhisperModel", side_effect=Exception("Model load failed")):
            with pytest.raises(Exception, match="Model load failed"):
                await service.initialize()
            
            assert service.model is None
            assert not service.is_ready()
    
    def test_is_ready(self):
        """Test readiness check."""
        service = WhisperService()
        
        # Not ready initially
        assert not service.is_ready()
        
        # Ready after setting model
        service.model = Mock()
        service._ready = True
        assert service.is_ready()
        
        # Not ready if model is None
        service.model = None
        assert not service.is_ready()
    
    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        """Test successful transcription."""
        service = WhisperService()
        service.model = Mock()
        service._ready = True
        
        # Mock transcription result
        mock_segments = [
            Mock(
                id=0,
                start=0.0,
                end=5.0,
                text="Hello world",
                words=[
                    Mock(start=0.0, end=1.0, word="Hello", probability=0.99),
                    Mock(start=1.0, end=2.0, word="world", probability=0.98)
                ]
            )
        ]
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0
        mock_info.duration_after_vad = 4.8
        mock_info.all_language_probs = {"en": 0.95, "es": 0.03}
        
        service.model.transcribe.return_value = (mock_segments, mock_info)
        
        # Create temporary audio file
        audio_content = b"fake audio data"
        filename = "test.wav"
        
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.wav"
            mock_temp_file.write = Mock()
            mock_temp.return_value.__enter__.return_value = mock_temp_file
            
            with patch("os.unlink") as mock_unlink:
                result = await service.transcribe(
                    audio_content=audio_content,
                    filename=filename,
                    language="en",
                    task="transcribe"
                )
        
        # Verify result structure
        assert result["filename"] == filename
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95
        assert result["duration"] == 5.0
        assert result["transcription"]["full_text"] == "Hello world"
        assert len(result["transcription"]["segments"]) == 1
        assert result["transcription"]["segments"][0]["text"] == "Hello world"
        assert len(result["transcription"]["segments"][0]["words"]) == 2
        assert result["model_info"]["model_size"] == "medium"  # default
        assert result["model_info"]["compute_type"] == "cpu"  # default
        
        # Verify model was called correctly
        service.model.transcribe.assert_called_once()
        call_args = service.model.transcribe.call_args
        assert call_args[0][0] == "/tmp/test.wav"  # file path
        assert call_args[1]["language"] == "en"
        assert call_args[1]["task"] == "transcribe"
        assert call_args[1]["beam_size"] == 5  # default
        assert call_args[1]["word_timestamps"] is True
        assert call_args[1]["vad_filter"] is True
        
        # Verify cleanup
        mock_unlink.assert_called_once_with("/tmp/test.wav")
    
    @pytest.mark.asyncio
    async def test_transcribe_not_ready(self):
        """Test transcription when service is not ready."""
        service = WhisperService()
        # Don't set model or ready flag
        
        with pytest.raises(RuntimeError, match="Whisper service not ready"):
            await service.transcribe(
                audio_content=b"fake audio",
                filename="test.wav"
            )
    
    @pytest.mark.asyncio
    async def test_transcribe_with_translate_task(self):
        """Test transcription with translate task."""
        service = WhisperService()
        service.model = Mock()
        service._ready = True
        
        # Mock transcription result
        mock_segments = [Mock(id=0, start=0.0, end=5.0, text="Hola mundo", words=[])]
        mock_info = Mock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0
        mock_info.duration_after_vad = 4.8
        mock_info.all_language_probs = {"es": 0.95}
        
        service.model.transcribe.return_value = (mock_segments, mock_info)
        
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.wav"
            mock_temp_file.write = Mock()
            mock_temp.return_value.__enter__.return_value = mock_temp_file
            
            with patch("os.unlink"):
                result = await service.transcribe(
                    audio_content=b"fake audio",
                    filename="test.wav",
                    language="es",
                    task="translate"
                )
        
        # Verify model was called with translate task
        call_args = service.model.transcribe.call_args
        assert call_args[1]["task"] == "translate"
        assert call_args[1]["language"] == "es"
    
    @pytest.mark.asyncio
    async def test_transcribe_no_language(self):
        """Test transcription without language specification."""
        service = WhisperService()
        service.model = Mock()
        service._ready = True
        
        # Mock transcription result
        mock_segments = [Mock(id=0, start=0.0, end=5.0, text="Hello world", words=[])]
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0
        mock_info.duration_after_vad = 4.8
        mock_info.all_language_probs = {"en": 0.95}
        
        service.model.transcribe.return_value = (mock_segments, mock_info)
        
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.wav"
            mock_temp_file.write = Mock()
            mock_temp.return_value.__enter__.return_value = mock_temp_file
            
            with patch("os.unlink"):
                result = await service.transcribe(
                    audio_content=b"fake audio",
                    filename="test.wav"
                )
        
        # Verify model was called without language
        call_args = service.model.transcribe.call_args
        assert call_args[1]["language"] is None
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        service = WhisperService()
        
        assert service._get_file_extension("test.wav") == ".wav"
        assert service._get_file_extension("test.mp3") == ".mp3"
        assert service._get_file_extension("test") == ".wav"  # default
        assert service._get_file_extension("") == ".wav"  # default
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test service cleanup."""
        service = WhisperService()
        service.model = Mock()
        service._ready = True
        
        await service.cleanup()
        
        assert service.model is None
        assert not service._ready


if __name__ == "__main__":
    pytest.main([__file__])
