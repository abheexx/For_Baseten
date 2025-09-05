"""
Tests for the main FastAPI application.
"""

import pytest
import io
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile

from main import app
from whisper_service import WhisperService


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_whisper_service():
    """Mock whisper service for testing."""
    mock_service = Mock(spec=WhisperService)
    mock_service.is_ready.return_value = True
    mock_service.transcribe = AsyncMock()
    return mock_service


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    # Create a minimal WAV file header (44 bytes)
    wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    return io.BytesIO(wav_header)


@pytest.fixture
def sample_transcription_result():
    """Sample transcription result."""
    return {
        "filename": "test.wav",
        "language": "en",
        "language_probability": 0.95,
        "duration": 5.0,
        "duration_after_vad": 4.8,
        "all_language_probs": {"en": 0.95, "es": 0.03, "fr": 0.02},
        "transcription": {
            "full_text": "Hello, this is a test transcription.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello, this is a test transcription.",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.99},
                        {"start": 0.5, "end": 1.0, "word": ",", "probability": 0.95},
                        {"start": 1.0, "end": 1.2, "word": "this", "probability": 0.98},
                        {"start": 1.2, "end": 1.4, "word": "is", "probability": 0.97},
                        {"start": 1.4, "end": 1.6, "word": "a", "probability": 0.96},
                        {"start": 1.6, "end": 2.0, "word": "test", "probability": 0.98},
                        {"start": 2.0, "end": 2.5, "word": "transcription", "probability": 0.97},
                        {"start": 2.5, "end": 3.0, "word": ".", "probability": 0.99}
                    ]
                }
            ]
        },
        "model_info": {
            "model_size": "medium",
            "compute_type": "cpu",
            "beam_size": 5
        }
    }


class TestHealthEndpoints:
    """Test health and readiness endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "whisper-inference-service"
    
    def test_readiness_check_not_ready(self, client):
        """Test readiness check when service is not ready."""
        response = client.get("/readyz")
        assert response.status_code == 503
        data = response.json()
        assert "not ready" in data["detail"]
    
    @patch("main.whisper_service")
    def test_readiness_check_ready(self, mock_whisper_service, client):
        """Test readiness check when service is ready."""
        mock_service = Mock()
        mock_service.is_ready.return_value = True
        mock_whisper_service.return_value = mock_service
        
        # Mock the global whisper_service
        with patch("main.whisper_service", mock_service):
            response = client.get("/readyz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["service"] == "whisper-inference-service"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        # Check for some expected metrics
        content = response.text
        assert "transcription_requests_total" in content
        assert "transcription_duration_seconds" in content
        assert "transcription_errors_total" in content


class TestTranscriptionEndpoint:
    """Test transcription endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "whisper-inference-service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "/docs" in data["docs"]
    
    def test_transcribe_no_service(self, client, sample_audio_file):
        """Test transcription when service is not available."""
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", sample_audio_file, "audio/wav")}
        )
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
    
    def test_transcribe_invalid_file_type(self, client, mock_whisper_service):
        """Test transcription with invalid file type."""
        with patch("main.whisper_service", mock_whisper_service):
            response = client.post(
                "/transcribe",
                files={"file": ("test.txt", b"not audio", "text/plain")}
            )
            assert response.status_code == 400
            assert "audio file" in response.json()["detail"]
    
    def test_transcribe_empty_file(self, client, mock_whisper_service):
        """Test transcription with empty file."""
        with patch("main.whisper_service", mock_whisper_service):
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", b"", "audio/wav")}
            )
            assert response.status_code == 400
            assert "Empty audio file" in response.json()["detail"]
    
    def test_transcribe_invalid_task(self, client, mock_whisper_service, sample_audio_file):
        """Test transcription with invalid task parameter."""
        with patch("main.whisper_service", mock_whisper_service):
            response = client.post(
                "/transcribe?task=invalid",
                files={"file": ("test.wav", sample_audio_file, "audio/wav")}
            )
            assert response.status_code == 400
            assert "transcribe or translate" in response.json()["detail"]
    
    @patch("main.whisper_service")
    def test_transcribe_success(self, mock_whisper_service, client, sample_audio_file, sample_transcription_result):
        """Test successful transcription."""
        # Setup mock
        mock_service = Mock()
        mock_service.is_ready.return_value = True
        mock_service.transcribe = AsyncMock(return_value=sample_transcription_result)
        mock_whisper_service.return_value = mock_service
        
        with patch("main.whisper_service", mock_service):
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", sample_audio_file, "audio/wav")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test.wav"
            assert data["language"] == "en"
            assert "Hello, this is a test transcription." in data["transcription"]["full_text"]
            assert len(data["transcription"]["segments"]) == 1
    
    @patch("main.whisper_service")
    def test_transcribe_with_language(self, mock_whisper_service, client, sample_audio_file, sample_transcription_result):
        """Test transcription with language parameter."""
        mock_service = Mock()
        mock_service.is_ready.return_value = True
        mock_service.transcribe = AsyncMock(return_value=sample_transcription_result)
        mock_whisper_service.return_value = mock_service
        
        with patch("main.whisper_service", mock_service):
            response = client.post(
                "/transcribe?language=en",
                files={"file": ("test.wav", sample_audio_file, "audio/wav")}
            )
            assert response.status_code == 200
            # Verify language was passed to transcribe method
            mock_service.transcribe.assert_called_once()
            call_args = mock_service.transcribe.call_args
            assert call_args[1]["language"] == "en"
    
    @patch("main.whisper_service")
    def test_transcribe_with_task(self, mock_whisper_service, client, sample_audio_file, sample_transcription_result):
        """Test transcription with task parameter."""
        mock_service = Mock()
        mock_service.is_ready.return_value = True
        mock_service.transcribe = AsyncMock(return_value=sample_transcription_result)
        mock_whisper_service.return_value = mock_service
        
        with patch("main.whisper_service", mock_service):
            response = client.post(
                "/transcribe?task=translate",
                files={"file": ("test.wav", sample_audio_file, "audio/wav")}
            )
            assert response.status_code == 200
            # Verify task was passed to transcribe method
            mock_service.transcribe.assert_called_once()
            call_args = mock_service.transcribe.call_args
            assert call_args[1]["task"] == "translate"
    
    @patch("main.whisper_service")
    def test_transcribe_service_error(self, mock_whisper_service, client, sample_audio_file):
        """Test transcription when service raises an error."""
        mock_service = Mock()
        mock_service.is_ready.return_value = True
        mock_service.transcribe = AsyncMock(side_effect=Exception("Transcription failed"))
        mock_whisper_service.return_value = mock_service
        
        with patch("main.whisper_service", mock_service):
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", sample_audio_file, "audio/wav")}
            )
            assert response.status_code == 500
            assert "Transcription failed" in response.json()["detail"]


class TestOpenAPIDocs:
    """Test OpenAPI documentation endpoints."""
    
    def test_docs_endpoint(self, client):
        """Test that docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test that redoc endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Whisper Inference Service"
        assert data["info"]["version"] == "1.0.0"
        assert "/transcribe" in data["paths"]
        assert "/healthz" in data["paths"]
        assert "/readyz" in data["paths"]
        assert "/metrics" in data["paths"]


if __name__ == "__main__":
    pytest.main([__file__])
