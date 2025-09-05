"""
Tests for configuration settings.
"""

import pytest
import os
from unittest.mock import patch

from config import Settings


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()
        
        assert settings.MODEL_SIZE == "medium"
        assert settings.COMPUTE == "cpu"
        assert settings.NUM_WORKERS == 1
        assert settings.BEAM_SIZE == 5
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.LOG_LEVEL == "INFO"
        assert settings.MAX_FILE_SIZE == 100 * 1024 * 1024  # 100MB
        assert ".mp3" in settings.ALLOWED_EXTENSIONS
        assert ".wav" in settings.ALLOWED_EXTENSIONS
    
    def test_environment_variables(self):
        """Test configuration from environment variables."""
        env_vars = {
            "MODEL_SIZE": "large",
            "COMPUTE": "gpu",
            "NUM_WORKERS": "2",
            "BEAM_SIZE": "10",
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "LOG_LEVEL": "DEBUG",
            "MAX_FILE_SIZE": "200000000"  # 200MB
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.MODEL_SIZE == "large"
            assert settings.COMPUTE == "gpu"
            assert settings.NUM_WORKERS == 2
            assert settings.BEAM_SIZE == 10
            assert settings.HOST == "127.0.0.1"
            assert settings.PORT == 9000
            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.MAX_FILE_SIZE == 200000000
    
    def test_validation_constraints(self):
        """Test configuration validation constraints."""
        # Test NUM_WORKERS constraints
        with patch.dict(os.environ, {"NUM_WORKERS": "0"}):
            with pytest.raises(ValueError):
                Settings()
        
        with patch.dict(os.environ, {"NUM_WORKERS": "5"}):
            with pytest.raises(ValueError):
                Settings()
        
        # Test BEAM_SIZE constraints
        with patch.dict(os.environ, {"BEAM_SIZE": "0"}):
            with pytest.raises(ValueError):
                Settings()
        
        with patch.dict(os.environ, {"BEAM_SIZE": "25"}):
            with pytest.raises(ValueError):
                Settings()
        
        # Test PORT constraints
        with patch.dict(os.environ, {"PORT": "0"}):
            with pytest.raises(ValueError):
                Settings()
        
        with patch.dict(os.environ, {"PORT": "70000"}):
            with pytest.raises(ValueError):
                Settings()
    
    def test_compute_type_validation(self):
        """Test compute type validation."""
        # Valid compute types
        with patch.dict(os.environ, {"COMPUTE": "cpu"}):
            settings = Settings()
            assert settings.COMPUTE == "cpu"
        
        with patch.dict(os.environ, {"COMPUTE": "gpu"}):
            settings = Settings()
            assert settings.COMPUTE == "gpu"
        
        # Invalid compute type
        with patch.dict(os.environ, {"COMPUTE": "invalid"}):
            with pytest.raises(ValueError):
                Settings()
    
    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        settings = Settings()
        
        expected_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
        assert settings.ALLOWED_EXTENSIONS == expected_extensions
    
    def test_env_file_config(self):
        """Test .env file configuration."""
        # This test would require creating a temporary .env file
        # For now, we'll just test that the config class is set up correctly
        settings = Settings()
        assert hasattr(settings.Config, 'env_file')
        assert settings.Config.env_file == ".env"
        assert settings.Config.case_sensitive is True


if __name__ == "__main__":
    pytest.main([__file__])
