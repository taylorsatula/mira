"""
Tests for the configuration management module.

This module tests the Config class, including loading from different
sources, accessing configuration values, and handling errors.
"""
import os
import json
import pytest
from pathlib import Path

from config import Config
from errors import ConfigError, ErrorCode


def test_config_defaults():
    """Test that default configuration values are set correctly."""
    config = Config()
    
    # Check default values
    assert config.get("log_level") == "INFO"
    assert config.get("data_dir") == "data"
    assert config.get("api.model") == "claude-3-7-sonnet-20250219"
    assert config.get("api.temperature") == 0.7
    assert config.get("conversation.max_history") == 10


def test_config_from_file(temp_dir):
    """Test loading configuration from a JSON file."""
    # Create a test config file
    config_file = temp_dir / "test_config.json"
    test_config = {
        "log_level": "DEBUG",
        "api": {
            "model": "custom-model",
            "temperature": 0.5
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(test_config, f)
    
    # Load the config
    config = Config(config_file)
    
    # Check values from file were loaded
    assert config.get("log_level") == "DEBUG"
    assert config.get("api.model") == "custom-model"
    assert config.get("api.temperature") == 0.5
    
    # Check defaults still apply for unspecified values
    assert config.get("data_dir") == "data"
    assert config.get("api.max_tokens") == 1000


def test_config_from_env(monkeypatch):
    """Test overriding configuration with environment variables."""
    # Set environment variables
    monkeypatch.setenv("AGENT_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("AGENT_API__MODEL", "env-model")
    monkeypatch.setenv("AGENT_API__TEMPERATURE", "0.3")
    
    # Load config
    config = Config()
    
    # Check environment values were loaded
    assert config.get("log_level") == "ERROR"
    assert config.get("api.model") == "env-model"
    assert config.get("api.temperature") == 0.3


def test_config_nested_values(test_config):
    """Test accessing nested configuration values."""
    # Access nested values with dot notation
    assert test_config.get("api.model") == "test-model"
    assert test_config.get("api.max_tokens") == 500
    
    # Access with dictionary syntax
    assert test_config["api.model"] == "test-model"
    
    # Access with get method default
    assert test_config.get("missing.key", "default") == "default"


def test_config_require():
    """Test requiring configuration values."""
    config = Config()
    
    # Should not raise for existing keys
    assert config.require("log_level") == "INFO"
    
    # Should raise for missing keys
    with pytest.raises(ConfigError) as excinfo:
        config.require("missing.key")
    
    assert excinfo.value.code == ErrorCode.MISSING_ENV_VAR


def test_config_invalid_file(temp_dir):
    """Test handling of invalid configuration files."""
    # Create an invalid JSON file
    invalid_file = temp_dir / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("{ invalid json ")
    
    # Should raise a ConfigError
    with pytest.raises(ConfigError) as excinfo:
        Config(invalid_file)
    
    assert excinfo.value.code == ErrorCode.INVALID_CONFIG


def test_config_file_not_found():
    """Test handling of non-existent configuration files."""
    # Should raise a ConfigError
    with pytest.raises(ConfigError) as excinfo:
        Config("/path/that/does/not/exist.json")
    
    assert excinfo.value.code == ErrorCode.CONFIG_NOT_FOUND


def test_config_complex_env_values(monkeypatch):
    """Test parsing complex values from environment variables."""
    # Set environment variables with JSON values
    monkeypatch.setenv("AGENT_TOOLS__ENABLED", "false")
    monkeypatch.setenv("AGENT_API__RATE_LIMIT_RPM", "20")
    monkeypatch.setenv("AGENT_COMPLEX", '{"key1": "value1", "key2": 42}')
    
    # Load config
    config = Config()
    
    # Check values were parsed correctly
    assert config.get("tools.enabled") is False
    assert config.get("api.rate_limit_rpm") == 20
    assert isinstance(config.get("complex"), dict)
    assert config.get("complex.key1") == "value1"
    assert config.get("complex.key2") == 42