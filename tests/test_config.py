"""
Tests for the configuration management system.
"""
import pytest
import os
from pathlib import Path
import tempfile
import json

from config import config
from config.app_config import AppConfig
from errors import ConfigError, ErrorCode


def test_config_initialization():
    """Test that the global config instance is properly initialized."""
    # Basic config check
    assert config is not None
    assert isinstance(config, AppConfig)
    
    # API config
    assert config.api.model.startswith("claude")
    assert config.api.max_tokens > 0
    assert 0 <= config.api.temperature <= 1
    
    # Paths config
    assert config.paths.data_dir == "data"
    assert config.paths.persistent_dir == "persistent"
    assert "async_results" in config.paths.async_results_dir
    assert "prompts" in str(config.paths.prompts_dir)
    
    # System config
    assert config.system.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert isinstance(config.system.streaming, bool)
    assert config.system.json_indent >= 0


def test_config_dot_notation_access():
    """Test that dot notation config access works properly."""
    # Direct attribute access
    assert config.api.model == config.get("api.model")
    assert config.system.log_level == config.get("system.log_level")
    
    # Non-existent key
    assert config.get("non_existent.key") is None
    assert config.get("non_existent.key", "default") == "default"


def test_config_require():
    """Test that require() method works correctly."""
    # Existing key should work
    assert config.require("api.model") == config.api.model
    
    # Non-existent key should raise ConfigError
    with pytest.raises(ConfigError) as exc_info:
        config.require("non_existent.key")
    assert exc_info.value.code == ErrorCode.MISSING_ENV_VAR


def test_config_extraction_templates():
    """Test that extraction templates are properly loaded."""
    # Check that extraction templates exist
    assert "general" in config.tools.extraction_templates
    assert "sentiment" in config.tools.extraction_templates
    assert "entities" in config.tools.extraction_templates


def test_config_as_dict():
    """Test that as_dict() method returns a complete dictionary."""
    config_dict = config.as_dict()
    
    # Check top-level keys
    assert "api" in config_dict
    assert "paths" in config_dict
    assert "system" in config_dict
    assert "tools" in config_dict
    assert "conversation" in config_dict
    
    # Check nested keys
    assert "model" in config_dict["api"]
    assert "log_level" in config_dict["system"]
    assert "extraction_templates" in config_dict["tools"]


def test_system_prompt_loading():
    """Test that system prompts can be loaded properly."""
    # Use a real prompt file
    try:
        prompt = config.get_system_prompt("main_system_prompt")
        assert prompt is not None
        assert len(prompt) > 0
    except ConfigError:
        pytest.skip("main_system_prompt.txt not found, skipping test")


def test_custom_config_direct_creation():
    """Test creating config by directly setting values."""
    # Create a custom config
    from config.schemas.base import ApiConfig, SystemConfig
    
    custom_api = ApiConfig(model="claude-test-model", max_tokens=2000)
    custom_system = SystemConfig(log_level="DEBUG")
    
    custom_config = AppConfig(api=custom_api, system=custom_system)
    
    # Check that custom values are set
    assert custom_config.api.model == "claude-test-model"
    assert custom_config.api.max_tokens == 2000
    assert custom_config.system.log_level == "DEBUG"
    
    # Check that default values are used for unspecified fields
    assert custom_config.paths.data_dir == "data"
    assert custom_config.conversation.max_history == 10


def test_env_var_override():
    """Test that environment variables can override configuration."""
    # Set environment variables for testing
    os.environ["AGENT_API__MODEL"] = "claude-env-test"
    os.environ["AGENT_SYSTEM__LOG_LEVEL"] = "WARNING"
    
    try:
        # Create a fresh instance to pick up env vars
        instance = AppConfig()
        # Load environment variables
        from config.app_config import load_dotenv
        load_dotenv()
        
        # Apply environment variables manually
        instance = AppConfig.load()
        
        # Check the values (should be overridden by environment variables)
        assert instance.api.model == "claude-env-test"
        assert instance.system.log_level == "WARNING"
    finally:
        # Clean up environment
        del os.environ["AGENT_API__MODEL"]
        del os.environ["AGENT_SYSTEM__LOG_LEVEL"]