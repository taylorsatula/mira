"""
Tests for the configuration management system.
"""
import pytest
import os
from pathlib import Path
import tempfile
import json
from unittest import mock

from pydantic import BaseModel, Field
from config import config
from config.config_manager import AppConfig, get_tool_config_static
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
    from config.config import ApiConfig, SystemConfig
    
    custom_api = ApiConfig(model="claude-test-model", max_tokens=2000)
    custom_system = SystemConfig(log_level="DEBUG")
    
    custom_config = AppConfig(api=custom_api, system=custom_system)
    
    # Check that custom values are set
    assert custom_config.api.model == "claude-test-model"
    assert custom_config.api.max_tokens == 2000
    assert custom_config.system.log_level == "DEBUG"
    
    # Check that default values are used for unspecified fields
    assert custom_config.paths.data_dir == "data"
    assert custom_config.conversation.max_history == 20


def test_env_var_override():
    """Test that environment variables can override configuration."""
    # Set environment variables for testing
    os.environ["AGENT_API__MODEL"] = "claude-env-test"
    os.environ["AGENT_SYSTEM__LOG_LEVEL"] = "WARNING"
    
    try:
        # Create a fresh instance to pick up env vars
        instance = AppConfig()
        # Load environment variables
        from config.config_manager import load_dotenv
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


# Tests for Dynamic Tool Configuration

class MockToolConfig(BaseModel):
    """Mock tool configuration for testing."""
    timeout: int = Field(default=30, description="Tool operation timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries for failed operations")
    api_key: str = Field(default="", description="API key for the service")


def test_tool_config_discovery():
    """Test discovery of tool configurations."""
    with mock.patch("importlib.import_module") as mock_import:
        # Setup mock for tool discovery
        mock_module = mock.MagicMock()
        mock_module.__path__ = ["/mock/tools"]
        mock_import.return_value = mock_module
        
        # Mock pkgutil.iter_modules to return a mock module info
        mock_module_info = mock.MagicMock()
        mock_module_info.name = "tools.mock_tool"
        
        with mock.patch("pkgutil.iter_modules", return_value=[mock_module_info]):
            # Mock the module with a Tool class
            mock_tool_module = mock.MagicMock()
            mock_tool_class = mock.MagicMock()
            mock_tool_class.name = "mock_tool"
            mock_tool_class.__name__ = "MockTool"
            mock_tool_class.__module__ = "tools.mock_tool"
            
            # Add the mock tool class to the mock module
            mock_tool_module.__name__ = "tools.mock_tool"
            type(mock_tool_module).MockTool = mock_tool_class
            
            # Mock the import of the tool module
            with mock.patch.dict("sys.modules", {"tools.mock_tool": mock_tool_module}):
                # Run the discovery method
                config_classes = AppConfig.discover_tool_configs()
                
                # Check that the default config class was created for the mock tool
                assert "mock_tool" in config_classes
                assert issubclass(config_classes["mock_tool"], BaseModel)


def test_tool_config_registration():
    """Test registration of custom tool configurations."""
    # Register a custom config class
    AppConfig.register_config("custom_tool", MockToolConfig)
    
    # Mock the discovery to return our registered config
    with mock.patch.object(AppConfig, "discover_tool_configs", return_value={"custom_tool": MockToolConfig}):
        # Create a new config instance
        app_config = AppConfig.load()
        
        # Check that our tool config was loaded
        assert "custom_tool" in app_config.tool_configs
        tool_config = app_config.tool_configs["custom_tool"]
        assert isinstance(tool_config, MockToolConfig)
        assert tool_config.timeout == 30
        assert tool_config.retry_count == 3
        assert tool_config.api_key == ""


def test_tool_config_from_json():
    """Test loading tool configurations from a JSON file."""
    # Register a custom config class
    AppConfig.register_config("custom_tool", MockToolConfig)
    
    # Create a temporary config file with tool config
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp_file:
        config_data = {
            "tool_configs": {
                "custom_tool": {
                    "timeout": 60,
                    "retry_count": 5,
                    "api_key": "test_key"
                }
            }
        }
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Mock the discovery to return our registered config
        with mock.patch.object(AppConfig, "discover_tool_configs", return_value={"custom_tool": MockToolConfig}):
            # Load configuration from file
            app_config = AppConfig.load(temp_file_path)
            
            # Check that our tool config was loaded with values from the file
            assert "custom_tool" in app_config.tool_configs
            tool_config = app_config.tool_configs["custom_tool"]
            assert isinstance(tool_config, MockToolConfig)
            assert tool_config.timeout == 60
            assert tool_config.retry_count == 5
            assert tool_config.api_key == "test_key"
    
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_get_tool_config():
    """Test getting tool configurations."""
    # Register a custom config class
    AppConfig.register_config("custom_tool", MockToolConfig)
    
    # Mock the discovery to return our registered config
    with mock.patch.object(AppConfig, "discover_tool_configs", return_value={"custom_tool": MockToolConfig}):
        # Create a new config instance
        app_config = AppConfig.load()
        
        # Test get_tool_config method
        tool_config = app_config.get_tool_config("custom_tool")
        assert isinstance(tool_config, MockToolConfig)
        assert tool_config.timeout == 30
        
        # Test get_tool_config with type casting
        typed_config = app_config.get_tool_config("custom_tool", MockToolConfig)
        assert isinstance(typed_config, MockToolConfig)
        
        # Test get_tool_config with non-existent tool
        with pytest.raises(ConfigError):
            app_config.get_tool_config("non_existent_tool")
        
        # Test the dot notation accessor
        assert app_config.get("tool_configs.custom_tool") == tool_config
        assert app_config.get("tool_configs.custom_tool.timeout") == 30
        assert app_config.get("tool_configs.non_existent", "default") == "default"


def test_get_tool_config_static():
    """Test the static helper function for getting tool configurations."""
    # Register a custom config class
    AppConfig.register_config("custom_tool", MockToolConfig)
    
    # Mock the discovery to return our registered config
    with mock.patch.object(AppConfig, "discover_tool_configs", return_value={"custom_tool": MockToolConfig}):
        # Create a new config instance and set it as the global config
        app_config = AppConfig.load()
        with mock.patch("config.config_manager.config", app_config):
            # Test the static helper function
            tool_config = get_tool_config_static("custom_tool")
            assert isinstance(tool_config, MockToolConfig)
            assert tool_config.timeout == 30
            
            # Test with type casting
            typed_config = get_tool_config_static("custom_tool", MockToolConfig)
            assert isinstance(typed_config, MockToolConfig)
            
            # Test with non-existent tool
            with pytest.raises(ConfigError):
                get_tool_config_static("non_existent_tool")