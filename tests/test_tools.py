"""
Tests for the tool system.

This module tests the Tool and ToolRepository classes, including
tool registration, discovery, and invocation.
"""
import pytest
import inspect
from unittest.mock import patch, MagicMock

from tools.repo import Tool, ToolRepository
from tools.sometool import WeatherTool
from errors import ToolError, ErrorCode


class TestTool(Tool):
    """Test tool implementation for testing."""
    name = "test_tool"
    description = "A tool for testing"
    usage_examples = [
        {
            "input": {"param1": "value1", "param2": 42},
            "output": {"result": "test result"}
        }
    ]
    
    def run(self, param1: str, param2: int = 0) -> dict:
        """
        Run the test tool.
        
        Args:
            param1: First parameter
            param2: Second parameter (optional)
            
        Returns:
            Result dictionary
        """
        return {
            "param1": param1,
            "param2": param2,
            "result": f"{param1}-{param2}"
        }


def test_tool_base_class():
    """Test the base Tool class properties and methods."""
    # Check class properties
    assert Tool.name == "base_tool"
    assert "Base tool class" in Tool.description
    assert Tool.usage_examples == []
    
    # Instantiate should work
    tool = Tool()
    assert tool.logger is not None
    
    # Run should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        tool.run(param="test")


def test_tool_implementation():
    """Test a concrete Tool implementation."""
    # Check class properties
    assert TestTool.name == "test_tool"
    assert TestTool.description == "A tool for testing"
    assert len(TestTool.usage_examples) == 1
    
    # Instantiate and run
    tool = TestTool()
    result = tool.run(param1="hello", param2=42)
    
    # Check result
    assert result["param1"] == "hello"
    assert result["param2"] == 42
    assert result["result"] == "hello-42"
    
    # Test with default parameter
    result = tool.run(param1="world")
    assert result["param1"] == "world"
    assert result["param2"] == 0
    assert result["result"] == "world-0"


def test_tool_parameter_schema():
    """Test generating parameter schema from tool methods."""
    # Get schema from TestTool
    schema = TestTool.get_parameter_schema()
    
    # Check schema structure
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    
    # Check properties
    properties = schema["properties"]
    assert "param1" in properties
    assert "param2" in properties
    assert properties["param1"]["type"] == "string"
    assert properties["param2"]["type"] == "integer"
    
    # Check required parameters
    assert "param1" in schema["required"]
    assert "param2" not in schema["required"]  # Has default value


def test_tool_definition():
    """Test generating tool definition for API integration."""
    # Get definition from TestTool
    definition = TestTool.get_tool_definition()
    
    # Check definition structure
    assert definition["name"] == "test_tool"
    assert definition["description"] == "A tool for testing"
    assert "input_schema" in definition
    
    # Check input schema
    assert definition["input_schema"]["type"] == "object"
    assert "param1" in definition["input_schema"]["properties"]
    assert "param2" in definition["input_schema"]["properties"]


def test_tool_repository_init():
    """Test initializing the tool repository."""
    # Initialize with default package
    repo = ToolRepository()
    assert repo.tools_package == "tools"
    
    # Initialize with custom package
    repo = ToolRepository(tools_package="custom_tools")
    assert repo.tools_package == "custom_tools"


def test_tool_repository_discovery(monkeypatch):
    """Test discovering tools."""
    # Mock pkgutil.iter_modules to return our test modules
    def mock_iter_modules(paths):
        return [
            (None, "repo", False),  # Should be skipped
            (None, "sometool", False),  # Should be loaded
        ]
    
    # Mock importlib.import_module
    def mock_import_module(name):
        if name == "tools":
            mock_module = MagicMock()
            mock_module.__path__ = ["/mock/path"]
            return mock_module
        elif name == "tools.sometool":
            # Create a module with our test tool
            mock_module = MagicMock()
            mock_module.WeatherTool = WeatherTool
            return mock_module
        else:
            raise ImportError(f"Cannot import {name}")
    
    # Apply mocks
    monkeypatch.setattr("pkgutil.iter_modules", mock_iter_modules)
    monkeypatch.setattr("importlib.import_module", mock_import_module)
    
    # Initialize repository with discovery
    repo = ToolRepository()
    
    # Check discovered tools
    assert "weather_tool" in repo.tools
    assert repo.tools["weather_tool"] == WeatherTool


def test_tool_registration():
    """Test registering tools manually."""
    # Initialize empty repository
    repo = ToolRepository()
    
    # Clear tools from discovery
    repo.tools = {}
    
    # Register a tool
    repo.register_tool(TestTool)
    
    # Check registration
    assert "test_tool" in repo.tools
    assert repo.tools["test_tool"] == TestTool
    
    # Register the same tool again (should overwrite)
    repo.register_tool(TestTool)
    assert repo.tools["test_tool"] == TestTool
    
    # Try to register an invalid tool (no name)
    class InvalidTool(Tool):
        name = ""
        
        def run(self, param):
            return param
    
    # Should not register and not raise an exception
    repo.register_tool(InvalidTool)
    assert "" not in repo.tools


def test_get_tool():
    """Test getting a tool by name."""
    # Initialize repository
    repo = ToolRepository()
    
    # Register a test tool
    repo.tools = {}  # Clear any discovered tools
    repo.register_tool(TestTool)
    
    # Get existing tool
    tool_class = repo.get_tool("test_tool")
    assert tool_class == TestTool
    
    # Get non-existent tool
    assert repo.get_tool("non_existent_tool") is None


def test_get_all_tool_definitions():
    """Test getting definitions for all tools."""
    # Initialize repository
    repo = ToolRepository()
    
    # Register test tools
    repo.tools = {}  # Clear any discovered tools
    repo.register_tool(TestTool)
    repo.register_tool(WeatherTool)
    
    # Get all definitions
    definitions = repo.get_all_tool_definitions()
    
    # Check definitions
    assert len(definitions) == 2
    
    # Find each tool's definition
    test_tool_def = next((d for d in definitions if d["name"] == "test_tool"), None)
    weather_tool_def = next((d for d in definitions if d["name"] == "weather_tool"), None)
    
    assert test_tool_def is not None
    assert weather_tool_def is not None
    assert test_tool_def["description"] == "A tool for testing"
    assert "Get current weather information" in weather_tool_def["description"]


def test_invoke_tool():
    """Test invoking a tool by name."""
    # Initialize repository
    repo = ToolRepository()
    
    # Register a test tool
    repo.tools = {}  # Clear any discovered tools
    repo.register_tool(TestTool)
    
    # Invoke the tool
    result = repo.invoke_tool("test_tool", {"param1": "hello", "param2": 42})
    
    # Check result
    assert result["param1"] == "hello"
    assert result["param2"] == 42
    assert result["result"] == "hello-42"
    
    # Invoke non-existent tool
    with pytest.raises(ToolError) as excinfo:
        repo.invoke_tool("non_existent_tool", {})
    assert excinfo.value.code == ErrorCode.TOOL_NOT_FOUND
    
    # Invoke with invalid parameters
    with pytest.raises(ToolError) as excinfo:
        repo.invoke_tool("test_tool", {"invalid_param": "value"})
    assert excinfo.value.code == ErrorCode.TOOL_EXECUTION_ERROR


def test_repo_special_methods():
    """Test special methods on the ToolRepository."""
    # Initialize repository
    repo = ToolRepository()
    
    # Register test tools
    repo.tools = {}  # Clear any discovered tools
    repo.register_tool(TestTool)
    repo.register_tool(WeatherTool)
    
    # Test len
    assert len(repo) == 2
    
    # Test contains
    assert "test_tool" in repo
    assert "weather_tool" in repo
    assert "non_existent_tool" not in repo


def test_weather_tool():
    """Test the included WeatherTool."""
    # Initialize the tool
    tool = WeatherTool()
    
    # Run with default parameters
    result = tool.run(location="New York")
    
    # Check result structure
    assert "location" in result
    assert "temperature" in result
    assert "conditions" in result
    assert "humidity" in result
    assert "wind_speed" in result
    assert result["location"] == "New York"
    assert result["units"] == "celsius"
    
    # Run with custom parameters
    result = tool.run(
        location="Boston",
        units="fahrenheit",
        include_forecast=True
    )
    
    # Check result
    assert result["location"] == "Boston"
    assert result["units"] == "fahrenheit"
    assert "forecast" in result
    assert len(result["forecast"]) == 5  # 5-day forecast
    
    # Check invalid units
    with pytest.raises(ValueError):
        tool.run(location="Chicago", units="invalid_unit")