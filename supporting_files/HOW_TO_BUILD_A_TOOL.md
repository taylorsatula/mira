# How to Build a Tool in BotWithMemory

This guide outlines the process for creating new tools that integrate seamlessly with the BotWithMemory system. It provides step-by-step instructions, best practices, and real examples from the codebase.

## Table of Contents
1. [Understanding the Tool Architecture](#understanding-the-tool-architecture)
2. [Tool Development Workflow](#tool-development-workflow)
3. [Creating a Basic Tool](#creating-a-basic-tool)
4. [Advanced Tool Features](#advanced-tool-features)
5. [Testing Your Tool](#testing-your-tool)
6. [Common Patterns and Best Practices](#common-patterns-and-best-practices)
7. [Troubleshooting](#troubleshooting)

## Understanding the Tool Architecture

The BotWithMemory system uses a pluggable tool architecture where each tool:

- Inherits from the base `Tool` class in `tools/repo.py`
- Has a unique name, description, and parameters
- Is automatically discovered and registered by the tool system
- Can be enabled/disabled dynamically during conversations

### Key Components

Before building a tool, familiarize yourself with these key files:

- **tools/repo.py**: Contains the base `Tool` class and `ToolRepository` for managing tools
- **errors.py**: Defines error types and error handling mechanisms
- **tools/tool_finder.py**: Tool that discovers and enables relevant tools based on user queries
- **tools/sample_tool.py**: Reference implementation demonstrating best practices

The `Tool` class provides the following interface that all tools must implement:

```python
class Tool(ABC):
    name = "base_tool"  # Must be unique across all tools
    description = "Base class for all tools"  # A detailed description of what the tool does
    usage_examples = []  # Examples of how to use the tool
    
    def __init__(self):
        self.logger = logging.getLogger(f"tools.{self.name}")
    
    @abstractmethod
    def run(self, **params) -> Dict[str, Any]:
        """Main method that executes the tool's functionality."""
        pass
```

## Tool Development Workflow

Follow this workflow when creating a new tool:

1. **Plan your tool's functionality**:
   - Define a clear, single responsibility for your tool
   - Determine required parameters and return values
   - Identify any external libraries or APIs you'll need
   
2. **Check for existing tools**:
   - Review existing tools to avoid duplication
   - Consider if your functionality can be built by composing existing tools
   
3. **Implement your tool**:
   - Create a new file in the `tools/` directory
   - Follow the structure and patterns from `sample_tool.py`
   - Write comprehensive documentation
   
4. **Write tests**:
   - Create a test file in the `tests/` directory
   - Ensure you test both happy paths and error scenarios
   
5. **Test in the conversation system**:
   - Run the bot and test your tool with real user queries
   - Check that `tool_finder` can discover and enable your tool appropriately

## Creating a Basic Tool

### Step 1: Create the Tool File

Create a new Python file in the `tools/` directory with a descriptive name, for example `my_new_tool.py`.

### Step 2: Import Required Dependencies

```python
import logging
from typing import Dict, List, Any, Optional

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
```

### Step 3: Define Your Tool Class

Create a class that inherits from `Tool` with a descriptive and concise name:

```python
class MyNewTool(Tool):
    """
    Detailed description of your tool's purpose and functionality.
    
    Explain what problem it solves and when it should be used.
    Also note any important limitations or requirements.
    """
    
    name = "my_new_tool"  # A unique, snake_case identifier
    description = """Your tool's detailed description goes here. This description will be 
    used by Claude to determine when to use your tool, so it should be comprehensive.
    
    The description should:
    1. Clearly explain what the tool does in detail
    2. Specify when it should be used (e.g., "Use this tool when...")
    3. Describe each parameter and how it affects the tool's behavior
    4. Note any important caveats or limitations
    
    Format this description with multiple paragraphs for readability.
    """
    
    usage_examples = [
        {
            "input": {"param1": "value1", "param2": "value2"},
            "output": {
                "result": "Example output"
            }
        }
    ]
```

### Step 4: Implement the Constructor

```python
def __init__(self):
    """Initialize your tool with any required setup."""
    super().__init__()
    # Tool-specific initialization
    self.logger.info("MyNewTool initialized")
```

For tools that need dependencies like an LLM bridge, accept them in the constructor:

```python
def __init__(self, llm_bridge: LLMBridge):
    """
    Initialize your tool with dependencies.
    
    Args:
        llm_bridge: LLMBridge instance for generating content
    """
    super().__init__()
    self.llm_bridge = llm_bridge
    # Other initialization
```

### Step 5: Implement the Run Method

```python
def run(
    self,
    required_param: str,
    optional_param: str = "default_value"
) -> Dict[str, Any]:
    """
    Execute the tool's functionality.
    
    This method serves as the main entry point for the tool. It validates inputs,
    processes the request, and returns structured data as a response.
    
    Args:
        required_param: Description of what this parameter does and its format
        optional_param: Description of this optional parameter with its default value
    
    Returns:
        A dictionary containing the tool's response with a clear structure
    
    Raises:
        ToolError: If parameters are invalid or an error occurs during execution
    """
    self.logger.info(f"Running MyNewTool with params: {required_param}, {optional_param}")
    
    # Use error_context for consistent error handling
    with error_context(
        component_name=self.name,
        operation="processing data",
        error_class=ToolError,
        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
        logger=self.logger
    ):
        # Input validation
        if not required_param:
            raise ToolError(
                "The required_param must not be empty",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_value": required_param}
            )
        
        # Tool logic goes here
        result = self._process_data(required_param, optional_param)
        
        # Return structured response
        return {
            "success": True,
            "result": result,
            "input_params": {
                "required_param": required_param,
                "optional_param": optional_param
            }
        }
```

### Step 6: Implement Helper Methods

Use private methods (prefixed with `_`) for internal functionality:

```python
def _process_data(self, input_data: str, options: str) -> str:
    """
    Process the input data according to specified options.
    
    Args:
        input_data: The input data to process
        options: Processing options
        
    Returns:
        The processed result
    """
    # Implementation details
    return f"Processed {input_data} with {options}"
```

## Advanced Tool Features

### Data Storage

For tools that need persistent data:

1. Store data in the `data/tools/your_tool_name/` directory
2. Create the directory structure in your `__init__` method:

```python
def __init__(self):
    super().__init__()
    # Set up data directories
    self.data_dir = os.path.join("data", "tools", "my_tool_name")
    os.makedirs(self.data_dir, exist_ok=True)
```

3. Use standard file operations to read/write data:

```python
def _save_data(self, data: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    file_path = os.path.join(self.data_dir, "my_data.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
```

### Tool Configuration

The system uses a registry-based configuration system that enables true drag-and-drop functionality. To add configuration to your tool:

1. Define a Pydantic configuration class in your tool module:

```python
from pydantic import BaseModel, Field
from config.registry import registry

class MyToolConfig(BaseModel):
    """Configuration for my_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=30, description="Timeout in seconds for operations")
    api_key: str = Field(default="", description="API key for external service")
```

2. Register your configuration with the registry:

```python
# Register with the registry
registry.register("my_tool", MyToolConfig)
```

3. Access your tool's configuration in your tool implementation:

```python
def run(self, query: str) -> Dict[str, Any]:
    # Import config when needed (avoids circular imports)
    from config import config
    
    # Access tool-specific configuration
    timeout = config.my_tool.timeout
    
    # Use configuration values in your code
    response = requests.get(
        f"https://api.example.com/search?q={query}",
        timeout=timeout
    )
    # ...
```

That's it! No additional steps are required. The configuration system will automatically detect and use your tool's configuration.

For more details on the configuration system, see `docs/Tool_Configuration_System.md`.

### Integration with External APIs

For tools that connect to external services:

1. Use the tool's configuration class for API keys and settings
2. Implement graceful error handling for API failures
3. Use appropriate timeouts and retries

```python
def run(self, query: str) -> Dict[str, Any]:
    try:
        # API call with appropriate timeout from tool's config
        response = requests.get(
            f"https://api.example.com/search?q={query}",
            timeout=config.my_tool.timeout
        )
        response.raise_for_status()
        return {"results": response.json()}
    except requests.RequestException as e:
        raise ToolError(
            f"API request failed: {str(e)}",
            ErrorCode.API_CONNECTION_ERROR,
            {"query": query, "error": str(e)}
        )
```

### Dependency Injection

The tool system automatically injects certain dependencies if your tool needs them. Currently supported:

- `LLMBridge`: For tools that need to generate content with the LLM
- `ToolRepository`: For tools that need to interact with other tools

```python
def __init__(self, llm_bridge: LLMBridge, tool_repo: ToolRepository):
    super().__init__()
    self.llm_bridge = llm_bridge
    self.tool_repo = tool_repo
```

## Testing Your Tool

Create a test file in the `tests/` directory with the name `test_your_tool_name.py`.

### Basic Test Structure

```python
import unittest
from unittest.mock import patch, MagicMock

from tools.my_new_tool import MyNewTool
from errors import ToolError

class TestMyNewTool(unittest.TestCase):
    """Tests for the MyNewTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = MyNewTool()
        
    def test_successful_execution(self):
        """Test that the tool works with valid inputs."""
        result = self.tool.run(required_param="test_value")
        self.assertTrue(result["success"])
        self.assertIn("result", result)
        
    def test_invalid_input(self):
        """Test that the tool handles invalid inputs properly."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(required_param="")
            
        self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")
```

### Mocking External Dependencies

For tools with external dependencies, use mocking to isolate your tests:

```python
@patch('requests.get')
def test_api_integration(self, mock_get):
    """Test API integration with mocked response."""
    # Set up mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "value"}
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    result = self.tool.run(query="test")
    
    # Verify the API was called correctly
    mock_get.assert_called_once_with(
        "https://api.example.com/search?q=test", 
        timeout=30
    )
    
    # Verify the result
    self.assertEqual(result["results"]["key"], "value")
```

### Integrated Testing

For thoroughness, add an integration test to ensure your tool works with the entire system:

```python
def test_tool_discoverable(self):
    """Test that the tool can be discovered by the system."""
    from tools.repo import ToolRepository
    
    repo = ToolRepository()
    repo.discover_tools()
    
    # Check that our tool is registered
    self.assertIn("my_new_tool", repo.list_all_tools())
```

## Common Patterns and Best Practices

### Error Handling

Always use the `error_context` manager and `ToolError` for consistent error handling:

```python
with error_context(
    component_name=self.name,
    operation="specific operation name",
    error_class=ToolError,
    error_code=ErrorCode.TOOL_EXECUTION_ERROR,
    logger=self.logger
):
    # Code that might raise exceptions
```

For input validation, use specific error codes:

```python
if invalid_condition:
    raise ToolError(
        "Clear error message explaining what's wrong",
        ErrorCode.TOOL_INVALID_INPUT,
        {"context": "Additional debugging information"}
    )
```

### Comprehensive Documentation

Always include detailed docstrings and parameter descriptions:

1. Class docstring: Explain what the tool does, when to use it, and important considerations
2. Method docstrings: Describe parameters, return values, and exceptions
3. Code comments: Explain complex or non-obvious logic

### Logging

Use appropriate logging levels:

```python
self.logger.debug("Detailed information for debugging")
self.logger.info("General information about tool operation")
self.logger.warning("Warning about potential issues")
self.logger.error("Error information when something fails")
```

### Tool Description Guidelines

Write detailed tool descriptions following these principles:

1. **Be comprehensive**: Include all details about what the tool does
2. **Explain when to use it**: "Use this tool when the user asks about X or needs Y"
3. **Detail parameters**: Explain each parameter's purpose and format
4. **Note limitations**: "This tool cannot X" or "This tool is limited to Y"
5. **Structure for readability**: Use paragraphs and formatting to make the description scannable

## Troubleshooting

### Common Issues

1. **Tool not discovered**: Check that your tool class has a unique `name` attribute and is not in a module starting with underscore
2. **Tool fails with parameter errors**: Ensure your `run` method's signature matches what's documented
3. **Tool not enabled automatically**: Verify your tool description is detailed enough for `tool_finder` to match it to user queries

### Debugging Tools

1. **Check logs**: Enable DEBUG logging to see detailed information
2. **Inspect tool registration**: Run `python -c "from tools.repo import ToolRepository; repo = ToolRepository(); repo.discover_tools(); print(repo.list_all_tools())"`
3. **Test manual enabling**: Try manually enabling your tool to verify it works correctly

## Examples from the Codebase

Take inspiration from these well-implemented tools:

1. **Sample Tool (WeatherTool)** - Basic pattern: `tools/sample_tool.py`
2. **Kasa Tool** - Hardware integration: `tools/kasa_tool.py`
3. **Questionnaire Tool** - Interactive user data collection: `tools/questionnaire_tool.py`

---

By following these guidelines, you'll create tools that integrate seamlessly with the BotWithMemory system, enhancing its capabilities while maintaining consistency and reliability.