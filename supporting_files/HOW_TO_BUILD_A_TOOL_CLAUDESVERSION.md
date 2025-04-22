# How to Build a Tool in BotWithMemory

This guide walks you through creating new tools that integrate with the BotWithMemory system. Follow these instructions to develop tools that maintain consistency with the existing architecture and best practices.

## Table of Contents
1. [Understanding the Tool Architecture](#understanding-the-tool-architecture)
2. [Tool Development Workflow](#tool-development-workflow)
3. [Creating a Basic Tool](#creating-a-basic-tool)
4. [Advanced Tool Features](#advanced-tool-features)
5. [Testing Your Tool](#testing-your-tool)
6. [Common Patterns and Best Practices](#common-patterns-and-best-practices)
7. [Troubleshooting](#troubleshooting)

## Understanding the Tool Architecture

BotWithMemory uses a pluggable tool architecture where each tool:

- Inherits from the base `Tool` class in `tools/repo.py`
- Has a unique name, description, and parameter definitions
- Is automatically discovered and registered by the system
- Can be dynamically enabled/disabled during conversations

### Key Components

Before you begin, familiarize yourself with these essential files:

- **tools/repo.py**: Contains the base `Tool` class and `ToolRepository` for tool management
- **errors.py**: Defines error types and error handling patterns
- **tools/tool_finder.py**: Discovers and enables relevant tools based on user queries
- **tools/sample_tool.py**: Reference implementation showing all recommended practices

The `Tool` base class defines this interface that all tools must implement:

```python
class Tool(ABC):
    name = "base_tool"  # Must be unique across all tools
    description = "Base class for all tools"  # Detailed description of functionality
    usage_examples = []  # Examples demonstrating proper usage
    
    def __init__(self):
        self.logger = logging.getLogger(f"tools.{self.name}")
    
    @abstractmethod
    def run(self, **params) -> Dict[str, Any]:
        """Main method that executes the tool's functionality."""
        pass
```

## Tool Development Workflow

Follow this step-by-step process when creating a new tool:

1. **Plan your tool's functionality**:
   - Define a single, clear purpose for your tool
   - List all required parameters and their expected formats
   - Determine the structure of return values
   - Identify any external dependencies your tool will need
   
2. **Check for existing tools**:
   - Review the current tools directory to avoid duplication
   - Consider whether you can achieve your goal by combining existing tools
   
3. **Implement your tool**:
   - Create a new file in the `tools/` directory
   - Follow the structure and patterns from `sample_tool.py`
   - Include comprehensive documentation at all levels
   
4. **Write tests**:
   - Create a corresponding test file in the `tests/` directory
   - Test both successful operations and error scenarios
   
5. **Validate in the conversation system**:
   - Run the bot and test your tool with realistic user queries
   - Verify that `tool_finder` correctly discovers and enables your tool

## Creating a Basic Tool

### Step 1: Create the Tool File

Create a new Python file in the `tools/` directory with a descriptive name (e.g., `my_new_tool.py`).

### Step 2: Import Required Dependencies

```python
import logging
from typing import Dict, List, Any, Optional

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
```

### Step 3: Define Your Tool Class

Create a class that inherits from `Tool` with a clear, descriptive name:

```python
class MyNewTool(Tool):
    """
    A comprehensive description of your tool's purpose and functionality.
    
    Include details about what problem it solves and when it should be used.
    Document any important limitations or requirements here.
    """
    
    name = "my_new_tool"  # A unique, snake_case identifier
    description = """
    Your tool's detailed description goes here. This text is critical as it helps
    the LLM determine when to use your tool.
    
    Your description should:
    1. Clearly explain what the tool does in specific detail
    2. State explicitly when it should be used (e.g., "Use this tool when...")
    3. Describe each parameter and how it affects the tool's behavior
    4. Document any important limitations or restrictions
    
    Structure this description with paragraphs for readability.
    """
    
    usage_examples = [
        {
            "input": {"param1": "value1", "param2": "value2"},
            "output": {
                "result": "Example output that would be returned"
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

For tools requiring dependencies like an LLM bridge:

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

Create private methods (prefixed with `_`) for internal functionality:

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

For tools requiring persistent data:

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

### External API Integration

For tools connecting to external services:

1. Add configuration in `config/config.py` for settings like API keys
2. Implement robust error handling for API failures
3. Use appropriate timeouts and retries

```python
def run(self, query: str) -> Dict[str, Any]:
    try:
        # API call with timeout
        response = requests.get(
            f"https://api.example.com/search?q={query}",
            timeout=config.my_api.timeout
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

The tool system automatically injects certain dependencies. Currently supported:

- `LLMBridge`: For generating content with the LLM
- `ToolRepository`: For interacting with other tools

```python
def __init__(self, llm_bridge: LLMBridge, tool_repo: ToolRepository):
    super().__init__()
    self.llm_bridge = llm_bridge
    self.tool_repo = tool_repo
```

## Testing Your Tool

Create a test file in the `tests/` directory named `test_your_tool_name.py`.

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

For tools with external dependencies, use mocking:

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

### Integration Testing

Add a test to verify your tool works with the entire system:

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

### Documentation Best Practices

Include comprehensive documentation at all levels:

1. **Class docstring**: Explain the tool's purpose, use cases, and limitations
2. **Method docstrings**: Document parameters, return values, and possible exceptions
3. **Code comments**: Explain complex logic or non-obvious implementation details

### Effective Logging

Use appropriate logging levels for different situations:

```python
self.logger.debug("Detailed information useful for debugging")
self.logger.info("General information about normal operation")
self.logger.warning("Indication of potential issues or unexpected behavior")
self.logger.error("Error information when something fails")
```

### Tool Description Guidelines

Write effective tool descriptions following these principles:

1. **Be comprehensive**: Include all relevant details about functionality
2. **Specify usage scenarios**: Use phrases like "Use this tool when the user asks about X"
3. **Document parameters thoroughly**: Explain the purpose and format of each parameter
4. **State limitations clearly**: "This tool cannot X" or "This tool is limited to Y"
5. **Structure for readability**: Use paragraphs and logical organization

## Troubleshooting

### Common Issues

1. **Tool not discovered**: Ensure your tool has a unique `name` attribute and isn't in a module starting with underscore
2. **Parameter errors**: Verify your `run` method's signature matches your documentation
3. **Tool not automatically enabled**: Check that your description is detailed enough for `tool_finder` to match it to user queries

### Debugging Techniques

1. **Review logs**: Enable DEBUG level logging for detailed information
2. **Check tool registration**: Run this command to verify registration:
   ```python
   python -c "from tools.repo import ToolRepository; repo = ToolRepository(); repo.discover_tools(); print(repo.list_all_tools())"
   ```
3. **Test manual enabling**: Try enabling your tool directly to verify functionality

## Example Tools

Study these well-implemented tools as references:

1. **WeatherTool** (`tools/sample_tool.py`): Demonstrates the basic tool pattern
2. **KasaTool** (`tools/kasa_tool.py`): Shows hardware integration
3. **QuestionnaireTool** (`tools/questionnaire_tool.py`): Implements interactive user data collection

---

By following these guidelines, you'll create tools that enhance the BotWithMemory system while maintaining consistency, reliability, and best practices.
