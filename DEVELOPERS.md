# Developer Documentation

This document provides detailed technical documentation for developers working with the AI Agent System.

## Core Design Principles

The AI Agent System is built on several core design principles:

1. **Single Responsibility**: Each module and class has a clear, single responsibility. This makes the code easier to understand, maintain, and extend.

2. **Clean Interfaces**: Components interact through well-defined interfaces, which reduces coupling and makes it easier to replace or mock components for testing.

3. **Minimal Dependencies**: The system minimizes external dependencies where possible, favoring the Python standard library.

4. **Standardized Error Handling**: A comprehensive error handling system with custom exceptions, error codes, clear error messages, and a unified context manager approach for consistent error wrapping.

5. **Typed Code**: The codebase uses type hints throughout to improve code clarity and enable static type checking.

6. **No Unnecessary Abstractions**: The system avoids overly complex abstractions, focusing on simple, understandable code.

## System Components

### Error Handling (`errors.py`)

The error system is built around the `AgentError` base class with specialized subclasses for different types of errors:

- `ConfigError`: Configuration-related errors
- `APIError`: API communication errors
- `FileOperationError`: File operation errors
- `ToolError`: Tool execution errors
- `ConversationError`: Conversation management errors

Error codes are defined in the `ErrorCode` enum, grouped by category:
- 1xx: Configuration errors
- 2xx: API errors
- 3xx: File operation errors
- 4xx: Tool errors
- 5xx: Conversation errors
- 9xx: Uncategorized/system errors

#### Centralized Error Context Manager

The system provides a centralized `error_context` context manager for standardized error handling:

```python
from errors import error_context, ToolError, ErrorCode

# Use with any component
with error_context(
    component_name="my_component",
    operation="specific operation",
    error_class=ToolError,  # Or other appropriate error class
    error_code=ErrorCode.TOOL_EXECUTION_ERROR,
    logger=self.logger
):
    # Code that might raise exceptions
    result = process_data(input)
```

This approach ensures consistent error handling, logging, and wrapping across all components.

#### Specialized Error Handling

Some components extend this pattern with domain-specific error handling:

- `LLMBridge` includes the `api_error_context` context manager with specialized API retry logic
- `PersistenceTool` uses custom handling for file-specific errors

The `handle_error` utility function provides standardized user-facing error messages.

### Configuration Management (`config.py`)

The configuration system loads settings from multiple sources in order of increasing precedence:

1. Default settings defined in the `Config.DEFAULT_CONFIG` dictionary
2. Configuration file (JSON format)
3. Environment variables prefixed with `AGENT_`

Nested configuration keys in environment variables use double underscores (`__`). For example, `AGENT_API__MODEL` sets the `api.model` configuration key.

The global `config` instance can be imported and used throughout the application:

```python
from config import config

# Access configuration values
model_name = config.get("api.model")
api_key = config.api_key
```

### File Operations (`crud.py` and `tools/persistence_tool.py`)

#### Basic File Operations (`crud.py`)

The `FileOperations` class provides CRUD operations for working with JSON files:

- `create`: Create a new JSON file
- `read`: Read data from a JSON file
- `write`: Write data to a JSON file
- `update`: Update data in a JSON file using a function
- `delete`: Delete a JSON file
- `list_files`: List JSON files in the data directory

All methods include proper error handling and validation.

#### Persistence Tool (`tools/persistence_tool.py`)

The `PersistenceTool` provides a higher-level interface for data persistence with a clear separation between data and file operations:

1. **Data Operations**:
   - `get_data`: Retrieve a specific value from a JSON file
   - `set_data`: Store a value in a JSON file
   - `delete_data`: Remove a value from a JSON file
   - `list_keys`: List all keys in a JSON file

2. **File Operations**:
   - `get_file`: Retrieve the entire contents of a JSON file
   - `set_file`: Store complete data in a JSON file
   - `list_files`: List all available JSON files

Example usage:
```python
# Get a specific value
result = persistence_tool.run(
    operation="get_data",
    location="preferences.json",
    key="theme"
)

# Store a value
result = persistence_tool.run(
    operation="set_data",
    location="preferences.json",
    key="theme",
    value="dark"
)

# Save an entire file
result = persistence_tool.run(
    operation="set_file",
    location="async_results/task-123.json",
    data={"task_id": "task-123", "result": {"data": "value"}}
)

# Load an entire file
result = persistence_tool.run(
    operation="get_file",
    location="async_results/task-123.json"
)
```

The persistence tool handles path resolution, directory creation, error handling, and provides a consistent interface for all persistence operations.

#### Working with Asynchronous Task Results

Asynchronous tasks store their results in the `persistent/async_results/` directory. To work with these results:

```python
# Save async task result
result = persistence_tool.run(
    operation="set_file",
    location="async_results/task-123.json",  # 'task-123' is the task ID
    data={
        "task_id": "task-123",
        "status": "completed",
        "result": {"data": "analysis results"}
    }
)

# Retrieve async task result
result = persistence_tool.run(
    operation="get_file",
    location="async_results/task-123.json"
)
# The value is in result["value"]

# List all async results
result = persistence_tool.run(
    operation="list_files"
)
# Filter for async results
async_files = [f for f in result["files"] if "async_results" in f]
```

The AsyncTaskManager automatically detects when the persistence tool is used to save results in the `async_results` directory and provides appropriate task completion messages.

### API Communication (`api/llm_bridge.py`)

The `LLMBridge` class handles communication with the Anthropic API, including:

- API authentication
- Request formatting
- Response parsing
- Error handling
- Rate limiting
- Retries with exponential backoff

The `generate_response` method sends a request to the API and returns the response:

```python
response = llm_bridge.generate_response(
    messages=[{"role": "user", "content": "Hello, world!"}],
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=1000,
    tools=tool_repo.get_all_tool_definitions()
)
```

Helper methods for extracting content from responses:
- `extract_text_content`: Get text content from a response
- `extract_tool_calls`: Get tool calls from a response

### Tool System (`tools/repo.py`)

The tool system is built around two main classes:

1. `Tool`: Base class for all tools, defining the tool interface
2. `ToolRepository`: Central registry for tool discovery and invocation

Tools are automatically discovered and registered when the `ToolRepository` is initialized. Each tool must inherit from the `Tool` base class and define:

- `name`: Unique name for the tool
- `description`: Description of what the tool does
- `run` method: Implementation of the tool's functionality

The `ToolRepository` handles tool discovery, registration, and invocation:

```python
# Get a tool definition
tool_definition = tool_repo.get_tool("tool_name").get_tool_definition()

# Invoke a tool
result = tool_repo.invoke_tool("tool_name", {"param1": "value1"})
```

### Conversation Management (`conversation.py`)

The `Conversation` class manages the conversation flow, including:

- Message history tracking
- Context management
- Tool result integration
- Conversation persistence

The `Message` dataclass represents individual messages in the conversation.

### Main Control Flow (`main.py`)

The `main.py` module provides the entry point for the application and handles:

- Command-line argument parsing
- System initialization
- Interactive command loop
- Error handling
- Conversation saving and loading

### External Stimulus Handling (`stimuli.py`)

The `stimuli.py` module provides interfaces and components for receiving and processing external triggers:

- `StimulusType`: Enum of supported stimulus types (message, notification, event, etc.)
- `Stimulus`: Dataclass representing an external stimulus with metadata
- `StimulusHandler`: Central manager for routing stimuli to appropriate handlers
- Helper functions for formatting stimuli for LLM prompts

The stimulus system follows a publish-subscribe pattern, where handlers can register for specific stimulus types.

#### Stimulus Implementation Patterns

The stimulus system provides several patterns for integrating external triggers with conversations:

##### 1. Basic Handler Registration

Define functions that process specific types of stimuli:

```python
def handle_notification(stimulus: Stimulus) -> None:
    # Process notification stimulus
    print(f"Received notification: {stimulus.content}")
    
# Initialize the handler
stimulus_handler = StimulusHandler()

# Register the handler for notifications
stimulus_handler.register_handler(StimulusType.NOTIFICATION, handle_notification)
```

##### 2. Stimulus Creation and Processing

Generate stimuli from external triggers:

```python
# Create a stimulus directly
notification = stimulus_handler.create_stimulus(
    stimulus_type=StimulusType.NOTIFICATION,
    content="New message received",
    source="message_service",
    metadata={"priority": "high"}
)

# Process the stimulus
stimulus_handler.process_stimulus(notification)

# Or do both in one step
stimulus_handler.create_and_process(
    stimulus_type=StimulusType.NOTIFICATION,
    content="New message received",
    source="message_service",
    metadata={"priority": "high"}
)
```

##### 3. Conversation Integration

There are several ways to integrate stimuli with conversations:

**Method 1: Direct Attachment**

The simplest approach is to attach a conversation to a stimulus handler:

```python
conversation = Conversation(system_prompt="You are a helpful assistant.")
stimulus_handler = StimulusHandler()

# Attach the conversation to the stimulus handler
stimulus_handler.attach_conversation(conversation)

# Now all stimuli will be automatically added to the conversation
stimulus_handler.create_and_process(
    stimulus_type=StimulusType.NOTIFICATION,
    content="System update available",
    source="update_service"
)
```

**Method 2: Response Processing**

For stimuli that require response handling:

```python
# Define a response callback
def handle_response(stimulus: Stimulus, response: str) -> None:
    print(f"AI responded to {stimulus.type.value}: {response}")
    # Take programmatic action based on the response
    if "update now" in response.lower():
        start_update_process()

# Attach with response processing
stimulus_handler.attach_conversation(
    conversation, 
    stimulus_types=[StimulusType.NOTIFICATION], 
    response_callback=handle_response
)
```

**Method 3: Manual Conversion**

For more control over how stimuli are added to conversations:

```python
def custom_handler(stimulus: Stimulus) -> None:
    # Format the stimulus as needed
    formatted_content = f"IMPORTANT ALERT: {stimulus.content}"
    
    # Add to conversation with custom metadata
    add_stimulus_to_conversation(stimulus, conversation)
    
    # Generate a response if needed
    response = conversation.generate_response("")
    print(f"Response: {response}")

stimulus_handler.register_handler(StimulusType.ALARM, custom_handler)
```

##### 4. Utility Functions

Several utility functions make it easier to work with stimuli in conversations:

```python
# Check if a message originated from a stimulus
if is_stimulus_message(message):
    # Get stimulus metadata
    metadata = get_stimulus_metadata(message)
    print(f"Message from stimulus of type: {metadata['stimulus_type']}")

# Process a stimulus through a conversation
response = process_stimulus(
    stimulus, 
    conversation,
    lambda s, r: print(f"Got response: {r}")
)
```

##### 5. Extending Stimulus Types

To add new stimulus types, extend the `StimulusType` enum:

```python
# Custom stimulus types
class ExtendedStimulusType(StimulusType):
    VOICE = "voice"  # Voice input stimulus
    LOCATION = "location"  # Location change stimulus
```

##### Implementation Details

- Stimuli are added to conversations as "user" messages (the API only accepts "user" and "assistant" roles)
- Metadata is attached to messages to track their stimulus origin
- Response handling can be automated through callbacks
- The system supports both synchronous and event-driven patterns

The stimulus system is designed to be flexible and extensible, allowing for integration with various external triggers while maintaining a consistent interface.

## Developing New Tools

### Step 1: Create a New Tool Class

Create a new file in the `tools` directory, defining a class that inherits from `Tool`:

```python
# tools/my_tool.py
from typing import Dict, Any
from tools.repo import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "A tool that does something useful"
    
    def run(self, input_param: str, optional_param: int = 0) -> Dict[str, Any]:
        """
        Run the tool.
        
        Args:
            input_param: The main input parameter
            optional_param: An optional parameter with a default value
            
        Returns:
            Result dictionary
        """
        # Tool implementation
        result = {
            "input": input_param,
            "processed": f"Processed: {input_param} (optional: {optional_param})"
        }
        return result
```

### Step 2: Ensure Type Hints and Documentation

For your tool to work properly with the parameter schema generation, ensure:

1. All parameters to the `run` method have type hints
2. The `run` method has a return type hint
3. The `run` method has a proper docstring describing the parameters

### Step 3: Handling Tool-Specific State

If your tool needs to maintain state:

1. Initialize the state in the `__init__` method
2. Use instance variables to store the state
3. Access the state in the `run` method

Example:

```python
def __init__(self):
    super().__init__()
    # Tool-specific state
    self.counter = 0
    self.results_cache = {}

def run(self, input_param: str) -> Dict[str, Any]:
    # Update state
    self.counter += 1
    
    # Check cache
    if input_param in self.results_cache:
        return self.results_cache[input_param]
    
    # Process and cache result
    result = self._process_input(input_param)
    self.results_cache[input_param] = result
    
    return result
```

### Step 4: Error Handling in Tools

Tools should use the centralized error context manager for consistent error handling:

```python
from errors import error_context, ToolError, ErrorCode

def run(self, input_param: str) -> Dict[str, Any]:
    # Validate input
    if not input_param:
        raise ToolError(
            "Input parameter cannot be empty",
            ErrorCode.TOOL_INVALID_INPUT,
            {"input": input_param}
        )
    
    # Use error context for operation that might fail
    with error_context(
        component_name=self.name,
        operation="processing input",
        error_class=ToolError,
        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
        logger=self.logger
    ):
        # Process input (any exceptions will be caught and properly wrapped)
        result = self._process_input(input_param)
        return result
```

This approach ensures:
- Consistent error handling across all tools
- Proper error wrapping with component name and operation context
- Automatic logging of errors
- Preservation of error details

## Tool Interface Contract

Tools must adhere to the following contract:

1. **Class Attributes**:
   - `name`: String, unique name for the tool
   - `description`: String, description of what the tool does
   - `usage_examples`: List, optional examples of how to use the tool

2. **Required Methods**:
   - `run(**kwargs)`: Execute the tool with the provided parameters

3. **Parameter Schema**:
   - Type hints on the `run` method's parameters are used to generate the parameter schema
   - Return type hints are recommended for documentation
   - Docstrings should describe parameters and return value

4. **Return Values**:
   - Tools should return JSON-serializable data (dict, list, str, int, float, bool, None)
   - Complex objects should be converted to dictionaries
   - Errors should be raised as exceptions, not returned as values

## State Management Best Practices

1. **Short-lived State**: Tools should generally avoid maintaining long-lived state. If state is needed, consider:
   - Storing state in external storage (files, databases)
   - Making state explicit in the tool's parameters and return values
   - Using a state manager if state needs to be shared between tools

2. **Conversation State**: For state that needs to persist across conversation turns:
   - Use the conversation's metadata for tool-specific state
   - Store the state in the tool's return value and extract it from future calls

3. **External Resources**: For tools that interact with external resources:
   - Open and close resources within a single tool call when possible
   - Use context managers to ensure resources are properly closed
   - If resources must persist, manage them carefully with proper error handling

## Testing Recommendations

1. **Unit Tests**: Write unit tests for all tools and components:
   - Test normal operation
   - Test edge cases
   - Test error conditions

2. **Mock External Dependencies**: Use mocks for external dependencies:
   - API calls
   - File operations
   - External services

3. **Parameterized Tests**: Use parameterized tests for testing different input variations.

4. **Integration Tests**: Write integration tests for the full system flow.

Example test for a tool:

```python
import pytest
from tools.my_tool import MyTool
from errors import ToolError

def test_my_tool_normal_operation():
    tool = MyTool()
    result = tool.run(input_param="test")
    assert "processed" in result
    assert result["processed"] == "Processed: test (optional: 0)"

def test_my_tool_with_optional_param():
    tool = MyTool()
    result = tool.run(input_param="test", optional_param=42)
    assert "processed" in result
    assert result["processed"] == "Processed: test (optional: 42)"

def test_my_tool_with_empty_input():
    tool = MyTool()
    with pytest.raises(ToolError) as exc_info:
        tool.run(input_param="")
    assert "Invalid input" in str(exc_info.value)
```

## Common Pitfalls and Solutions

### API Rate Limiting

The `LLMBridge` class includes rate limiting, but be careful when:
- Running multiple instances of the application
- Making API calls outside of the `LLMBridge`

Solution: Use a centralized rate limiter or implement distributed rate limiting.

### Tool Discovery Issues

If tools aren't being discovered:
- Ensure the tool class inherits from `Tool`
- Ensure the tool is in a module in the `tools` package
- Check for import errors in the tool module

### Error Handling in Tool Calls

If a tool raises an uncaught exception, it will be caught by the `ToolRepository` and wrapped in a `ToolError`. However, it's better to handle errors within the tool and raise appropriate `ToolError` exceptions.

### Context Length Management

The conversation manager tries to manage context length by limiting the history, but for complex tools or long conversations, you may need additional measures:
- Implement message summarization
- Implement more aggressive history pruning
- Use embeddings or retrieval for handling long-term context

## Integration Patterns

### Tool Composition

Tools can be composed by having one tool call another:

```python
def run(self, input_param: str) -> Dict[str, Any]:
    # Get the tool repository
    from tools.repo import ToolRepository
    tool_repo = ToolRepository()
    
    # Call another tool
    pre_process_result = tool_repo.invoke_tool("preprocessor", {"input": input_param})
    
    # Process the result
    final_result = self._process_result(pre_process_result)
    
    return final_result
```

However, use this pattern sparingly to avoid circular dependencies.

### External Services Integration

For tools that integrate with external services:

1. Use a separate class for the service client:
   ```python
   # tools/services/weather_service.py
   class WeatherService:
       def get_weather(self, location: str) -> Dict[str, Any]:
           # Implementation...
   
   # tools/weather_tool.py
   from tools.services.weather_service import WeatherService
   
   class WeatherTool(Tool):
       def __init__(self):
           super().__init__()
           self.service = WeatherService()
       
       def run(self, location: str) -> Dict[str, Any]:
           return self.service.get_weather(location)
   ```

2. Handle service errors appropriately:
   ```python
   def run(self, location: str) -> Dict[str, Any]:
       try:
           return self.service.get_weather(location)
       except ServiceError as e:
           raise ToolError(f"Weather service error: {e}", ErrorCode.TOOL_EXECUTION_ERROR)
   ```

3. Consider implementing retry logic for flaky services.

## Conclusion

This document covers the core technical aspects of the AI Agent System. For further questions or clarifications, please consult the code documentation or open an issue on the project repository.