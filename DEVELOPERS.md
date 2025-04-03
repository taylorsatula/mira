# DEVELOPERS.md

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Setup & Development Environment](#setup--development-environment)
4. [Configuration System](#configuration-system)
5. [Tool Development Guide](#tool-development-guide)
6. [Asynchronous Processing](#asynchronous-processing)
7. [Testing Strategy](#testing-strategy)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Contributing Guidelines](#contributing-guidelines)
11. [Troubleshooting Common Issues](#troubleshooting-common-issues)
12. [Advanced Usage Patterns](#advanced-usage-patterns)
13. [Contribution Best Practices](#contribution-best-practices)

## Architecture Overview

This project implements a conversational AI agent system with memory, tool integration, and background task processing. The architecture follows these core principles:

1. **Conversation-Driven Design**: The system maintains a conversation context that drives interactions with the LLM and tool execution.

2. **Tool Composition**: Functionality is implemented through composable tools with single responsibilities rather than monolithic features.

3. **Persistent Memory**: The system maintains state across sessions through file-based persistence.

4. **Asynchronous Processing**: Time-consuming tasks can be offloaded to a background service.

5. **Error Resilience**: Comprehensive error handling strategies protect against API failures, invalid inputs, and system errors.

The system operates in this general flow:

```
User Input → Conversation Manager → LLM Bridge → Tool Selection → Tool Execution → Response Formation → User Output
                      ↑                                   ↓
                      └───────── Persistent Storage ──────┘
```

With background processing:

```
Task Request → Async Client → File Queue → Background Service → Tool Execution → Result Storage → Notification → Main Application
```

## Core Components

### Conversation Manager (`conversation.py`)

The conversation manager tracks message history and handles context management:

- **Message Structure**: Messages are stored with role, content, and metadata
- **Context Management**: Implements pruning strategies when context exceeds token limits
- **Tool Integration**: Processes tool calls from LLM responses and integrates results

```python
# Core methods
add_message(role, content, metadata)
get_formatted_messages(include_tools=True)
process_tool_calls(tool_calls)
```

Interface contract requirements:
- Messages must maintain their structure through serialization/deserialization
- Tool results must be integrated in a format the LLM can process
- Context pruning must preserve critical conversation elements

### LLM Bridge (`api/llm_bridge.py`)

The bridge handles all communication with the Anthropic API:

- **API Integration**: Manages authentication, request formation, and response parsing
- **Rate Limiting**: Implements token bucket algorithm for respecting API rate limits
- **Error Handling**: Provides comprehensive error handling with exponential backoff

```python
# Core methods
generate_response(messages, tools=None, stream=False)
extract_text(response)
extract_tool_calls(response)
```

Critical specifications:
- Request formation MUST follow Anthropic API specifications exactly
- Rate limiting MUST respect both token-per-minute and requests-per-minute limits
- Retry logic MUST use exponential backoff with jitter
- API status codes MUST be mapped to appropriate error types

### Tool System (`tools/repo.py`, `tools/tool_finder.py`)

The tool system provides a framework for extending the agent's capabilities:

- **Tool Repository**: Central registry for all available tools
- **Tool Interface**: Common interface for all tools
- **Tool Discovery**: Dynamic tool selection based on conversation

```python
# Core interfaces
class Tool(ABC):
    @abstractmethod
    def run(self, **kwargs):
        pass
        
class ToolRepository:
    register_tool(tool)
    get_tool(name)
    execute_tool(name, **kwargs)
```

Implementation requirements:
- Tools MUST implement the Tool interface
- Repository MUST handle tool dependencies
- Tool errors MUST be properly propagated and handled

### Configuration System (`config/config.py`, `config/config_manager.py`)

The configuration system manages application settings from multiple sources:

- **Configuration Schema**: Pydantic models define and validate settings
- **Loading Hierarchy**: Environment variables override file-based configuration
- **Access Methods**: Dot notation and get/require methods

```python
# Core access patterns
config.api.model                             # Direct attribute access
config.get("api.model", default="claude-3")  # Get with default
config.require("api.key")                    # Raises error if not found
```

Critical specifications:
- Configuration MUST be validated against Pydantic schemas
- Environment variables MUST override file settings using AGENT__ prefix
- Missing required values MUST raise appropriate exceptions

### Background Service (`background_service.py`, `async_client.py`)

The background processing system handles asynchronous task execution:

- **Task Queue**: File-based queue using directory structure
- **Task Lifecycle**: Pending → Running → Completed/Failed
- **Notification System**: Signals task completion status

```
Directories:
- pending/      # Tasks waiting to be processed
- running/      # Tasks currently being executed
- completed/    # Successfully completed tasks
- failed/       # Failed tasks with error information
```

Implementation requirements:
- Task state transitions MUST be atomic when possible
- Task results MUST be properly persisted
- Background service MUST handle interruptions gracefully

## Setup & Development Environment

### Requirements

- Python 3.10 or higher
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/botwithmemory.git
cd botwithmemory

# Install dependencies
pip install -r requirements.txt

# Set up your API key
export AGENT__API__KEY=your_anthropic_api_key
```

### Directory Structure

```
botwithmemory/
│
├── api/                  # API integration
├── config/               # Configuration system
│   └── prompts/          # System prompts
├── tools/                # Tool implementations
├── data/                 # Data storage
├── persistent/           # Persisted tasks and conversations
│   ├── pending/          # Pending background tasks
│   ├── running/          # Currently executing tasks
│   ├── completed/        # Completed tasks
│   └── failed/           # Failed tasks with error info
└── tests/                # Test suite
```

### Development Workflow

1. Configure your environment:
   ```bash
   # Setup environment variables
   export AGENT__API__KEY=your_anthropic_api_key
   # For development
   export AGENT__PATHS__DATA_DIR=./data
   ```

2. Run the application:
   ```bash
   python main.py
   ```

3. Start the background service (separate terminal):
   ```bash
   python background_service.py
   ```

4. Running tests:
   ```bash
   pytest
   # Specific test file
   pytest tests/test_config.py
   # Specific test
   pytest tests/test_config.py::test_config_loading
   ```

5. Code style enforcement:
   ```bash
   # Linting
   flake8
   # Type checking
   mypy .
   # Formatting
   black .
   ```

## Configuration System

The configuration system uses Pydantic for schema validation and loads settings from multiple sources:

### Configuration Structure

```
AppConfig
├── API
│   ├── model
│   ├── key
│   └── api_url
├── Paths
│   ├── config_dir
│   ├── data_dir
│   └── prompts_dir
├── Conversation
│   ├── max_tokens
│   ├── context_window
│   └── temperature
├── Tools
│   └── [tool-specific configs]
└── System
    ├── default_background_delay
    └── log_level
```

### Loading Hierarchy (highest precedence first)

1. Environment variables (prefixed with `AGENT__`)
2. User configuration file
3. Default configuration values

### Environment Variables

Environment variables use double underscores to represent nested keys:

```bash
# Examples:
export AGENT__API__KEY=your_api_key
export AGENT__API__MODEL=claude-3-sonnet-20240229
export AGENT__CONVERSATION__MAX_TOKENS=4000
```

### Configuration Files

Configuration is stored in JSON format:

```json
{
  "api": {
    "model": "claude-3-sonnet-20240229",
    "key": "your_api_key"
  },
  "conversation": {
    "max_tokens": 4000
  }
}
```

### Extending Configuration

To add new configuration sections:

1. Define a Pydantic model in `config/config.py`
2. Add it to the main `AppConfig` class
3. Provide defaults in the `get_defaults()` method

Example pattern:
```python
class NewFeatureConfig(BaseModel):
    enabled: bool = True
    timeout: int = 30

class AppConfig(BaseModel):
    # Existing sections...
    new_feature: NewFeatureConfig
```

## Tool Development Guide

Tools extend the agent's capabilities by providing specific functionality through a consistent interface.

### Tool Interface

All tools must implement the `Tool` abstract base class:

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool"""
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description"""
        
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Parameter schema"""
        
    @abstractmethod
    def run(self, **kwargs):
        """Execute the tool functionality"""
```

### Tool Development Process

1. **Define tool specification**
   - Unique name
   - Clear description of functionality
   - Parameter schema with types and descriptions
   - Expected return value format

2. **Implement tool class**
   - Inherit from `Tool` base class
   - Implement required properties and methods
   - Include input validation
   - Handle errors appropriately
   - Return results in a consistent format

3. **Register with repository**
   - Tools are automatically discovered and registered if they inherit from `Tool`
   - Custom registration can be done explicitly with `repository.register_tool(MyTool())`

### Tool Implementation Pattern

```python
# Pseudocode for tool implementation
class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"
        
    @property
    def description(self) -> str:
        return "Performs a specific function"
        
    @property
    def parameters(self) -> dict:
        return {
            "parameter1": {
                "type": "string",
                "description": "Description of parameter1",
                "required": True
            },
            # Additional parameters...
        }
    
    def run(self, **kwargs):
        # 1. Validate inputs
        self._validate_inputs(kwargs)
        
        # 2. Process the core functionality
        result = self._process(kwargs)
        
        # 3. Format the output
        return self._format_output(result)
        
    def _validate_inputs(self, inputs):
        # Validation logic
        if "required_param" not in inputs:
            raise ToolError("Missing required parameter")
    
    def _process(self, inputs):
        # Core processing logic
        pass
        
    def _format_output(self, result):
        # Output formatting logic
        return {
            "status": "success",
            "data": result
        }
```

### Background-Capable Tools

To make a tool background-capable:

```python
class BackgroundCapableTool(Tool):
    @property
    def background_capable(self) -> bool:
        return True
```

### Tool Composition

Tools should be designed for composition. Example of composing tools:

```python
def run(self, **kwargs):
    # Extract information using extraction tool
    extracted_data = self.tool_repository.execute_tool(
        "extraction_tool", 
        text=kwargs["text"],
        extraction_type="addresses"
    )
    
    # Store results using persistence tool
    return self.tool_repository.execute_tool(
        "persistence_tool",
        operation="set_data",
        file_name="addresses.json",
        data=extracted_data
    )
```

## Asynchronous Processing

The system supports offloading time-consuming tasks to a background service.

### Task Lifecycle

1. **Task Creation**: Main application creates a task with:
   - Unique ID
   - Tool name and parameters
   - Priority and metadata

2. **Task Submission**: Async client writes task to pending directory

3. **Task Execution**: Background service:
   - Moves task from pending to running
   - Executes specified tool with parameters
   - Writes results or errors to disk
   - Moves task to completed or failed directory

4. **Result Retrieval**: Main application:
   - Checks task status
   - Retrieves results when complete

### Directory Structure

The background service uses a directory-based queue:

```
persistent/
├── pending/      # Tasks waiting to be processed
├── running/      # Tasks currently being executed
├── completed/    # Successfully completed tasks
└── failed/       # Failed tasks with error information
```

Each task is stored as a JSON file containing:
- Task ID
- Creation timestamp
- Tool name and parameters
- Execution result or error information
- Completion timestamp

### Using the Async Client

```python
# Pseudocode for submitting and checking background tasks
from async_client import AsyncClient

# Initialize client
async_client = AsyncClient()

# Submit task
task_id = async_client.submit_task(
    tool_name="long_running_tool",
    parameters={"param1": "value1"}
)

# Check status
status = async_client.get_task_status(task_id)

# Get results when complete
if status == "completed":
    results = async_client.get_task_results(task_id)
```

### Implementation Considerations

- Tasks must be serializable to JSON
- Background tasks should be idempotent when possible
- Long-running tasks should provide progress updates
- Error handling should be comprehensive

## Testing Strategy

The project uses pytest for testing with a focus on unit and integration tests.

### Test Structure

```
tests/
├── unit/               # Unit tests for individual components
├── integration/        # Tests for component interactions
├── fixtures/           # Test fixtures and data
└── conftest.py         # Pytest configuration
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
   - Config system
   - Tool implementations
   - LLM bridge
   - Conversation manager

2. **Integration Tests**: Test component interactions
   - Tool composition
   - Background task execution
   - End-to-end conversation flow

3. **Mock Tests**: Test with mock LLM responses
   - Conversation dynamics
   - Tool call handling
   - Error scenarios

### LLM Response Mocking

```python
# Pseudocode for mocking LLM responses
def test_tool_call_handling(mocker):
    # Mock LLM response with tool calls
    mock_llm_response = {
        "content": [],
        "tool_calls": [
            {
                "name": "test_tool",
                "parameters": {"param1": "value1"}
            }
        ]
    }
    
    # Apply mock
    mocker.patch("api.llm_bridge.LLMBridge.generate_response", 
                 return_value=mock_llm_response)
    
    # Run conversation with input that should trigger tool
    conversation = Conversation(config, tool_repository)
    conversation.add_user_message("Use the test tool")
    response = conversation.get_assistant_response()
    
    # Verify tool was called with expected parameters
    assert "test_tool was called" in response
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_config_loading

# Run with coverage
pytest --cov=.
```

## Error Handling

The project implements a comprehensive error handling strategy.

### Error Types

```
BaseError
├── ConfigError
│   ├── ConfigFileError
│   └── ConfigValidationError
├── APIError
│   ├── RateLimitError
│   ├── AuthenticationError
│   └── ServiceUnavailableError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolExecutionError
│   └── ToolParameterError
├── ConversationError
├── PersistenceError
└── BackgroundServiceError
```

### Error Handling Patterns

1. **Specific Exception Types**: Use the most specific exception type

   ```python
   if not os.path.exists(config_file):
       raise ConfigFileError(f"Configuration file not found: {config_file}")
   ```

2. **Contextual Information**: Include relevant context in error messages

   ```python
   try:
       response = requests.post(url, json=payload, timeout=30)
       response.raise_for_status()
   except requests.exceptions.RequestException as e:
       raise APIError(f"API request failed: {str(e)}", status_code=getattr(e.response, 'status_code', None))
   ```

3. **Error Codes**: Include error codes for programmatic handling

   ```python
   class APIError(BaseError):
       def __init__(self, message, status_code=None, error_code=None):
           self.status_code = status_code
           self.error_code = error_code or "API_ERROR"
           super().__init__(message)
   ```

4. **Recovery Strategies**: Implement appropriate recovery mechanisms

   ```python
   # Pseudocode for retry logic
   def make_api_request_with_retry(payload, max_retries=3):
       retries = 0
       while retries < max_retries:
           try:
               return make_api_request(payload)
           except RateLimitError:
               # Exponential backoff
               sleep_time = 2 ** retries
               time.sleep(sleep_time)
               retries += 1
           except AuthenticationError:
               # Don't retry auth errors
               raise
       
       # If we get here, we've exhausted retries
       raise APIError("Maximum retries exceeded")
   ```

### Logging

- Use Python's logging module consistently
- Log at appropriate severity levels
- Include contextual information in log messages
- Avoid logging sensitive data

```python
logger = logging.getLogger(__name__)

try:
    # Potentially failing operation
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {str(e)}", exc_info=True)
    raise OperationError(f"Failed to complete operation: {str(e)}")
```

## Performance Considerations

### Token Usage Optimization

1. **Context Pruning**: Implement strategies to keep context within token limits
   - Summarize older messages
   - Remove redundant information
   - Prioritize recent and relevant context

2. **Prompt Engineering**: Design efficient prompts
   - Clear instructions with minimal token usage
   - Structured formats that are easy to parse
   - Examples that demonstrate concise patterns

3. **Tool Result Management**: Handle tool results efficiently
   - Return only necessary data
   - Format results to minimize token usage
   - Consider truncating verbose outputs

### Rate Limiting

The system uses a token bucket algorithm to manage API rate limits:

- Respects token-per-minute limits
- Manages request-per-minute limits
- Supports burst requests when capacity is available
- Implements backoff strategies when limits are reached

### Memory Management

Strategies for managing conversation memory:

1. **Selective Persistence**: Store only necessary information
2. **Garbage Collection**: Clean up completed/failed tasks
3. **Disk Usage Monitoring**: Track and limit persistent storage

### Caching

Implement caching for frequently accessed data:

1. **Configuration Caching**: Cache loaded configuration
2. **Tool Results Caching**: Cache results of deterministic tools
3. **LLM Response Caching**: Consider caching for identical inputs

## Contributing Guidelines

### Code Review Process

1. Pull requests require at least one reviewer approval
2. Automated tests must pass
3. Code must follow style guidelines
4. Documentation must be updated

### Pull Request Template

```markdown
## Description
<!-- Describe the changes and the motivation -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (please describe)

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Error handling implemented
```

### Documentation Requirements

- Update relevant documentation for new features
- Include docstrings for all public methods and classes
- Provide examples for new functionality
- Document configuration changes

### Testing Expectations

- Unit tests for all new functionality
- Integration tests for component interactions
- Edge cases and error scenarios covered
- No decrease in code coverage

## Troubleshooting Common Issues

### API Connection Issues

**Symptoms:**
- API requests failing with connection errors
- Timeouts when generating responses

**Potential Solutions:**
1. Verify API key is set correctly
2. Check API endpoint URL is correct
3. Ensure network connectivity
4. Verify service status with Anthropic

### Configuration Problems

**Symptoms:**
- Application fails during startup with configuration errors
- Unexpected behavior due to misconfiguration

**Potential Solutions:**
1. Check environment variables are set correctly
2. Verify configuration file format
3. Use configuration validation methods
4. Check directory permissions

### Tool Execution Failures

**Symptoms:**
- Tools failing with execution errors
- Missing tool dependencies

**Potential Solutions:**
1. Check tool parameters are correct
2. Verify tool dependencies are registered
3. Review tool error messages for specific issues
4. Check permissions for file operations

### Background Service Issues

**Symptoms:**
- Tasks stuck in pending or running state
- Background service not processing tasks

**Potential Solutions:**
1. Verify background service is running
2. Check file permissions in task directories
3. Review service logs for errors
4. Restart background service if necessary

## Advanced Usage Patterns

### Multi-Tool Composition

Create complex workflows by composing multiple tools:

```python
# Pseudocode for multi-tool composition
def process_document(document_text):
    # Extract entities
    entities = tool_repository.execute_tool(
        "extraction_tool",
        text=document_text,
        extraction_type="entities"
    )
    
    # Categorize entities
    categorized = tool_repository.execute_tool(
        "classification_tool",
        items=entities,
        categories=["person", "organization", "location"]
    )
    
    # Store results
    tool_repository.execute_tool(
        "persistence_tool",
        operation="set_data",
        file_name="processed_entities.json",
        data=categorized
    )
    
    return categorized
```

### Custom System Prompt Strategies

Tailor system prompts for specific use cases:

1. **Specialized Agent Roles**: Create domain-specific agents
2. **Task-Specific Instructions**: Optimize prompts for particular tasks
3. **Dynamic Prompts**: Modify prompts based on conversation state

Example prompt pattern:
```
You are a specialized agent focusing on {domain}.
Your primary task is to {main_task}.
When providing information, prioritize {priority_aspect}.
Use the following tools when appropriate: {available_tools}.
```

### Advanced Background Processing

Patterns for complex background task management:

1. **Task Dependencies**: Create workflows with dependent tasks
2. **Priority Queues**: Implement task prioritization
3. **Resource Allocation**: Limit concurrent tasks based on resource usage
4. **Progress Reporting**: Provide real-time progress updates

### Memory Optimization Techniques

Advanced strategies for memory management:

1. **Conversation Summarization**: Dynamically summarize conversation history
2. **Selective Context**: Include only relevant parts of history based on current query
3. **Information Extraction**: Extract and store key information separate from full context
4. **Memory Hierarchies**: Implement short-term and long-term memory structures

## Contribution Best Practices

These principles guide effective contributions to the project:

### Problem Diagnosis

1. **Root Cause Analysis**: Focus on underlying issues rather than symptoms
   - Trace errors to their source using logs and stack traces
   - Consider system interactions that might contribute to issues
   - Test hypotheses systematically

2. **Context Gathering**: Understand the surrounding code before making changes
   - Review related files and dependencies
   - Understand the design patterns in use
   - Consider the historical context of the code

### Implementation Approach

1. **Minimal Changes**: Prefer targeted edits over large refactors
   - Limit scope to the specific issue
   - Avoid changing interfaces unless necessary
   - Make the smallest change that solves the problem

2. **Pattern Consistency**: Follow established patterns in the codebase
   - Match code style and approach
   - Use existing abstractions when available
   - Maintain architectural boundaries

3. **Step-by-Step Testing**: Verify changes incrementally
   - Test each logical change separately
   - Add tests before implementing changes when possible
   - Ensure tests cover the specific issue being addressed

### Code Quality

1. **Interface Preservation**: Maintain backwards compatibility
   - Preserve function signatures when possible
   - Add deprecation warnings before removing functionality
   - Document interface changes thoroughly

2. **Error Handling**: Implement comprehensive error management
   - Use appropriate exception types
   - Include contextual information in error messages
   - Handle edge cases explicitly

3. **Dependency Management**: Be cautious with dependencies
   - Prefer standard library solutions when possible
   - Justify new dependencies with specific benefits
   - Consider the maintenance burden of dependencies

### Review and Reflection

1. **Self-Review**: Critically evaluate your own changes
   - Review diffs before submitting
   - Question assumptions in your implementation
   - Consider potential failure modes

2. **Documentation**: Update documentation to reflect changes
   - Update inline documentation
   - Revise relevant developer documentation
   - Include examples for new functionality

3. **Knowledge Transfer**: Share insights from your work
   - Document non-obvious decisions
   - Update contribution guidelines with new learnings
   - Share patterns that were effective