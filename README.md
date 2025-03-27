# AI Agent System

A minimalist Python foundation for building AI agents with a clean, modular architecture.

## Overview

This system provides a foundation for creating AI agents with tool-based capabilities. It abstracts away the complexities of API communication, tool management, and conversation handling, allowing you to focus on building intelligent agents.

Key features:
- Clean, modular architecture with clear separation of concerns
- Anthropic Claude API integration with proper error handling and rate limiting
- Extensible tool system for adding custom capabilities
- Conversation management with history tracking
- Persistent storage with dedicated data and file operations
- Background task processing with reliable storage of results
- Standardized error handling with context managers

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-agent-system.git
   cd ai-agent-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the provided template:
   ```
   cp .env.template .env
   ```

5. Add your Anthropic API key to the `.env` file:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Quick Start

1. Run the agent in interactive mode:
   ```
   python main.py
   ```

2. Start chatting with the agent! Type your messages and press Enter.

3. Special commands in interactive mode:
   - `save`: Save the current conversation
   - `clear`: Clear the conversation history
   - `exit` or `quit`: End the session

## Configuration

The system uses a centralized configuration system with Pydantic for validation. Configuration can be set in two ways:
1. Environment variables in `.env` file
2. Command-line arguments

Environment variables should be prefixed with `AGENT_`. For nested settings, use double underscores `__`. For example:
```
AGENT_SYSTEM__LOG_LEVEL=DEBUG
AGENT_API__MODEL=claude-3-7-sonnet-20250219
AGENT_API__TEMPERATURE=0.8
```

All configuration settings have sensible defaults, so you only need to specify values you want to override.

Command-line options:
- `--config` or `-c`: Path to a JSON configuration file
- `--conversation` or `-id`: Conversation ID to load
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Basic Usage Examples

### Creating a Simple Agent

```python
from config import config
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from conversation import Conversation

# The config instance is already initialized with sensible defaults
print(f"Using model: {config.api.model}")
print(f"Log level: {config.system.log_level}")

# Initialize components
llm_bridge = LLMBridge()
tool_repo = ToolRepository()

# Create a conversation
conversation = Conversation(
    system_prompt="You are a helpful assistant that specializes in weather information.",
    llm_bridge=llm_bridge,
    tool_repo=tool_repo
)

# Generate a response
response = conversation.generate_response("What's the weather like in New York?")
print(response)
```

### Adding a Custom Tool

Create a new file in the `tools` directory:

```python
# tools/custom_tool.py
from tools.repo import Tool
from errors import error_context, ToolError, ErrorCode

class MyCustomTool(Tool):
    name = "my_custom_tool"
    description = "A custom tool that does something useful"
    
    def run(self, parameter1: str, parameter2: int = 0) -> dict:
        """
        Run the custom tool.
        
        Args:
            parameter1: The first parameter
            parameter2: The second parameter (default: 0)
            
        Returns:
            Result dictionary
        """
        # Use the centralized error context manager for error handling
        with error_context(
            component_name=self.name,
            operation="processing parameters",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Tool implementation here
            result = {
                "parameter1": parameter1,
                "parameter2": parameter2,
                "processed": f"{parameter1}-{parameter2}"
            }
            return result
```

The tool will be automatically discovered and registered when the system starts.

### Using the Persistence Tool

The system provides a flexible persistence tool for storing and retrieving data:

```python
from tools.repo import ToolRepository

# Initialize the tool repository
tool_repo = ToolRepository()

# Get the persistence tool
persistence_tool = tool_repo.get_tool("persistence")

# Store data (key-value operations)
result = persistence_tool.run(
    operation="set_data",
    location="user_preferences.json",
    key="theme",
    value="dark"
)

# Retrieve data
result = persistence_tool.run(
    operation="get_data",
    location="user_preferences.json",
    key="theme"
)
theme = result["value"]  # "dark"

# Store an entire file
result = persistence_tool.run(
    operation="set_file",
    location="user_data.json",
    data={"name": "John", "preferences": {"theme": "dark"}}
)

# Retrieve an entire file
result = persistence_tool.run(
    operation="get_file",
    location="user_data.json"
)
user_data = result["value"]
```

### Working with Asynchronous Tasks

```python
# Schedule a background task
task_result = tool_repo.invoke_tool("schedule_async_task", {
    "description": "Analyze data",
    "task_prompt": "Analyze the provided data and save the results.",
    "notify_on_completion": True
})
task_id = task_result["task_id"]

# Check task status
status_result = tool_repo.invoke_tool("check_async_task", {
    "task_id": task_id
})

# Retrieve task results
if status_result["status"] == "completed":
    result = persistence_tool.run(
        operation="get_file",
        location=f"async_results/{task_id}.json"
    )
    analysis_data = result["value"]
```

### Saving and Loading Conversations

```python
from crud import FileOperations
from conversation import Conversation

# Initialize file operations
file_ops = FileOperations("data")

# Save a conversation
conversation_data = conversation.to_dict()
file_ops.write(f"conversation_{conversation.conversation_id}", conversation_data)

# Load a conversation
loaded_data = file_ops.read(f"conversation_{conversation_id}")
loaded_conversation = Conversation.from_dict(loaded_data)
```

### Handling External Stimuli

The system can process external triggers using the stimulus handling system:

```python
from stimuli import StimulusHandler, StimulusType, Stimulus
from conversation import Conversation

# Initialize components
conversation = Conversation(system_prompt="You are a helpful assistant.")
stimulus_handler = StimulusHandler()

# Method 1: Simple integration - attach a conversation to the stimulus handler
stimulus_handler.attach_conversation(conversation)

# Process an external notification (automatically sent to the conversation)
stimulus_handler.create_and_process(
    stimulus_type=StimulusType.NOTIFICATION,
    content="Battery is low (15%)",
    source="system_monitor",
    metadata={"urgency": "medium"}
)

# Check the conversation history - it will contain the stimulus as a user message
print(conversation.messages[-1].content)  # Shows: [NOTIFICATION from system_monitor]: Battery is low (15%)

# Method 2: Advanced integration - with response handling
def process_response(stimulus: Stimulus, response: str) -> None:
    print(f"Response to {stimulus.type.value}: {response}")
    # Take action based on response
    if "urgent" in response.lower():
        print("Taking urgent action!")

# Process a stimulus and automatically handle the response
stimulus_handler.process_stimulus_with_conversation(
    Stimulus(
        type=StimulusType.SENSOR,
        content="Temperature is 95°C",
        source="thermal_sensor"
    ),
    response_callback=process_response
)

# Method 3: Custom handlers for specific stimulus types
def handle_api_events(stimulus: Stimulus) -> None:
    if "critical" in stimulus.content.lower():
        # Process directly without conversation
        print(f"Critical API event: {stimulus.content}")
    else:
        # Add to conversation for AI processing
        process_stimulus(stimulus, conversation)

# Register a custom handler for API events
stimulus_handler.register_handler(StimulusType.API, handle_api_events)
```

## System Architecture

The system consists of the following main components:

- **`main.py`**: Central entry point and control flow
- **`config.py`**: Configuration management and defaults
- **`errors.py`**: Error handling and exception hierarchy
- **`api/llm_bridge.py`**: Anthropic API communication
- **`tools/repo.py`**: Tool interface and registry
- **`tools/*.py`**: Individual tool implementations
- **`crud.py`**: File operations and persistence
- **`conversation.py`**: Conversation and context management
- **`stimuli.py`**: External stimulus handling

Data flows through the system as follows:

1. User input is received through the CLI or external stimuli
2. The conversation manager adds the input to the conversation history
3. External stimuli can be processed by the stimulus handler and added to the context
4. The LLM bridge sends the conversation to the Anthropic API
5. If tool calls are in the response, the tool repository executes them
6. Tool results are added to the conversation and another LLM call is made
7. The final response is displayed to the user
8. The conversation can be saved to a file for persistence

For more detailed technical information, see [DEVELOPERS.md](DEVELOPERS.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.