# Tool Configuration System

This document explains the Tool Configuration System implemented in this project, which enables true drag-and-drop functionality for tools.

## Overview

The Tool Configuration System uses a registry-first approach to enable drag-and-drop tool functionality without circular imports. This design allows tools to be added to the project by simply copying them into the tools directory - no additional configuration steps are required.

## Key Components

### 1. Configuration Registry

The registry (`config/registry.py`) serves as a centralized store for tool configuration classes. It is initialized first, before any other imports, and provides a bridge between tools and the configuration system.

```python
# Import the registry
from config.registry import registry

# Register a tool configuration
registry.register("my_tool", MyToolConfig)

# Get a tool configuration
config_class = registry.get("my_tool")
```

### 2. Tool Configuration Registration

Tools register their configuration classes with the registry during module import. This ensures that the configuration is available before the tool is used.

```python
# In tools/my_tool.py
from pydantic import BaseModel, Field
from config.registry import registry

# Define configuration class
class MyToolConfig(BaseModel):
    enabled: bool = Field(default=True)
    api_key: str = Field(default="")
    timeout: int = Field(default=30)

# Register with registry
registry.register("my_tool", MyToolConfig)
```

### 3. Dynamic Configuration Access

The configuration system accesses these registered configurations through the registry, without importing the tools directly.

```python
# Access tool configuration
from config import config
api_key = config.my_tool.api_key
```

## How It Works

### Initialization Sequence

1. **Registry First**: The registry is initialized when `config/__init__.py` is imported
2. **Tool Registration**: Tools register themselves with the registry when they're imported
3. **Config Access**: The configuration system accesses the registry to retrieve tool configurations
4. **Lazy Instantiation**: Tool configuration instances are created only when accessed

### Avoiding Circular Imports

The design avoids circular imports by:

1. Initializing the registry before any other imports
2. Having tools import the registry directly, not the config
3. Having config access tool configurations through the registry, not by importing tools
4. Using deferred imports for config when needed by tools at runtime

## Creating a New Tool with Configuration

To create a new tool with custom configuration:

1. Define a configuration class using Pydantic:

```python
# tools/my_new_tool.py
from pydantic import BaseModel, Field
from config.registry import registry

class MyNewToolConfig(BaseModel):
    """Configuration for my new tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    api_key: str = Field(default="", description="API key for external service")
    timeout: int = Field(default=30, description="Operation timeout in seconds")
    
# Register with registry
registry.register("my_new_tool", MyNewToolConfig)
```

2. Create your tool class:

```python
class MyNewTool(Tool):
    name = "my_new_tool"
    description = "Description of what this tool does"
    
    def run(self, **params):
        # Access configuration
        from config import config
        api_key = config.my_new_tool.api_key
        timeout = config.my_new_tool.timeout
        
        # Use configuration in implementation
        # ...
```

3. That's it! No additional steps are needed.

## Benefits of this Approach

1. **True Drag-and-Drop**: Add tools by simply copying them to the tools directory
2. **No Manual Steps**: No configuration files need to be edited
3. **No Script Execution**: No scripts need to be run to update configuration
4. **Type Safety**: Full Pydantic model validation for tool configurations
5. **IDE Support**: Excellent IDE support with type hints and auto-completion
6. **No Circular Imports**: The design avoids circular dependencies

## Examples

See `tools/sample_tool.py` for a complete example of a tool with custom configuration.