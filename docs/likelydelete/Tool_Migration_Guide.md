# Tool Migration Guide

This guide explains how to migrate existing tools to use the new registry-based configuration system.

## Overview

The botwithmemory project has implemented a new tool configuration system that enables true drag-and-drop functionality. This guide provides step-by-step instructions for migrating existing tools to the new pattern.

## Migration Steps

For each tool in the `tools/` directory, follow these steps:

1. **Import the Registry**

   Add the registry import at the top of your tool file:

   ```python
   from config.registry import registry
   ```

2. **Define Configuration Class in Tool Module**

   Move the tool's configuration class from `config/config.py` into the tool module:

   ```python
   from pydantic import BaseModel, Field

   class MyToolConfig(BaseModel):
       """Configuration for my_tool."""
       enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
       # Add any other configuration fields your tool needs
       api_key: str = Field(default="", description="API key for external service")
       timeout: int = Field(default=30, description="Operation timeout in seconds")
   ```

3. **Register Configuration with Registry**

   After defining the configuration class, register it with the registry:

   ```python
   # Register with registry
   registry.register("my_tool", MyToolConfig)
   ```

4. **Use Deferred Config Import in Run Method**

   Update your tool's run method to use deferred config import:

   ```python
   def run(self, **params):
       # Import config when needed (avoids circular imports)
       from config import config
       
       # Access tool configuration
       timeout = config.my_tool.timeout
       
       # Rest of your implementation
       # ...
   ```

## Example: Before and After

### Before Migration

```python
# In config/config.py
class EmailToolConfig(BaseModel):
    """Configuration for email_tool."""
    enabled: bool = Field(default=True)
    # Other fields...

# In tools/email_tool.py
from config import config

class EmailTool(Tool):
    # ...
    
    def run(self, **params):
        # Access configuration
        settings = config.email_tool
        # ...
```

### After Migration

```python
# In tools/email_tool.py
from pydantic import BaseModel, Field
from config.registry import registry

class EmailToolConfig(BaseModel):
    """Configuration for email_tool."""
    enabled: bool = Field(default=True)
    # Other fields...

# Register with registry
registry.register("email_tool", EmailToolConfig)

class EmailTool(Tool):
    # ...
    
    def run(self, **params):
        # Deferred import
        from config import config
        
        # Access configuration
        settings = config.email_tool
        # ...
```

## Tools to Migrate

The following tools need to be migrated to the new configuration system:

- calendar_tool.py
- customerdatabase_tool.py
- email_tool.py
- http_tool.py
- kasa_tool.py
- maps_tool.py
- questionnaire_tool.py
- reminder_tool.py
- square_tool.py
- translation_tool.py
- tool_feedback.py

## Testing Your Migration

After updating a tool, verify that it works correctly by:

1. Running the appropriate test for the tool (`pytest tests/test_your_tool.py`)
2. Checking that the tool's configuration is accessible via the registry
3. Ensuring that the tool can import and use its configuration at runtime

## Additional Notes

- For tools that don't have configuration, you can still register a basic configuration class with just the `enabled` field
- Make sure to use the same name for registration as the tool's `name` attribute
- The `sample_tool.py` has already been updated to the new pattern and can be used as a reference

For more details on the new configuration system, see [Tool_Configuration_System.md](Tool_Configuration_System.md).