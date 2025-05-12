# Working Memory System

## Overview

The Working Memory system is a centralized solution for managing dynamic content in system prompts. It provides a standardized API for various components to add, remove, and update content that should be included in prompts during conversation turns.

This document explains how the system works and provides guidance on extending it with new functionality.

## Core Architecture

### Design Principles

1. **Managers Own Their Content**: Large subsystems like WorkflowManager and ToolRepository manage their own working memory content.
2. **Standalone Trinkets**: Smaller, standalone functionality (like time information) is implemented as "trinkets" within working_memory.py.
3. **Centralized Interface**: All dynamic content is accessed through a single `get_prompt_content()` method.
4. **Category Organization**: Content is organized by categories for easy management.

### Key Components

#### WorkingMemory Class

The `WorkingMemory` class (in working_memory.py) is the central component that:
- Stores content items with categories
- Provides methods to add, remove, and retrieve content
- Handles registration and updating of manager components

```python
working_memory = WorkingMemory()
item_id = working_memory.add("Current time is 2023-01-01", "datetime")
all_content = working_memory.get_prompt_content()
```

#### Memory Trinkets

Trinkets are utility classes that manage specific types of standalone memory content:

- **TimeManager**: Updates current date and time information
- **UserInfoManager**: Loads user information from files
- **SystemStatusManager**: Tracks system status and notifications

#### Manager Integration

Larger subsystems like `WorkflowManager` and `ToolRepository` integrate with working memory by:
1. Accepting a working_memory instance during initialization
2. Implementing an `update_working_memory()` method
3. Registering with working_memory to receive update calls

## Usage Guide

### Adding Content to Working Memory

Content can be added directly or through a manager/trinket:

```python
# Direct addition
item_id = working_memory.add("Content text", "my_category")

# Through a trinket
time_manager = TimeManager(working_memory)
time_manager.update_datetime_info()  # Adds datetime info to working memory
```

### Removing Content

Content can be removed by ID or category:

```python
# Remove by ID
working_memory.remove(item_id)

# Remove by category
num_removed = working_memory.remove_by_category("my_category")
```

### Retrieving Content

To get all content formatted for inclusion in a system prompt:

```python
all_content = working_memory.get_prompt_content()
```

To get specific items by category:

```python
items = working_memory.get_items_by_category("datetime")
```

### Updating Before Responses

Before generating each response, update all dynamic content:

```python
# Update standalone trinkets
time_manager.update_datetime_info()

# Update all registered managers
working_memory.update_all_managers()
```

## Implementing New Functionality

### Creating a New Trinket

For standalone functionality, create a new trinket class in working_memory.py:

```python
class MyNewTrinket:
    """Manager for my new type of content."""

    def __init__(self, working_memory: WorkingMemory):
        """Initialize with working memory reference."""
        self.working_memory = working_memory
        self._content_id = None  # Track content ID for updates
        
        # Add initial content
        self.update_content()
        
    def update_content(self) -> None:
        """Update content in working memory."""
        # Remove existing content if present
        if self._content_id:
            self.working_memory.remove(self._content_id)
            
        # Create new content
        content = f"# My Content\nGenerated content here"
        
        # Add to working memory
        self._content_id = self.working_memory.add(
            content=content,
            category="my_category"
        )
```

### Integrating a Manager

For larger subsystems that should manage their own content:

1. Add working_memory parameter to the constructor:

```python
def __init__(self, other_params, working_memory=None):
    self.working_memory = working_memory
```

2. Implement update_working_memory method:

```python
def update_working_memory(self) -> None:
    """Update this manager's content in working memory."""
    if not self.working_memory:
        return
        
    # Implementation goes here to add/update content
```

3. Register the manager in main.py:

```python
working_memory.register_manager(my_manager)
```

### Best Practices

1. **Always Track Content IDs**: Keep track of added content IDs to allow for clean updates
2. **Use Descriptive Categories**: Choose clear category names to help with debugging
3. **Check Working Memory Existence**: Always check if working_memory is available before using it
4. **Consider Caching**: For expensive content generation, consider caching to avoid unnecessary updates
5. **Keep Content Focused**: Each content item should serve a specific purpose

## Example: Weather Information Trinket

Here's a complete example of adding a weather information trinket:

```python
class WeatherInfoManager:
    """Manager for weather information in system prompts."""

    def __init__(self, working_memory: WorkingMemory, weather_service):
        """
        Initialize a new weather info manager.

        Args:
            working_memory: WorkingMemory instance for storing weather info
            weather_service: Service for fetching weather data
        """
        self.working_memory = working_memory
        self.weather_service = weather_service
        self._weather_id = None
        self._last_update = None
        self._update_interval = 3600  # Update once per hour
        
        # Initial update
        self.update_weather_info()
        
    def update_weather_info(self, force=False) -> None:
        """
        Update weather information in working memory.
        
        Args:
            force: Force update even if the update interval hasn't elapsed
        """
        now = time.time()
        
        # Skip update if not forced and interval hasn't elapsed
        if not force and self._last_update and (now - self._last_update < self._update_interval):
            return
            
        # Remove existing weather info if present
        if self._weather_id:
            self.working_memory.remove(self._weather_id)
            
        try:
            # Get current weather data
            weather_data = self.weather_service.get_current_weather()
            
            # Format weather information
            weather_info = f"# Current Weather\n"
            weather_info += f"Location: {weather_data['location']}\n"
            weather_info += f"Temperature: {weather_data['temperature']}°C\n"
            weather_info += f"Conditions: {weather_data['conditions']}\n"
            
            # Add to working memory
            self._weather_id = self.working_memory.add(
                content=weather_info,
                category="weather"
            )
            
            # Update timestamp
            self._last_update = now
            
        except Exception as e:
            logger.error(f"Failed to update weather information: {e}")
```

## Debugging Tips

1. **Inspect Categories**: Check which categories are in use with `get_items_by_category()`
2. **Check for Duplicates**: Look for duplicate categories if content appears multiple times
3. **Log Content Updates**: Add debug logging to track when content is added/updated
4. **Track Content IDs**: Log IDs returned from `add()` to trace the lifecycle of content items

## Asyncio Integration

The Working Memory System now includes components for managing asyncio operations across automation tasks.

### Asyncio Utilities (`utils/async_utils.py`)

This module provides utilities for managing asyncio event loops in a thread-safe way:

- `get_or_create_event_loop()`: Creates or retrieves a thread-local event loop
- `run_coroutine(coro)`: Runs a coroutine using the thread's event loop
- `close_event_loop()`: Properly cleans up and closes the thread's event loop

### Asyncio Tool Base (`tools/asyncio_tool_base.py`)

A base class for tools that use asyncio, providing:

- Thread-local state management with `get_thread_data()`, `set_thread_data()`, etc.
- A common `run_async()` method to run coroutines
- A customizable `cleanup()` method for releasing resources

### Implementation in Automation Engine

The automation engine has been enhanced to:

- Initialize an event loop at the start of each automation execution
- Clean up tool resources when an automation completes
- Close the event loop when the automation thread terminates

### Using Asyncio in Tools

To create a tool that properly uses asyncio:

1. Inherit from `AsyncioToolBase`
2. Replace `asyncio.run()` calls with `self.run_async()`
3. Store device connections and other state using `self.set_thread_data()`
4. Implement the `cleanup()` method to properly release resources

Example with the Kasa tool:

```python
class KasaTool(AsyncioToolBase):
    # ...
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        # ...
        if operation == "discover_devices":
            return self.run_async(self._discover_devices(**kwargs))
        # ...

    @property
    def _device_instances(self):
        return self.get_thread_data('device_instances', {})

    def cleanup(self) -> None:
        # Clear device instances to release resources
        self.set_thread_data('device_instances', {})
        super().cleanup()
```

### Best Practices for Asyncio Tools

1. **Use AsyncioToolBase**: Inherit from AsyncioToolBase for any tool using asyncio
2. **Replace asyncio.run()**: Always use `self.run_async()` instead of `asyncio.run()`
3. **Store Thread-Local State**: Use `self.set_thread_data()` for caching connections
4. **Implement Cleanup**: Always implement `cleanup()` to release resources
5. **Reuse Connections**: Cache and reuse connections throughout automation execution