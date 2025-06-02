# Task Scheduler System Guide

The Task Scheduler is a system that allows for the automatic execution of tasks at specified times without user intervention. It provides a way to schedule both simple direct tool executions and complex LLM-orchestrated tasks.

## Key Features

- **Schedule tasks to run at specific times** - one-time or recurring (minutely, hourly, daily, weekly, monthly)
- **Two execution modes**:
  - **Direct execution**: Execute a specific tool with predetermined parameters
  - **LLM-orchestrated execution**: Use the LLM to interpret a task description and decide which tools to use
- **Schedule management**: Create, view, update, delete, and execute tasks
- **Notification system**: Task results are stored and displayed in conversations
- **Timezone support**: Schedule tasks in any timezone

## Components

The Task Scheduler system consists of several components:

1. **ScheduledTask** - Database model for storing task definitions and state
2. **TaskScheduler** - Core scheduling and execution engine
3. **TaskNotification** - Database model for storing task execution results
4. **Scheduler Service** - Integration with the main application
5. **Scheduler Tool** - User interface for managing scheduled tasks

## Using the Scheduler Tool

You can use the `scheduler_tool` to create and manage scheduled tasks. Here are some examples:

### Creating a Direct Task

```json
{
  "operation": "create_task",
  "kwargs": {
    "name": "Daily Inventory Check",
    "frequency": "daily",
    "scheduled_time": "09:00:00",
    "execution_mode": "direct",
    "tool_name": "reminder_tool",
    "operation": "add_reminder",
    "parameters": {
      "title": "Check inventory levels", 
      "date": "today",
      "description": "Review stock levels for popular items"
    }
  }
}
```

### Creating an LLM-Orchestrated Task

```json
{
  "operation": "create_task",
  "kwargs": {
    "name": "Weekly Sales Analysis",
    "frequency": "weekly",
    "day_of_week": 0,  // Monday
    "scheduled_time": "08:00:00",
    "execution_mode": "orchestrated",
    "task_description": "Generate a sales report for the past week, analyze top-selling products, and email a summary to the sales team."
  }
}
```

### Listing Tasks

```json
{
  "operation": "get_tasks",
  "kwargs": {
    "status": "scheduled"
  }
}
```

### Getting Task Details

```json
{
  "operation": "get_task",
  "kwargs": {
    "task_id": "task_12345"
  }
}
```

### Updating a Task

```json
{
  "operation": "update_task",
  "kwargs": {
    "task_id": "task_12345",
    "scheduled_time": "10:00:00"
  }
}
```

### Executing a Task Immediately

```json
{
  "operation": "execute_task_now",
  "kwargs": {
    "task_id": "task_12345"
  }
}
```

### Cancelling a Task

```json
{
  "operation": "cancel_task",
  "kwargs": {
    "task_id": "task_12345"
  }
}
```

### Deleting a Task

```json
{
  "operation": "delete_task",
  "kwargs": {
    "task_id": "task_12345"
  }
}
```

## Task Frequencies

- **once**: Run once at the specified time
- **minutely**: Run every minute
- **hourly**: Run every hour
- **daily**: Run every day at the specified time
- **weekly**: Run every week on the specified day of week (0=Monday, 6=Sunday)
- **monthly**: Run every month on the specified day of month
- **custom**: Custom schedule (e.g., "every 3 days")

## Execution Modes

### Direct Execution

Direct execution mode allows you to specify exactly which tool to run with what parameters. This is useful for simple, well-defined tasks.

Required parameters:
- `tool_name`: The name of the tool to execute
- `operation`: The operation to perform
- `parameters`: Parameters to pass to the tool

### LLM-Orchestrated Execution

LLM-orchestrated execution mode allows you to describe a task in natural language and let the LLM decide which tools to use. This is useful for more complex tasks that might require multiple tool calls or conditional logic.

Required parameters:
- `task_description`: A natural language description of what the task should accomplish

Optional parameters:
- `task_prompt`: A custom system prompt for the LLM
- `available_tools`: List of tool names the LLM can use (if not specified, all available tools will be used)

## Task Notifications

When tasks execute, they generate notifications that are displayed in the conversation. These notifications include:

- The task name
- The execution status (success or failure)
- The execution result or error message

## Architecture

The Task Scheduler system is designed to be integrated with the main application:

1. **Integration**: The scheduler runs in a background thread within the main process
2. **Lifecycle**: It starts when the main application starts and stops when the main application stops
3. **Thread safety**: All operations are thread-safe
4. **Persistence**: Tasks and notifications are stored in the SQLite database

## Configuration

The Task Scheduler system can be configured in `config.py`:

```python
class SchedulerConfig(BaseModel):
    """Configuration for the task scheduler."""
    enabled: bool = Field(default=True, description="Whether the scheduler is enabled by default")
    check_interval: int = Field(default=60, description="Interval in seconds to check for scheduled tasks")
    log_level: str = Field(default="INFO", description="Logging level for the scheduler")
    max_concurrent_tasks: int = Field(default=5, description="Maximum number of tasks to run concurrently")
    task_timeout: int = Field(default=300, description="Default timeout for tasks in seconds")
    orchestration_model: str = Field(default="claude-3-haiku-20240307", 
                                   description="Model to use for orchestrated tasks")
    default_system_prompt: str = Field(default="You are a task automation assistant...",
                                     description="Default system prompt for orchestrated tasks")
```

The scheduler tool also has its own configuration:

```python
class SchedulerToolConfig(BaseModel):
    """Configuration for the scheduler_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    max_tasks_per_user: int = Field(default=20, description="Maximum number of tasks a user can create")
```

## Best Practices

1. **Use specific task names** that clearly indicate the task's purpose
2. **Provide detailed descriptions for orchestrated tasks** to ensure the LLM understands what to do
3. **Use the appropriate execution mode** for your task:
   - Direct execution for simple, predictable tasks
   - LLM-orchestrated execution for complex tasks requiring reasoning
4. **Consider timezone differences** when scheduling tasks
5. **Limit the number of tasks** to prevent resource contention
6. **Test tasks by executing them immediately** before scheduling them for recurring execution