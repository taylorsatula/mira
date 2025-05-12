# JSON Automation Format

This directory contains JSON-based automation definitions that can be loaded and executed by the automation engine. This approach enables developers to easily create and modify automations without programming.

## Automation Structure

Each `.json` file in this directory represents a single automation. The file name should be descriptive and match the `id` field of the automation (with `.json` extension).

### Common Fields

All automations include these common fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the automation |
| `name` | string | Human-readable name |
| `description` | string | Detailed description |
| `type` | string | Either `"simple_task"` or `"sequence"` |
| `frequency` | string | One of: `"once"`, `"minutely"`, `"hourly"`, `"daily"`, `"weekly"`, `"monthly"`, `"custom"` |
| `scheduled_time` | string | Time to run in HH:MM:SS format (can be null for immediate execution) |
| `timezone` | string | IANA timezone name (e.g., "America/New_York") |
| `status` | string | One of: `"active"`, `"paused"`, `"completed"`, `"failed"`, `"archived"` |
| `max_executions` | integer | Maximum number of times to run (null = unlimited) |
| `user_id` | string | ID of the user who created the automation |

### Simple Task Fields

When `type` is `"simple_task"`, these additional fields are required:

| Field | Type | Description |
|-------|------|-------------|
| `execution_mode` | string | Either `"direct"` or `"orchestrated"` |
| `tool_name` | string | Name of the tool to execute (for `direct` mode) |
| `operation` | string | Tool operation to execute (for `direct` mode) |
| `parameters` | object | Parameters for the operation |
| `task_description` | string | Description of the task (for `orchestrated` mode) |
| `available_tools` | array | List of tool names to make available (for `orchestrated` mode) |

### Sequence Fields

When `type` is `"sequence"`, these additional fields are required:

| Field | Type | Description |
|-------|------|-------------|
| `error_policy` | string | One of: `"stop"`, `"continue"`, `"retry"`, `"alternative"`, `"silent"` |
| `timeout` | integer | Timeout in seconds for the entire sequence |
| `steps` | array | Array of step objects defining the sequence |

### Step Structure

Each step in a sequence includes:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the step |
| `name` | string | Human-readable name |
| `position` | integer | Order in the sequence (starts at 1) |
| `description` | string | Detailed description |
| `execution_mode` | string | Either `"direct"` or `"orchestrated"` |
| `tool_name` | string | Name of the tool to execute (for `direct` mode) |
| `operation` | string | Tool operation to execute (for `direct` mode) |
| `task_description` | string | Description of the task (for `orchestrated` mode) |
| `available_tools` | array | List of tool names to make available (for `orchestrated` mode) |
| `parameters` | object | Parameters for the operation |
| `output_key` | string | Key to store the result under |
| `condition_type` | string | One of: `"always"`, `"if_success"`, `"if_failure"`, `"if_data"`, `"if_no_data"` |
| `condition_data_key` | string | Data key to check for condition (for data conditions) |
| `condition_value` | any | Value to compare against (for data conditions) |
| `condition_operator` | string | Operator for comparison (e.g., `"eq"`, `"gt"`, `"contains"`) |
| `error_policy` | string | Step-specific error policy (overrides sequence policy) |
| `max_retries` | integer | Maximum retry attempts on failure |
| `retry_delay` | integer | Seconds to wait between retries |
| `on_success_step_id` | string | ID of next step on success (overrides normal flow) |
| `on_failure_step_id` | string | ID of next step on failure |

## Parameter Templates

Parameters can include templates to reference data from previous steps or context:

- Use `{step_id.result}` to reference a step's result
- Use `{step_id.result.property}` to access specific properties
- For comparison operations, reference data with `{data_key}`

## Example Types

This directory includes several example automation types:

1. **Simple Task** (daily_weather_check.json): Single-operation automation that runs directly
2. **Multi-Step Sequence** (customer_followup_sequence.json): Series of operations that pass data
3. **Conditional Automation** (smart_home_conditional.json): Sequence with branching logic
4. **Scheduled Automation** (weekly_report_generator.json): Sequence with complex scheduling

## Creating New Automations

To create a new automation:

1. Copy an existing template closest to your needs
2. Modify the fields as required
3. Save with a descriptive filename matching the `id` field (with `.json` extension)
4. The automation engine will automatically load and execute it based on its schedule

## Available Tools

The following tools are available for use in automations:

- `weather_tool` - Weather forecasts and conditions
- `email_tool` - Sending and managing emails
- `customerdatabase_tool` - Customer data management
- `reminder_tool` - Setting and tracking reminders
- `kasa_tool` - Smart home device control
- `maps_tool` - Location and mapping features
- `translation_tool` - Text translation services

Refer to each tool's documentation for available operations and parameters.