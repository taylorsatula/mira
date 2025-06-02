# Automation Template Generator Guide

This document outlines how to convert semi-structured natural language descriptions into valid JSON automation templates.

## Request Format

When a user wants to create a new automation, they can use this simplified format:

```
AUTOMATION REQUEST

NAME: [Descriptive name]
TRIGGER: [When/how often to run]
STEPS:
1. [First step in plain language]
2. [Second step]
3. If [condition]:
   - [Action if true]
   Else:
   - [Action if false]
4. [Additional steps]
```

## Conversion Guidelines

When processing these requests:

1. Use the exact JSON structure from `/task_manager/automations/smart_home_advanced.json` as the base template
2. Map the user's NAME to the "name" and generate a kebab-case "id" field
3. Parse the TRIGGER to determine "frequency", "scheduled_time", and "timezone" values
4. Convert each STEP to a valid step object in the "steps" array
5. For conditional logic, create appropriate "condition_type", "condition_data_key", "condition_operator", and "condition_value" settings
6. Ensure all required fields are populated with sensible defaults when not specified
7. Maintain proper positioning and task relationships in the step sequence

## Smart Home Device References

When working with device names and attributes:

1. Check `/data/tools/kasa_tool/cache/devices.json` for the list of available devices
2. Use exact device names from this file
3. Verify that requested operations are valid for the specific device type
4. Include proper error handling steps for device unavailability

## Example Conversion

### Input Format:
```
AUTOMATION REQUEST

NAME: Evening Comfort Setup
TRIGGER: 6:00PM every Wednesday
STEPS:
1. Set dripping_lantern_top brightness to 100 using kasa_tool
2. Check weather for "Huntsville, AL" using weather_tool
3. If temperature is between 60-70:
   - Set fan_state to "cool" using ecobee_tool
   Else:
   - Set fan_state to "off" using ecobee_tool
4. Send report to me@example.com
```

### Resulting JSON Structure:
```json
{
  "id": "evening-comfort-setup",
  "name": "Evening Comfort Setup",
  "description": "Sets up comfortable evening environment based on weather conditions",
  "type": "sequence",
  "frequency": "weekly",
  "scheduled_time": "18:00:00",
  "timezone": "America/Chicago",
  "status": "active",
  "error_policy": "continue",
  "timeout": 1800,
  "max_executions": null,
  "user_id": "system",
  
  "steps": [
    {
      "id": "set-lamp-brightness",
      "name": "Set Lamp Brightness",
      "position": 1,
      "description": "Set the dripping lantern top to full brightness",
      "execution_mode": "direct",
      "tool_name": "kasa_tool",
      "operation": "set_brightness",
      "parameters": {
        "device_id": "dripping_lantern_top",
        "brightness": 100
      },
      "output_key": "lamp_brightness_result",
      "condition_type": "always",
      "max_retries": 2,
      "retry_delay": 30
    },
    ...
  ]
}
```

## Validation Checks

Before returning the final JSON:

1. Verify all required fields are present
2. Check that device names match those in the device cache
3. Ensure all conditional logic has proper output_key references
4. Validate JSON structure is correct and follows template pattern
5. Add appropriate error handling step(s) if not explicitly requested

## Additional Considerations

- Default timezone to "America/Chicago" unless specified
- Set reasonable retry settings for smart home operations
- Add appropriate descriptions for each step based on its function
- Ensure the automation has a clear error handling strategy
- Use consistent naming conventions for step IDs and output keys