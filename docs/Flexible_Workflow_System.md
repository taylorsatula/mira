# Flexible Workflow System

The Mira Flexible Workflow System provides an intelligent, adaptive approach to multi-step processes that avoids rigid linear progressions in favor of a more natural conversational experience.

## Key Features

### Prerequisite-Based Progression
- Steps are defined by their dependencies, not sequential order
- Multiple paths can be taken through a workflow
- Steps become available when their prerequisites are completed

### Data-Driven Workflows
- Tracks what information has been collected
- Updates available steps based on gathered data
- Auto-completes steps when data is already known

### Initial Data Extraction
- Uses LLM to extract structured data from the triggering message
- Skips steps that are satisfied by the initial message
- Reduces unnecessary back-and-forth with users

### Conditional Step Visibility
- Steps can be hidden or shown based on collected information
- Optional steps can be skipped when not needed
- Workflows adapt to user input dynamically

### Contextual Tool Access
- Only enables tools relevant to currently available steps
- Avoids overwhelming the LLM with unnecessary tool options
- Updates tool access as workflow progresses

### Natural Conversation Flow
- Users can move back to previously completed steps
- Multiple steps can be available at once
- Interface surfaces options rather than forcing a specific path

## Workflow Definition Example

```json
{
  "id": "smart_home_automation",
  "name": "Smart Home Automation Setup",
  "version": 2,
  "description": "Process for creating complex home automation routines",
  
  // Define preferred starting points (steps with no prerequisites are fallbacks)
  "entry_points": ["automation_type"],
  
  "steps": {
    "automation_type": {
      "id": "automation_type",
      "description": "Determine automation type",
      "tools": [],
      "guidance": "Ask about automation type: time-based, event-based, or sensor-based.",
      "prerequisites": [],
      "optional": false,
      "provides_data": ["automation_type", "automation_name"],
      "next_suggestions": ["schedule_info", "trigger_info", "sensor_info"]
    },
    
    "schedule_info": {
      "id": "schedule_info",
      "description": "Collect schedule information",
      "tools": ["calendar_tool"],
      "guidance": "Collect schedule details: one-time or recurring, days, times, etc.",
      "prerequisites": ["automation_type"],
      "optional": true,
      "provides_data": ["schedule_type", "run_time", "run_days"],
      "next_suggestions": ["action_selection"],
      // Only show this step for time-based automations
      "condition": "workflow_data.automation_type === 'time-based'"
    },
    
    // Additional steps...
  },
  
  // Define completion requirements
  "completion_requirements": {
    "required_steps": ["save_automation"],
    "required_data": ["automation_id", "success_message"]
  },
  
  // Define data schema for this workflow
  "data_schema": {
    "automation_type": {
      "type": "string", 
      "description": "Type of automation (time-based, event-based, sensor-based)"
    },
    // Additional data fields...
  }
}
```

## Implementation Details

### Workflow Manager
The `WorkflowManager` class coordinates workflow execution:
- Loads workflow definitions from JSON files
- Tracks completed and available steps
- Manages workflow state and data collection
- Integrates with the conversation system

### LLM Integration
- System prompt extensions guide the LLM through available steps
- Navigation commands allow the LLM to control workflow progression
- Data extraction uses LLM to parse initial user requests

### Conversation Flow
1. User message triggers workflow detection
2. LLM analyzes message to extract initial data
3. Workflow initializes with available steps
4. LLM guides user through steps, adapting based on inputs
5. Workflow tracks progress and completes when requirements are met

## Benefits

1. **Enhanced User Experience**
   - Conversations feel natural rather than scripted
   - Avoids repetitive questions when information is already provided
   - Adapts to user's preferred order of providing information

2. **Efficiency Improvements**
   - Skips unnecessary steps automatically
   - Extracts data from initial messages
   - Reduces conversation turns for common tasks

3. **Development Flexibility**
   - Workflows are defined in JSON, not hardcoded
   - New workflow steps can be added without changing existing code
   - Conditional logic allows for complex branching processes

4. **Context Management**
   - Only relevant tools are enabled at each step
   - System prompt provides appropriate guidance
   - Progress visualization helps users understand the process