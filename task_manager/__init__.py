"""
Task manager package for scheduling and executing automated tasks and sequences.

This package provides a unified system for managing automated tasks, including
both simple recurring tasks and multi-step sequences, with a common scheduling system.
"""

# Make automation system components available at package level
from task_manager.automation import (
    Automation, AutomationType, AutomationStatus, 
    ExecutionMode, TaskFrequency, AutomationStep, 
    AutomationExecution, StepExecution, ErrorPolicy,
    ConditionType, TriggerType, ExecutionStatus
)
from task_manager.automation_engine import (
    get_automation_engine, initialize_automation_engine, 
    AutomationEngine, TemplateEngine
)