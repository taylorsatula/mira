"""
Task manager package for scheduling and chaining automated tasks.

This package provides a system for scheduling tasks, executing chains of tasks,
and managing notifications related to automated task execution.
"""

# Make task scheduler components available at package level
from task_manager.task_scheduler import TaskScheduler, TaskFrequency, TaskStatus
from task_manager.scheduled_task import ScheduledTask, ExecutionMode
from task_manager.task_notification import TaskNotification, NotificationManager

# Make chain components available at package level
from task_manager.task_chain import TaskChain, ChainStep, ChainStatus, ErrorPolicy
from task_manager.chain_execution import ChainExecution, StepExecution
from task_manager.chain_executor import ChainExecutor
from task_manager.chain_template import TemplateEngine