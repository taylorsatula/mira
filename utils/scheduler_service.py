"""
Scheduler service utility for integrating task scheduling with the main application.

This module provides functions to initialize, start, and stop the task scheduler
as part of the main application lifecycle. The scheduler runs in a background thread
within the same process as the main application.
"""

import logging
import threading
from typing import Dict, Any, Optional

from config import config
from task_manager.task_scheduler import TaskScheduler
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from task_manager.task_notification import NotificationManager

# Global scheduler instance
_scheduler: Optional[TaskScheduler] = None
_scheduler_thread: Optional[threading.Thread] = None
_notification_manager: Optional[NotificationManager] = None

# Configure logger
logger = logging.getLogger(__name__)


def initialize_scheduler(
    tool_repo: Optional[ToolRepository] = None,
    llm_bridge: Optional[LLMBridge] = None
) -> Dict[str, Any]:
    """
    Initialize the task scheduler and notification manager.
    
    Args:
        tool_repo: Repository of available tools
        llm_bridge: LLM bridge for orchestrated tasks
        
    Returns:
        Dictionary containing scheduler components
    """
    global _scheduler, _notification_manager
    
    logger.info("Initializing task scheduler...")
    
    # Create scheduler instance
    _scheduler = TaskScheduler(
        tool_repo=tool_repo,
        llm_bridge=llm_bridge
    )
    
    # Create notification manager
    _notification_manager = NotificationManager()
    
    logger.info("Task scheduler initialized")
    
    return {
        'scheduler': _scheduler,
        'notification_manager': _notification_manager
    }


def start_scheduler() -> bool:
    """
    Start the scheduler in a background thread.
    
    Returns:
        True if started successfully, False otherwise
    """
    global _scheduler, _scheduler_thread
    
    if not _scheduler:
        logger.error("Cannot start scheduler: not initialized")
        return False
        
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("Scheduler is already running")
        return True
        
    # Create a dedicated thread for the scheduler
    _scheduler_thread = threading.Thread(
        target=_scheduler_worker,
        name="SchedulerThread",
        daemon=True  # Make the thread a daemon so it exits when the main process exits
    )
    
    # Start the thread
    _scheduler_thread.start()
    logger.info("Task scheduler started in background thread")
    
    return True


def stop_scheduler() -> bool:
    """
    Stop the scheduler.
    
    Returns:
        True if stopped successfully, False otherwise
    """
    global _scheduler
    
    if not _scheduler:
        logger.warning("Cannot stop scheduler: not initialized")
        return False
        
    # Stop the scheduler
    _scheduler.stop()
    logger.info("Task scheduler stopped")
    
    return True


def get_scheduler() -> Optional[TaskScheduler]:
    """
    Get the scheduler instance.
    
    Returns:
        The scheduler instance or None if not initialized
    """
    global _scheduler
    return _scheduler


def get_notification_manager() -> Optional[NotificationManager]:
    """
    Get the notification manager instance.
    
    Returns:
        The notification manager instance or None if not initialized
    """
    global _notification_manager
    return _notification_manager


def is_scheduler_running() -> bool:
    """
    Check if the scheduler is running.
    
    Returns:
        True if the scheduler is running, False otherwise
    """
    global _scheduler, _scheduler_thread
    
    if not _scheduler or not _scheduler_thread:
        return False
        
    return _scheduler_thread.is_alive() and _scheduler.running


def _scheduler_worker() -> None:
    """
    Worker function for the scheduler thread.
    
    This function starts the scheduler and keeps the thread alive
    until the scheduler is stopped.
    """
    global _scheduler
    
    if not _scheduler:
        logger.error("Cannot run scheduler worker: scheduler not initialized")
        return
        
    try:
        # Start the scheduler
        _scheduler.start()
        
        # Keep the thread alive while the scheduler is running
        while _scheduler.running:
            import time
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in scheduler worker: {e}", exc_info=True)
    finally:
        # Ensure the scheduler is stopped
        if _scheduler.running:
            _scheduler.stop()