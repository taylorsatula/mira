"""
Task automation controller for coordinating scheduled tasks and task chains.

This module provides functions to initialize, start, and stop both the task scheduler
and the task chain system as part of the main application lifecycle, ensuring
they work together with a unified notification system.
"""

import logging
from typing import Dict, Any, Optional, List

from config import config
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from task_manager.task_notification import NotificationManager

from utils import scheduler_service
from utils import chain_scheduler_service

# Configure logger
logger = logging.getLogger(__name__)


def initialize_systems(
    tool_repo: Optional[ToolRepository] = None,
    llm_bridge: Optional[LLMBridge] = None
) -> Dict[str, Any]:
    """
    Initialize all task scheduling and chain systems.
    
    This function initializes both the task scheduler and chain scheduler
    systems, ensuring they share the same notification manager.
    
    Args:
        tool_repo: Repository of available tools
        llm_bridge: LLM bridge for LLM-orchestrated execution
        
    Returns:
        Dictionary containing all initialized components
    """
    logger.info("Initializing task automation systems...")
    
    # Initialize scheduler first to get the notification manager
    scheduler_components = scheduler_service.initialize_scheduler(
        tool_repo=tool_repo,
        llm_bridge=llm_bridge
    )
    
    # Get notification manager to share across systems
    notification_manager = scheduler_components.get('notification_manager')
    
    # Initialize chain scheduler with shared notification manager
    chain_components = chain_scheduler_service.initialize_chain_scheduler(
        tool_repo=tool_repo,
        llm_bridge=llm_bridge,
        notification_manager=notification_manager
    )
    
    # Combine components
    components = {
        'scheduler': scheduler_components.get('scheduler'),
        'notification_manager': notification_manager,
        'chain_executor': chain_components.get('executor')
    }
    
    logger.info("All task automation systems initialized")
    
    return components


def start_systems() -> None:
    """
    Start all task scheduling and chain systems.
    
    This function starts both the task scheduler and chain scheduler
    background threads.
    """
    logger.info("Starting task automation systems...")
    
    # Start scheduler
    if config.get("scheduler", {}).get("enabled", True):
        scheduler_service.start_scheduler()
        logger.info("Task scheduler started")
    
    # Start chain scheduler
    if config.get("task_chain", {}).get("enabled", True):
        chain_scheduler_service.start_chain_scheduler()
        logger.info("Chain scheduler started")
    
    logger.info("All task automation systems started")


def stop_systems() -> None:
    """
    Stop all task scheduling and chain systems.
    
    This function stops both the task scheduler and chain scheduler
    background threads, ensuring clean shutdown.
    """
    logger.info("Stopping task automation systems...")
    
    # Stop chain scheduler
    if chain_scheduler_service.is_chain_scheduler_running():
        chain_scheduler_service.stop_chain_scheduler()
        logger.info("Chain scheduler stopped")
    
    # Stop scheduler
    if scheduler_service.is_scheduler_running():
        scheduler_service.stop_scheduler()
        logger.info("Task scheduler stopped")
    
    logger.info("All task automation systems stopped")


def get_notification_manager() -> Optional[NotificationManager]:
    """
    Get the shared notification manager instance.
    
    Returns:
        The notification manager instance or None if not initialized
    """
    return scheduler_service.get_notification_manager()


def check_pending_notifications(conversation_id: Optional[str] = None, limit: int = 5) -> List[Any]:
    """
    Check for pending task and chain notifications to display in a conversation.
    
    This function retrieves pending notifications and handles both task
    and chain notifications consistently.
    
    Args:
        conversation_id: ID of the conversation to check notifications for
        limit: Maximum number of notifications to return
        
    Returns:
        List of pending notifications
    """
    notification_manager = get_notification_manager()
    if not notification_manager:
        return []
        
    # Use the existing notification manager's method to get pending notifications
    # This will include both task and chain notifications
    return notification_manager.get_pending_notifications(
        conversation_id=conversation_id,
        limit=limit
    )


def display_and_mark_notifications(conversation, limit: int = 5) -> None:
    """
    Display pending task and chain notifications and mark them as displayed.
    
    This function checks for pending notifications, displays them,
    and marks them as displayed. It should be called at the beginning
    of each conversation turn.
    
    Args:
        conversation: The conversation to display notifications in
        limit: Maximum number of notifications to display
    """
    notification_manager = get_notification_manager()
    if not notification_manager:
        return
        
    # Get pending notifications for this conversation
    pending_notifications = notification_manager.get_pending_notifications(
        conversation_id=conversation.conversation_id,
        limit=limit
    )
    
    # Add notifications to the conversation
    for notification in pending_notifications:
        # Add as assistant message with is_notification flag
        conversation.add_message(
            "assistant",
            f"{notification.title}\n\n{notification.content}",
            {"is_notification": True, "notification_id": notification.id}
        )
        
        # Mark as displayed
        notification_manager.mark_notification_displayed(notification.id)