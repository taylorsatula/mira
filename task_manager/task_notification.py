"""
Task notification module for managing task execution results.

This module defines a database model for storing and retrieving task execution
results, allowing the conversation system to poll for task notifications.
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union

from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, or_, Enum as SQLAEnum, Index, ForeignKey
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from db import Base, Database
from config.registry import registry
from errors import ToolError, ErrorCode, error_context

# Configure logger
logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Enumeration of notification types."""
    TASK_COMPLETED = "task_completed"   # Task completed successfully
    TASK_FAILED = "task_failed"         # Task execution failed
    SYSTEM = "system"                   # System notification


class NotificationPriority(str, Enum):
    """Enumeration of notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(str, Enum):
    """Enumeration of notification status values."""
    NEW = "new"                  # New notification, not yet displayed
    DISPLAYED = "displayed"      # Notification has been displayed
    ACKNOWLEDGED = "acknowledged" # User has acknowledged the notification
    DISMISSED = "dismissed"      # User has dismissed the notification


class TaskNotification(Base):
    """
    Task notification model for storing task execution results.
    
    Maps to the 'task_notifications' table with columns for notification
    details, task results, and user interaction tracking.
    """
    __tablename__ = 'task_notifications'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Notification metadata
    title = Column(String, nullable=False)
    content = Column(Text)
    notification_type = Column(SQLAEnum(NotificationType), default=NotificationType.TASK_COMPLETED)
    priority = Column(SQLAEnum(NotificationPriority), default=NotificationPriority.NORMAL)
    status = Column(SQLAEnum(NotificationStatus), default=NotificationStatus.NEW)
    
    # Task information
    task_id = Column(String, ForeignKey('scheduled_tasks.id', ondelete='CASCADE'))
    task_name = Column(String)
    
    # Result data
    result = Column(MutableDict.as_mutable(JSON))
    error = Column(Text)
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_task_notifications_status', 'status'),
        Index('ix_task_notifications_priority', 'priority'),
        Index('ix_task_notifications_created_at', 'created_at'),
        Index('ix_task_notifications_task_id', 'task_id'),
        Index('ix_task_notifications_priority_created_at', 'priority', 'created_at'),
    )
    
    # Relationships with tasks
    task = relationship("ScheduledTask", foreign_keys=[task_id], back_populates="notifications")
    related_task = relationship("ScheduledTask", back_populates="notification",
                               primaryjoin="TaskNotification.id == ScheduledTask.notification_id")
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    displayed_at = Column(DateTime)
    acknowledged_at = Column(DateTime)
    
    # Display settings
    display_in_next_conversation = Column(String)  # ID of conversation to display in, if any
    auto_dismiss_after_display = Column(Boolean, default=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the notification
        """
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "notification_type": self.notification_type.value if self.notification_type else None,
            "priority": self.priority.value if self.priority else None,
            "status": self.status.value if self.status else None,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "displayed_at": self.displayed_at.isoformat() if self.displayed_at else None, 
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "display_in_next_conversation": self.display_in_next_conversation,
            "auto_dismiss_after_display": self.auto_dismiss_after_display
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskNotification':
        """
        Create a TaskNotification instance from a dictionary.
        
        Args:
            data: Dictionary representation of a notification
            
        Returns:
            TaskNotification instance
        """
        notification = cls(
            id=data.get("id") or str(uuid.uuid4()),
            title=data.get("title"),
            content=data.get("content"),
            task_id=data.get("task_id"),
            task_name=data.get("task_name"),
            result=data.get("result"),
            error=data.get("error"),
            display_in_next_conversation=data.get("display_in_next_conversation"),
            auto_dismiss_after_display=data.get("auto_dismiss_after_display", False)
        )
        
        # Set enum values if provided
        if "notification_type" in data and data["notification_type"]:
            notification.notification_type = NotificationType(data["notification_type"])
            
        if "priority" in data and data["priority"]:
            notification.priority = NotificationPriority(data["priority"])
            
        if "status" in data and data["status"]:
            notification.status = NotificationStatus(data["status"])
        else:
            notification.status = NotificationStatus.NEW
            
        # Set timestamps if provided
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                notification.created_at = datetime.fromisoformat(data["created_at"])
            else:
                notification.created_at = data["created_at"]
                
        if "displayed_at" in data and data["displayed_at"]:
            if isinstance(data["displayed_at"], str):
                notification.displayed_at = datetime.fromisoformat(data["displayed_at"])
            else:
                notification.displayed_at = data["displayed_at"]
                
        if "acknowledged_at" in data and data["acknowledged_at"]:
            if isinstance(data["acknowledged_at"], str):
                notification.acknowledged_at = datetime.fromisoformat(data["acknowledged_at"])
            else:
                notification.acknowledged_at = data["acknowledged_at"]
                
        return notification


class NotificationManager:
    """
    Manager class for working with task notifications.
    
    Provides methods for creating, retrieving, and updating notifications.
    """
    def __init__(self):
        """
        Initialize the notification manager.
        """
        self.db = Database()
        self.logger = logging.getLogger(__name__)
    
    def create_notification(self, notification_data: Dict[str, Any]) -> TaskNotification:
        """
        Create a new task notification.
        
        Args:
            notification_data: Notification definition data
            
        Returns:
            The created notification
            
        Raises:
            ToolError: If the notification is invalid
        """
        with error_context(
            component_name="NotificationManager",
            operation="create_notification",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Create the notification object
            notification = TaskNotification.from_dict(notification_data)
            
            # Ensure required fields
            if not notification.title:
                raise ToolError(
                    "Notification title is required",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Save to database
            self.db.add(notification)
            self.logger.info(f"Created new notification: {notification.id}")
            
            return notification
    
    def get_pending_notifications(
        self, 
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> List[TaskNotification]:
        """
        Get pending (new) notifications for display.
        
        Args:
            conversation_id: ID of the conversation to get notifications for
            limit: Maximum number of notifications to return
            
        Returns:
            List of pending notifications
        """
        with self.db.get_session() as session:
            query = session.query(TaskNotification).filter(
                TaskNotification.status == NotificationStatus.NEW
            )
            
            # Filter by conversation ID if provided
            if conversation_id:
                query = query.filter(
                    or_(
                        TaskNotification.display_in_next_conversation.is_(None),
                        TaskNotification.display_in_next_conversation == conversation_id
                    )
                )
                
            # Order by priority and creation time
            query = query.order_by(
                TaskNotification.priority.desc(),
                TaskNotification.created_at.asc()
            )
            
            # Apply limit
            query = query.limit(limit)
            
            return query.all()
    
    def mark_notification_displayed(self, notification_id: str) -> Optional[TaskNotification]:
        """
        Mark a notification as displayed.
        
        Args:
            notification_id: ID of the notification to update
            
        Returns:
            The updated notification or None if not found
        """
        notification = self.db.get(TaskNotification, notification_id)
        if not notification:
            return None
            
        notification.status = NotificationStatus.DISPLAYED
        notification.displayed_at = datetime.now(timezone.utc)
        
        # Auto-dismiss if configured
        if notification.auto_dismiss_after_display:
            notification.status = NotificationStatus.DISMISSED
            
        self.db.update(notification)
        return notification
    
    def mark_notification_acknowledged(self, notification_id: str) -> Optional[TaskNotification]:
        """
        Mark a notification as acknowledged by the user.
        
        Args:
            notification_id: ID of the notification to update
            
        Returns:
            The updated notification or None if not found
        """
        notification = self.db.get(TaskNotification, notification_id)
        if not notification:
            return None
            
        notification.status = NotificationStatus.ACKNOWLEDGED
        notification.acknowledged_at = datetime.now(timezone.utc)
        self.db.update(notification)
        return notification
    
    def dismiss_notification(self, notification_id: str) -> Optional[TaskNotification]:
        """
        Dismiss a notification.
        
        Args:
            notification_id: ID of the notification to dismiss
            
        Returns:
            The updated notification or None if not found
        """
        notification = self.db.get(TaskNotification, notification_id)
        if not notification:
            return None
            
        notification.status = NotificationStatus.DISMISSED
        self.db.update(notification)
        return notification
    
    def get_notifications_for_task(self, task_id: str) -> List[TaskNotification]:
        """
        Get all notifications for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of notifications for the task
        """
        # Using the task relationship for improved query performance
        with self.db.get_session() as session:
            task = session.query(TaskNotification).filter(
                TaskNotification.task_id == task_id
            ).order_by(TaskNotification.created_at.desc()).all()
            return task
    
    def create_task_result_notification(
        self,
        task_id: str,
        task_name: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> TaskNotification:
        """
        Create a notification for a task execution result.
        
        Args:
            task_id: ID of the task
            task_name: Name of the task
            success: Whether the task succeeded
            result: Task execution result (if successful)
            error: Error message (if failed)
            conversation_id: ID of the conversation to display the notification in
            
        Returns:
            The created notification
        """
        notification_type = NotificationType.TASK_COMPLETED if success else NotificationType.TASK_FAILED
        priority = NotificationPriority.NORMAL if success else NotificationPriority.HIGH
        
        # Create title and content
        if success:
            title = f"✅ Task '{task_name}' completed successfully"
            content = f"Scheduled task '{task_name}' completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            title = f"❌ Task '{task_name}' failed"
            content = f"Scheduled task '{task_name}' failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if error:
                content += f"\n\nError: {error}"
        
        # Create notification data
        notification_data = {
            "title": title,
            "content": content,
            "notification_type": notification_type.value,
            "priority": priority.value,
            "task_id": task_id,
            "task_name": task_name,
            "result": result,
            "error": error,
            "display_in_next_conversation": conversation_id
        }
        
        return self.create_notification(notification_data)