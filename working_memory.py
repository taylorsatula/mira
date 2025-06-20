"""
Working memory module for centralized system prompt content management.

This module handles the dynamic addition and removal of content to be included
in the system prompt. It provides a standardized API for various components
to manage their own prompt content.

It also contains 'trinkets' - smaller utility classes that manage specific
types of content in the working memory, such as time information and system status.
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import utilities needed by trinkets
from utils.timezone_utils import (
    utc_now, ensure_utc, convert_from_utc, format_datetime,
    get_default_timezone
)
from auth import get_current_user_id

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Working memory with automatic user partitioning"""
    
    def __init__(self):
        # User memories stored by user_id
        self._user_memories: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._managers: List[Any] = []
    
    def _get_user_memory(self) -> Dict[str, Dict[str, Any]]:
        """Get memory for current user"""
        user_id = get_current_user_id()
        if user_id not in self._user_memories:
            self._user_memories[user_id] = {}
        return self._user_memories[user_id]

    def add(self, content: str, category: str) -> str:
        """Add content to current user's memory"""
        if not content or not category:
            raise ValueError("Content and category cannot be empty")
        
        item_id = str(uuid.uuid4())
        user_memory = self._get_user_memory()
        user_memory[item_id] = {
            "content": content,
            "category": category,
            "metadata": {}
        }
        return item_id

    def remove(self, item_id: str) -> bool:
        """Remove item from current user's memory"""
        user_memory = self._get_user_memory()
        if item_id in user_memory:
            del user_memory[item_id]
            return True
        return False

    def remove_by_category(self, category: str) -> int:
        """Remove all items of a specific category for current user"""
        user_memory = self._get_user_memory()
        ids = [id for id, item in user_memory.items()
               if item["category"] == category]

        for id in ids:
            del user_memory[id]

        return len(ids)

    def get_prompt_content(self) -> str:
        """Get prompt content for current user"""
        user_memory = self._get_user_memory()
        if not user_memory:
            return ""
        return "\n\n".join(item["content"] for item in user_memory.values())

    def get_items_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all items of a specific category for current user"""
        user_memory = self._get_user_memory()
        return [item.copy() for item_id, item in user_memory.items()
                if item["category"] == category]

    def register_manager(self, manager: Any) -> None:
        """
        Register a manager that will update its own content in working memory.

        Managers should implement an update_working_memory method that will be called
        to refresh content in working memory when needed.

        Args:
            manager: Manager instance to register

        Raises:
            ValueError: If the manager doesn't have an update_working_memory method
        """
        if not hasattr(manager, 'update_working_memory'):
            logger.warning(f"Manager {manager.__class__.__name__} has no update_working_memory method")

        # Store the manager
        self._managers.append(manager)
        logger.info(f"Registered manager: {manager.__class__.__name__} (total managers: {len(self._managers)})")

    def update_all_managers(self) -> None:
        """
        Update content from all registered managers.

        This method calls update_working_memory on all registered managers to ensure
        their content in working memory is up to date. It should be called before
        generating each response.
        """
        logger.info(f"Updating {len(self._managers)} registered managers")
        for i, manager in enumerate(self._managers):
            try:
                if hasattr(manager, 'update_working_memory'):
                    logger.info(f"Updating manager {i+1}/{len(self._managers)}: {manager.__class__.__name__}")
                    manager.update_working_memory()
            except Exception as e:
                logger.error(f"Error updating working memory from {manager.__class__.__name__}: {e}")
    
    def cleanup_all_managers(self) -> None:
        """
        Clean up all registered managers and clear working memory.
        
        This method calls cleanup on all registered managers that support it
        and clears all memory items. Should be called on system shutdown.
        """
        manager_count = len(self._managers)
        
        # Clean up registered managers
        for manager in self._managers:
            try:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {manager.__class__.__name__}: {e}")
        
        # Clear all user memory items
        total_items = sum(len(user_mem) for user_mem in self._user_memories.values())
        self._user_memories.clear()
        
        # Clear managers list
        self._managers.clear()
        
        logger.info(f"Working memory cleanup complete: cleared {total_items} items and {manager_count} managers")


# =====================================================================
# Memory Trinkets - Utility classes for managing specific memory content
# =====================================================================

# Import only what's needed for the trinkets
import os
from typing import List


class BaseTrinket:
    """
    Base class for working memory trinkets with common cleanup patterns.
    
    Provides standardized content management and cleanup functionality
    to prevent memory leaks in long-running processes.
    """
    
    def __init__(self, working_memory: WorkingMemory):
        """
        Initialize base trinket.
        
        Args:
            working_memory: WorkingMemory instance
        """
        self.working_memory = working_memory
        self._content_ids: List[str] = []  # Track all content IDs for cleanup
    
    def _add_content(self, content: str, category: str) -> str:
        """
        Add content to working memory with automatic tracking.
        
        Args:
            content: Content to add
            category: Category for the content
            
        Returns:
            Content ID for manual removal if needed
        """
        content_id = self.working_memory.add(content, category)
        self._content_ids.append(content_id)
        return content_id
    
    def _remove_content(self, content_id: str) -> bool:
        """
        Remove specific content and stop tracking it.
        
        Args:
            content_id: ID of content to remove
            
        Returns:
            True if content was found and removed
        """
        success = self.working_memory.remove(content_id)
        if success and content_id in self._content_ids:
            self._content_ids.remove(content_id)
        return success
    
    def cleanup(self) -> None:
        """
        Clean up all tracked content from working memory.
        
        This method should be called on system shutdown or trinket disposal
        to prevent memory leaks.
        """
        removed_count = 0
        for content_id in self._content_ids:
            if self.working_memory.remove(content_id):
                removed_count += 1
        
        self._content_ids.clear()
        logger.debug(f"{self.__class__.__name__} cleaned up {removed_count} content items")

class ReminderManager(BaseTrinket):
    """
    Manager for reminder information in system prompts.

    This class is responsible for adding overdue and upcoming reminders
    to the working memory so they appear in the system prompt throughout
    the conversation, allowing the assistant to remind the user about them.
    """

    def __init__(self, working_memory: WorkingMemory):
        """
        Initialize a new reminder manager.

        Args:
            working_memory: WorkingMemory instance to use for storing reminder information
        """
        super().__init__(working_memory)
        self._reminder_id: Optional[str] = None

        # Automatically check for reminders on initialization
        self.update_reminder_info()

    def update_reminder_info(self) -> None:
        """
        Update reminder information in working memory.

        This method should be called before each response generation to
        ensure the reminder information is current.
        """
        # Remove existing reminder info if present
        if self._reminder_id:
            self._remove_content(self._reminder_id)
            self._reminder_id = None

        try:
            # Create ReminderTool instance for accessing reminders
            from tools.reminder_tool import ReminderTool
            reminder_tool = ReminderTool()

            # Get overdue reminders (due date has passed but not completed)
            overdue_result = reminder_tool.run(
                operation="get_reminders",
                date_type="overdue"
            )

            # Get today's reminders
            today_result = reminder_tool.run(
                operation="get_reminders",
                date_type="today"
            )

            # Format reminders for system prompt if any exist
            active_reminders = []

            # Process overdue reminders
            if overdue_result.get("count", 0) > 0:
                reminders = overdue_result.get("reminders", [])
                # Filter out completed reminders
                overdue_reminders = [
                    r for r in reminders if not r.get('completed', False)
                ]
                active_reminders.extend([
                    {
                        "title": r["title"],
                        "date": r["reminder_date"],
                        "status": "OVERDUE"
                    } for r in overdue_reminders
                ])

            # Process today's reminders
            if today_result.get("count", 0) > 0:
                reminders = today_result.get("reminders", [])
                # Filter out completed reminders
                today_reminders = [
                    r for r in reminders if not r.get('completed', False)
                ]
                active_reminders.extend([
                    {
                        "title": r["title"],
                        "date": r["reminder_date"],
                        "status": "TODAY"
                    } for r in today_reminders
                ])

            # Only add reminders to working memory if there are active ones
            if active_reminders:
                import datetime
                # Create the reminder information text
                reminder_info = "# Active Reminders\n"
                reminder_info += "The user has the following reminders:\n\n"

                # Format each reminder
                for reminder in active_reminders:
                    date_obj = datetime.datetime.fromisoformat(reminder["date"])
                    formatted_time = date_obj.strftime("%I:%M %p")
                    status_tag = f"[{reminder['status']}]"
                    reminder_info += f"* {status_tag} {reminder['title']} at {formatted_time}\n"

                reminder_info += "\nPlease remind the user about these during the conversation if relevant."

                # Add to working memory
                self._reminder_id = self._add_content(
                    content=reminder_info,
                    category="reminders"
                )

                logger.debug(f"Added {len(active_reminders)} reminders to working memory (item ID: {self._reminder_id})")
            else:
                logger.debug("No active reminders to add to working memory")

        except Exception as e:
            logger.error(f"Error updating reminder information: {e}")

class TimeManager(BaseTrinket):
    """
    Manager for date and time information in system prompts.

    This class is responsible for updating the current date and time
    information in the working memory before each response generation.
    """

    def __init__(self, working_memory: WorkingMemory):
        """
        Initialize a new time manager.

        Args:
            working_memory: WorkingMemory instance to use for storing time information
        """
        super().__init__(working_memory)
        self._datetime_id: Optional[str] = None

        # Automatically update time on initialization
        self.update_datetime_info()

    def update_datetime_info(self) -> None:
        """
        Update the current date and time information in working memory.

        This method should be called before each response generation to
        ensure the time information is current.
        """
        # Remove existing datetime info if present
        if self._datetime_id:
            self._remove_content(self._datetime_id)
            self._datetime_id = None

        # Get current time in UTC
        current_time = utc_now()

        # Convert to user's timezone
        user_tz = get_default_timezone()
        local_time = convert_from_utc(current_time, user_tz)

        # Format both UTC and local time
        formatted_local = format_datetime(local_time, 'date_time', include_timezone=True)
        formatted_utc = format_datetime(current_time, 'date_time')

        # Create the datetime information text
        datetime_info = f"# Current Date and Time\n"
        datetime_info += f"The current date and time is {formatted_local} (UTC: {formatted_utc})."

        # Add to working memory
        self._datetime_id = self._add_content(
            content=datetime_info,
            category="datetime"
        )

        logger.debug(f"Updated datetime information in working memory (item ID: {self._datetime_id})")


class SystemStatusManager(BaseTrinket):
    """
    Manager for system status information in system prompts.

    This class is responsible for tracking and updating system status
    information in the working memory.
    """

    def __init__(self, working_memory: WorkingMemory):
        """
        Initialize a new system status manager.

        Args:
            working_memory: WorkingMemory instance to use for storing status information
        """
        super().__init__(working_memory)
        self._status_id: Optional[str] = None
        self._status_info: Dict[str, Any] = {
            "started_at": utc_now(),
            "warnings": 0,
            "errors": 0,
            "notices": []
        }

        # Add initial status
        self.update_status()

    def update_status(self) -> None:
        """
        Update the system status information in working memory.
        """
        # Remove existing status if present
        if self._status_id:
            self._remove_content(self._status_id)
            self._status_id = None

        # Format the status information
        status_text = f"# System Status\n"
        status_text += f"System running since: {format_datetime(self._status_info['started_at'], 'date_time')}\n"

        # Add any notices
        if self._status_info["notices"]:
            status_text += "\nNotices:\n"
            for notice in self._status_info["notices"][-3:]:  # Show only the most recent 3
                status_text += f"- {notice}\n"

        # Add to working memory
        self._status_id = self._add_content(
            content=status_text,
            category="system_status"
        )

        logger.debug(f"Updated system status in working memory (item ID: {self._status_id})")

    def add_notice(self, notice: str) -> None:
        """
        Add a notice to the system status.

        Args:
            notice: The notice text to add
        """
        self._status_info["notices"].append(notice)
        # Limit notices to prevent memory growth
        if len(self._status_info["notices"]) > 50:
            self._status_info["notices"] = self._status_info["notices"][-30:]
        self.update_status()
    
    def cleanup(self) -> None:
        """
        Clean up system status manager resources.
        
        Extends base cleanup to also clear notices list.
        """
        # Call base cleanup
        super().cleanup()
        
        # Clear notices list to free memory
        self._status_info["notices"].clear()
        self._status_id = None




class ProactiveMemoryTrinket(BaseTrinket):
    """
    Trinket for proactive memory surfacing in working memory.
    
    Uses stateless functions to find and surface relevant memories
    based on recent conversation context.
    """
    
    def __init__(self, working_memory: WorkingMemory, memory_manager, conversation):
        """
        Initialize proactive memory trinket.
        
        Args:
            working_memory: WorkingMemory instance
            memory_manager: LT_Memory MemoryManager instance
            conversation: Conversation instance for context
        """
        super().__init__(working_memory)
        self.memory_manager = memory_manager
        self.conversation = conversation
        self._proactive_id: Optional[str] = None
        
        # Register as a manager to get update calls
        working_memory.register_manager(self)
        
    def update_working_memory(self) -> None:
        """
        Called by working memory before each response.
        Delegates to update_proactive_memories.
        """
        self.update_proactive_memories()
        
    def update_proactive_memories(self) -> None:
        """
        Update working memory with proactively surfaced memories.
        """
        logger.info("ProactiveMemoryTrinket: Starting memory update cycle")
        
        # Remove existing proactive memory content
        if self._proactive_id:
            self._remove_content(self._proactive_id)
            self._proactive_id = None
        
        try:
            # Import here to avoid circular dependencies
            from lt_memory.proactive_memory import get_relevant_memories, format_relevant_memories
            
            # Find relevant memories using weighted context
            relevant_memories = get_relevant_memories(self.conversation.messages, self.memory_manager)
            
            if relevant_memories:
                # Format and add to working memory
                memory_content = format_relevant_memories(relevant_memories)
                self._proactive_id = self._add_content(
                    content=memory_content,
                    category="proactive_memories"
                )
                
                logger.debug(f"Surfaced {len(relevant_memories)} proactive memories")
                    
        except Exception as e:
            logger.warning(f"Failed to update proactive memories: {e}")


class ConversationArchiveManager(BaseTrinket):
    """
    Manager for conversation archive access in working memory.
    
    This class provides a lightweight interface for injecting archived
    conversations into the current context through working memory.
    """
    
    def __init__(self, working_memory: WorkingMemory, conversation_timeline_manager):
        """
        Initialize conversation archive manager.
        
        Args:
            working_memory: WorkingMemory instance
            conversation_timeline_manager: ConversationTimelineManager instance from lt_memory
        """
        super().__init__(working_memory)
        self.archive_bridge = conversation_timeline_manager
        self._archive_content_id: Optional[str] = None
        
        # Initialize with archive status
        self.update_archive_content()
    
    def update_archive_content(self) -> None:
        """
        Update working memory with current archive content.
        
        This method is called automatically to refresh injected
        conversation content in working memory. Includes automatic
        recent context (previous day + past week).
        """
        # Remove existing archive content
        if self._archive_content_id:
            self._remove_content(self._archive_content_id)
            self._archive_content_id = None
        
        try:
            content_parts = []
            
            # 1. Get automatic recent context
            recent_context = self._get_automatic_recent_context()
            if recent_context:
                content_parts.append(recent_context)
            
            # 2. Get manually injected conversations
            injected_content = self.archive_bridge.get_current_injections_content()
            if injected_content:
                content_parts.append("\n" + injected_content)
            
            # Combine all content
            if content_parts:
                full_content = "\n".join(content_parts)
                
                # Add to working memory
                self._archive_content_id = self._add_content(
                    content=full_content,
                    category="archived_conversations"
                )
                
                logger.debug(f"Updated archive content in working memory (item ID: {self._archive_content_id})")
            else:
                logger.debug("No archive content to add to working memory")
                
        except Exception as e:
            logger.error(f"Error updating archive content in working memory: {e}")
    
    def inject_conversation(self, target_date, include_full_messages: bool = False) -> Dict[str, Any]:
        """
        Inject an archived conversation into working memory.
        
        Args:
            target_date: Date of conversation to inject (date object or ISO string)
            include_full_messages: Whether to include full message content
            
        Returns:
            Operation results
        """
        try:
            # Parse date if string
            if isinstance(target_date, str):
                from datetime import date
                target_date = date.fromisoformat(target_date)
            
            # Use bridge to inject conversation
            result = self.archive_bridge.inject_conversation(
                target_date, include_full_messages
            )
            
            if result["success"]:
                # Update working memory content
                self.update_archive_content()
            
            return result
            
        except Exception as e:
            logger.error(f"Error injecting conversation: {e}")
            return {
                "success": False,
                "message": f"Error injecting conversation: {str(e)}"
            }
    
    def remove_conversation(self, target_date) -> Dict[str, Any]:
        """
        Remove an injected conversation from working memory.
        
        Args:
            target_date: Date of conversation to remove
            
        Returns:
            Operation results
        """
        try:
            # Parse date if string
            if isinstance(target_date, str):
                from datetime import date
                target_date = date.fromisoformat(target_date)
            
            # Use bridge to remove conversation
            result = self.archive_bridge.remove_conversation(target_date)
            
            if result["success"]:
                # Update working memory content
                self.update_archive_content()
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing conversation: {e}")
            return {
                "success": False,
                "message": f"Error removing conversation: {str(e)}"
            }
    
    def clear_all_injections(self) -> Dict[str, Any]:
        """
        Clear all injected conversations from working memory.
        
        Returns:
            Operation results
        """
        try:
            # Use bridge to clear injections
            result = self.archive_bridge.clear_all_injections()
            
            if result["success"]:
                # Update working memory content
                self.update_archive_content()
            
            return result
            
        except Exception as e:
            logger.error(f"Error clearing injections: {e}")
            return {
                "success": False,
                "message": f"Error clearing injections: {str(e)}"
            }
    
    def inject_date_range(self, start_date, end_date, summary_only: bool = True) -> Dict[str, Any]:
        """
        Inject multiple conversations from a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            summary_only: Whether to include only summaries
            
        Returns:
            Operation results
        """
        try:
            # Parse dates if strings
            if isinstance(start_date, str):
                from datetime import date
                start_date = date.fromisoformat(start_date)
            if isinstance(end_date, str):
                from datetime import date
                end_date = date.fromisoformat(end_date)
            
            # Use bridge to inject date range
            result = self.archive_bridge.inject_date_range(
                start_date, end_date, summary_only
            )
            
            if result["success"]:
                # Update working memory content
                self.update_archive_content()
            
            return result
            
        except Exception as e:
            logger.error(f"Error injecting date range: {e}")
            return {
                "success": False,
                "message": f"Error injecting date range: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current archive status.
        
        Returns:
            Archive status information
        """
        try:
            return self.archive_bridge.get_archive_status()
        except Exception as e:
            logger.error(f"Error getting archive status: {e}")
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "injected_dates": [],
                "injection_count": 0
            }
    
    def _get_automatic_recent_context(self) -> Optional[str]:
        """
        Get automatic recent context (previous day + weekly summary).
        
        Returns:
            Formatted context string or None if no recent context
        """
        try:
            # Get recent context summaries from conversation archive
            context_data = self.archive_bridge.memory_manager.conversation_archive.get_recent_context_summaries()
            
            content_parts = []
            
            # Add yesterday's summary
            if context_data.get("yesterday"):
                yesterday_info = context_data["yesterday"]
                content_parts.append(
                    f"# Recent Context\n\n"
                    f"## Yesterday ({yesterday_info['date']})\n"
                    f"**Summary:** {yesterday_info['summary']}\n"
                )
            
            # Add weekly summary
            if context_data.get("weekly"):
                weekly_info = context_data["weekly"]
                content_parts.append(
                    f"\n## Past Week ({weekly_info['date_range']})\n"
                    f"**Summary:** {weekly_info['summary']}\n"
                )
            
            if content_parts:
                return "".join(content_parts) + "\n*This recent context is automatically included to maintain conversational continuity.*"
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting automatic recent context: {e}")
            return None



