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

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Centralized manager for dynamic system prompt content.

    This class provides a standard interface for adding, removing, and retrieving
    content that should be included in the system prompt during conversations.

    Components register their content with the working memory with a specific
    category for organization. Each content item is assigned a unique ID that
    can be used to update or remove it later.

    Managers (like WorkflowManager and ToolRepository) can be registered directly
    and are responsible for managing their own content in working memory.
    """

    def __init__(self):
        """Initialize a new working memory instance."""
        # Main storage for memory items
        self._memory_items: Dict[str, Dict[str, Any]] = {}
        # Store registered managers
        self._managers: List[Any] = []

    def add(self, content: str, category: str) -> str:
        """
        Add content to working memory.

        Args:
            content: The content to add
            category: Category for organization

        Returns:
            item_id: Unique ID for the added item

        Raises:
            ValueError: If content or category is empty
        """
        if not content or not category:
            logger.error("Attempted to add empty content or category")
            raise ValueError("Content and category cannot be empty")

        item_id = str(uuid.uuid4())
        self._memory_items[item_id] = {
            "content": content,
            "category": category,
            "metadata": {}  # Reserved for future use
        }
        logger.debug(f"Added item {item_id} to working memory (category: {category})")
        return item_id

    def remove(self, item_id: str) -> bool:
        """
        Remove content by ID.

        Args:
            item_id: The ID of item to remove

        Returns:
            bool: True if item was removed, False if not found
        """
        if item_id in self._memory_items:
            category = self._memory_items[item_id]["category"]
            del self._memory_items[item_id]
            logger.debug(f"Removed item {item_id} from working memory (category: {category})")
            return True

        logger.warning(f"Attempted to remove non-existent item: {item_id}")
        return False

    def remove_by_category(self, category: str) -> int:
        """
        Remove all items of a specific category.

        Args:
            category: Category to remove

        Returns:
            int: Number of items removed
        """
        ids = [id for id, item in self._memory_items.items()
               if item["category"] == category]

        for id in ids:
            del self._memory_items[id]

        if ids:
            logger.debug(f"Removed {len(ids)} items with category '{category}'")
        return len(ids)

    def get_prompt_content(self) -> str:
        """
        Generate formatted content for the system prompt.

        Returns:
            str: Concatenated content items
        """
        if not self._memory_items:
            logger.warning("Getting prompt content from empty working memory")
            return ""

        content = "\n\n".join(item["content"] for item in self._memory_items.values())
        logger.debug(f"Generated prompt content with {len(self._memory_items)} items")
        return content

    def get_items_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all items of a specific category.

        Args:
            category: Category to retrieve

        Returns:
            List of memory items with specified category
        """
        return [item.copy() for item_id, item in self._memory_items.items()
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
        logger.info(f"Registered manager: {manager.__class__.__name__}")

    def update_all_managers(self) -> None:
        """
        Update content from all registered managers.

        This method calls update_working_memory on all registered managers to ensure
        their content in working memory is up to date. It should be called before
        generating each response.
        """
        for manager in self._managers:
            try:
                if hasattr(manager, 'update_working_memory'):
                    manager.update_working_memory()
            except Exception as e:
                logger.error(f"Error updating working memory from {manager.__class__.__name__}: {e}")


# =====================================================================
# Memory Trinkets - Utility classes for managing specific memory content
# =====================================================================

# Import only what's needed for the trinkets
import os
from typing import List

class ReminderManager:
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
        self.working_memory = working_memory
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
            self.working_memory.remove(self._reminder_id)

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
                self._reminder_id = self.working_memory.add(
                    content=reminder_info,
                    category="reminders"
                )

                logger.debug(f"Added {len(active_reminders)} reminders to working memory (item ID: {self._reminder_id})")
            else:
                logger.debug("No active reminders to add to working memory")

        except Exception as e:
            logger.error(f"Error updating reminder information: {e}")

class TimeManager:
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
        self.working_memory = working_memory
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
            self.working_memory.remove(self._datetime_id)

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
        self._datetime_id = self.working_memory.add(
            content=datetime_info,
            category="datetime"
        )

        logger.debug(f"Updated datetime information in working memory (item ID: {self._datetime_id})")


class SystemStatusManager:
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
        self.working_memory = working_memory
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
            self.working_memory.remove(self._status_id)

        # Format the status information
        status_text = f"# System Status\n"
        status_text += f"System running since: {format_datetime(self._status_info['started_at'], 'date_time')}\n"

        # Add any notices
        if self._status_info["notices"]:
            status_text += "\nNotices:\n"
            for notice in self._status_info["notices"][-3:]:  # Show only the most recent 3
                status_text += f"- {notice}\n"

        # Add to working memory
        self._status_id = self.working_memory.add(
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
        self.update_status()


class UserInfoManager:
    """
    Manager for user information in system prompts.

    This class is responsible for loading and updating user information
    from files into the working memory.
    """

    def __init__(self, working_memory: WorkingMemory, config_manager):
        """
        Initialize a new user information manager.

        Args:
            working_memory: WorkingMemory instance to use for storing user information
            config_manager: Configuration manager instance
        """
        self.working_memory = working_memory
        self.config = config_manager
        self._user_info_id: Optional[str] = None

        # Load user information on initialization
        self.load_user_information()

    def load_user_information(self) -> None:
        """
        Load user information from file into working memory.

        This method should be called during initialization and when
        the user information file is updated.
        """
        # Remove existing user info if present
        if self._user_info_id:
            self.working_memory.remove(self._user_info_id)

        # Get path to user information file
        prompts_dir = self.config.paths.prompts_dir
        user_info_path = os.path.join(prompts_dir, "user_information.txt")

        try:
            # Load user information from file
            with open(user_info_path, "r") as f:
                user_info = f.read().strip()

            # Format the user information
            formatted_info = f"# USER INFORMATION\n{user_info}"

            # Add to working memory
            self._user_info_id = self.working_memory.add(
                content=formatted_info,
                category="user_information"
            )

            logger.info(f"Loaded user information into working memory (item ID: {self._user_info_id})")

        except FileNotFoundError:
            logger.warning(f"User information file not found: {user_info_path}")
        except Exception as e:
            logger.error(f"Error loading user information: {e}")




