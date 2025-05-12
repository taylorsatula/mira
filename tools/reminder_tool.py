"""
Reminder tool for managing scheduled reminders.

This tool allows users to create reminders with specific dates, details,
and contact information. It stores reminders in a SQLite database and provides
functions to add, retrieve, and manage reminders.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dateutil import parser as date_parser
from sqlalchemy import Column, String, DateTime, Text, Boolean
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from db import Database, Base
from config.registry import registry
from utils.timezone_utils import (
    validate_timezone, get_default_timezone, convert_to_timezone,
    format_datetime, parse_time_string, utc_now, ensure_utc
)

# Define configuration class for ReminderTool
class ReminderToolConfig(BaseModel):
    """Configuration for the reminder_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    # Add any other configuration fields specific to this tool

# Register with registry
registry.register("reminder_tool", ReminderToolConfig)


class Reminder(Base):
    """
    Reminder model for storing reminder data.
    
    Maps to the 'reminders' table with columns for reminder details
    including contact information and scheduling.
    """
    __tablename__ = 'reminders'

    # Primary key
    id = Column(String, primary_key=True)
    
    # Reminder details
    title = Column(String, nullable=False)
    description = Column(Text)
    
    # Scheduling information
    reminder_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: utc_now())
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime)
    
    # Contact information
    contact_name = Column(String)
    contact_email = Column(String)
    contact_phone = Column(String)
    customer_id = Column(String)  # Reference to customer in customer database if available
    
    # Additional data (optional)
    additional_notes = Column(Text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the reminder with timestamps in user timezone
        """
        # Get user's timezone
        user_tz = get_default_timezone()
        
        # Function to format datetime with proper timezone
        def format_dt(dt: Optional[datetime]) -> Optional[str]:
            if not dt:
                return None
            # Convert to user timezone before serializing
            dt_aware = convert_to_timezone(dt, user_tz)
            return dt_aware.isoformat()
        
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "reminder_date": format_dt(self.reminder_date),
            "created_at": format_dt(self.created_at),
            "completed": self.completed,
            "completed_at": format_dt(self.completed_at),
            "contact_name": self.contact_name,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "customer_id": self.customer_id,
            "additional_notes": self.additional_notes,
            "timezone": user_tz  # Include timezone information
        }


class ReminderTool(Tool):
    """
    Tool for managing reminders with customer contact integration.
    
    This tool allows setting and retrieving reminders for specific dates with
    detailed contact information. It integrates with the customer database
    to fetch complete contact details when available.
    """

    name = "reminder_tool"
    simple_description = """
    Manages scheduled reminders with contact information integration. Use this tool when the user
    wants to create, view, or manage reminders about tasks, follow-ups, or appointments."""
    
    implementation_details = """
    
    IMPORTANT: This tool requires parameters to be passed as a JSON string in the "kwargs" field.
    The tool supports these operations:
    
    1. add_reminder: Create a new reminder with a date, description, and optional contact info.
       - Required: title (brief description), date (when to be reminded)
       - Optional: description (details), contact_name, contact_email, contact_phone
       - If a contact name is provided, attempts to find matching customer information
       - Returns the created reminder with a unique identifier
    
    2. get_reminders: Retrieve reminders for a specific date range.
       - Required: date_type ("today", "tomorrow", "upcoming", "past", "all", "date" or "range")
       - If date_type is "date", requires specific_date parameter
       - If date_type is "range", requires start_date and end_date parameters
       - Returns list of reminders matching the criteria
    
    3. mark_completed: Mark a reminder as completed.
       - Required: reminder_id (the ID of the reminder to mark as completed)
       - Returns the updated reminder
       
    4. update_reminder: Update an existing reminder's details.
       - Required: reminder_id (the ID of the reminder to update)
       - Optional: Any fields to update (title, description, date, contact information)
       - Returns the updated reminder
       
    5. delete_reminder: Remove a reminder.
       - Required: reminder_id (the ID of the reminder to delete)
       - Returns confirmation of deletion
       
    This tool automatically integrates with the customer database when contact names are provided,
    fetching complete contact information for known customers. When creating reminders with a name
    that doesn't match an existing customer, the tool will prompt for additional contact details.
    """
    
    description = simple_description + implementation_details
    
    usage_examples = [
        {
            "input": {
                "operation": "add_reminder",
                "kwargs": "{\"title\": \"Window cleaning follow-up\", \"date\": \"in 3 weeks\", \"description\": \"Call to schedule window cleaning service\", \"contact_name\": \"John Smith\"}"
            },
            "output": {
                "reminder": {
                    "id": "rem_123456",
                    "title": "Window cleaning follow-up",
                    "description": "Call to schedule window cleaning service",
                    "reminder_date": "2025-05-13T12:00:00",
                    "contact_name": "John Smith",
                    "contact_email": "john.smith@example.com",
                    "contact_phone": "256-555-1234"
                }
            }
        },
        {
            "input": {
                "operation": "get_reminders",
                "kwargs": "{\"date_type\": \"upcoming\"}"
            },
            "output": {
                "reminders": [
                    {
                        "id": "rem_123456",
                        "title": "Window cleaning follow-up",
                        "description": "Call to schedule window cleaning service",
                        "reminder_date": "2025-05-13T12:00:00",
                        "contact_name": "John Smith"
                    }
                ]
            }
        }
    ]

    def __init__(self):
        """Initialize the reminder tool with database access."""
        super().__init__()
        self.db = Database()
        
        # Ensure data directory exists
        self.data_dir = os.path.join("data", "tools", "reminder_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("ReminderTool initialized")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a reminder operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        1. add_reminder: Create a new reminder
           - Required: title, date
           - Optional: description, contact_name, contact_email, contact_phone, additional_notes
           - Returns: Dict with created reminder

        2. get_reminders: Retrieve reminders
           - Required: date_type ("today", "tomorrow", "upcoming", "past", "all", "date" or
            "range")
           - If date_type is "date", requires specific_date parameter
           - If date_type is "range", requires start_date and end_date parameters
           - Returns: Dict with list of reminders

        3. mark_completed: Mark a reminder as completed
           - Required: reminder_id
           - Returns: Dict with updated reminder

        4. update_reminder: Update an existing reminder
           - Required: reminder_id
           - Optional: Any fields to update (title, description, date, contact information)
           - Returns: Dict with updated reminder

        5. delete_reminder: Delete a reminder
           - Required: reminder_id
           - Returns: Dict with deletion confirmation
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Route to the appropriate operation
            if operation == "add_reminder":
                return self._add_reminder(**kwargs)
            elif operation == "get_reminders":
                return self._get_reminders(**kwargs)
            elif operation == "mark_completed":
                return self._mark_completed(**kwargs)
            elif operation == "update_reminder":
                return self._update_reminder(**kwargs)
            elif operation == "delete_reminder":
                return self._delete_reminder(**kwargs)
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "add_reminder, get_reminders, mark_completed, "
                    "update_reminder, delete_reminder",
                    ErrorCode.TOOL_INVALID_INPUT,
                )

    def _add_reminder(
        self,
        title: str,
        date: str,
        description: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_email: Optional[str] = None,
        contact_phone: Optional[str] = None,
        additional_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a new reminder with optional contact information.
        
        Args:
            title: Brief title or subject of the reminder
            date: When the reminder should occur (can be natural language like
                "tomorrow" or "in 3 weeks")
            description: Detailed description of the reminder
            contact_name: Name of the contact associated with this reminder
            contact_email: Email of the contact
            contact_phone: Phone number of the contact
            additional_notes: Any additional information to store with the reminder
            
        Returns:
            Dict containing the created reminder
            
        Raises:
            ToolError: If required fields are missing or date parsing fails
        """
        self.logger.info(f"Adding reminder: {title} for {date}")
        
        # Validate required parameters
        if not title:
            raise ToolError(
                "Title is required for adding a reminder",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not date:
            raise ToolError(
                "Date is required for adding a reminder",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Parse the date from natural language
        try:
            reminder_date = self._parse_date(date)
        except Exception as e:
            raise ToolError(
                f"Failed to parse date '{date}': {str(e)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Generate a unique ID for the reminder
        import uuid
        reminder_id = f"rem_{uuid.uuid4().hex[:8]}"
        
        # Check if contact name exists in customer database
        customer_info = None
        if contact_name:
            customer_info = self._lookup_customer(contact_name)
            
        # Create the reminder object
        reminder = Reminder(
            id=reminder_id,
            title=title,
            description=description,
            reminder_date=ensure_utc(reminder_date),  # Ensure it's UTC-aware
            created_at=utc_now(),  # Use UTC-aware time
            completed=False,
            contact_name=contact_name,
            contact_email=contact_email,
            contact_phone=contact_phone,
            additional_notes=additional_notes
        )
        
        # Update with customer info if available
        if customer_info:
            customer = customer_info.get("customer", {})
            reminder.customer_id = customer.get("id")
            
            # Only update contact details if they weren't provided
            if not contact_email and "email_address" in customer:
                reminder.contact_email = customer.get("email_address")
                
            if not contact_phone and "phone_number" in customer:
                reminder.contact_phone = customer.get("phone_number")
        
        # Save reminder to database
        try:
            self.db.add(reminder)
            self.logger.info(f"Added reminder with ID: {reminder_id}")
        except Exception as e:
            self.logger.error(f"Error saving reminder: {e}")
            raise ToolError(
                f"Failed to save reminder: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Prepare response
        result = {
            "reminder": reminder.to_dict(),
            "message": f"Reminder added for {reminder_date.strftime('%Y-%m-%d')}"
        }
        
        # Add customer details to response if found
        if customer_info:
            result["customer_found"] = True
            result["customer_info"] = customer_info.get("customer", {})
            result["message"] += f" with contact information for {contact_name}"
        elif contact_name:
            result["customer_found"] = False
            result["message"] += f". No customer record found for {contact_name}."
            
        return result

    def _get_reminders(
        self,
        date_type: str,
        specific_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get reminders based on date criteria.
        
        Args:
            date_type: Type of date query ("today", "tomorrow", "upcoming", "past",
                "all", "date", "range")
            specific_date: Specific date string (required if date_type is "date")
            start_date: Start date string (required if date_type is "range")
            end_date: End date string (required if date_type is "range")
            
        Returns:
            Dict containing list of reminders matching the criteria
            
        Raises:
            ToolError: If parameters are invalid or missing required fields
        """
        self.logger.info(f"Getting reminders with date_type: {date_type}")
        
        # Validate date_type
        valid_date_types = ["today", "tomorrow", "upcoming", "past", "all", "date", "range", "overdue"]
        if date_type not in valid_date_types:
            raise ToolError(
                f"Invalid date_type: {date_type}. Must be one of {valid_date_types}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Use UTC for all internal datetime operations
        today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        # No timezone conversion needed here - we'll work in UTC and only convert at display time
        
        with self.db.get_session() as session:
            query = session.query(Reminder)
            
            # Apply date filters based on date_type
            if date_type == "today":
                tomorrow = today + timedelta(days=1)
                query = query.filter(
                    Reminder.reminder_date >= today,
                    Reminder.reminder_date < tomorrow
                )
                date_description = "today"
                
            elif date_type == "tomorrow":
                tomorrow = today + timedelta(days=1)
                day_after = tomorrow + timedelta(days=1)
                query = query.filter(
                    Reminder.reminder_date >= tomorrow,
                    Reminder.reminder_date < day_after
                )
                date_description = "tomorrow"
                
            elif date_type == "upcoming":
                query = query.filter(
                    Reminder.reminder_date >= today,
                    Reminder.completed.is_(False)
                )
                date_description = "upcoming"
                
            elif date_type == "overdue":
                # Overdue reminders: past due date and not completed
                query = query.filter(
                    Reminder.reminder_date < today,
                    Reminder.completed.is_(False)
                )
                date_description = "overdue"

            elif date_type == "past":
                query = query.filter(
                    Reminder.reminder_date < today
                )
                date_description = "past"

            elif date_type == "all":
                # No filter needed
                date_description = "all"
                
            elif date_type == "date":
                if not specific_date:
                    raise ToolError(
                        "specific_date is required when date_type is 'date'",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Use our timezone-aware date parser
                    parsed_date = self._parse_date(specific_date)
                    next_date = parsed_date + timedelta(days=1)
                    query = query.filter(
                        Reminder.reminder_date >= parsed_date,
                        Reminder.reminder_date < next_date
                    )
                    # Format the date in user timezone
                    user_tz = get_default_timezone()
                    date_str = format_datetime(parsed_date, "date", user_tz)
                    date_description = f"on {date_str}"
                except Exception as e:
                    raise ToolError(
                        f"Failed to parse specific_date '{specific_date}': {str(e)}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
            elif date_type == "range":
                if not start_date or not end_date:
                    raise ToolError(
                        "start_date and end_date are required when date_type is 'range'",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Use our timezone-aware date parser for both dates
                    parsed_start = self._parse_date(start_date)
                    # Include end date fully
                    parsed_end = self._parse_date(end_date) + timedelta(days=1)
                    query = query.filter(
                        Reminder.reminder_date >= parsed_start,
                        Reminder.reminder_date < parsed_end
                    )
                    # Format the dates in user timezone
                    user_tz = get_default_timezone()
                    start_str = format_datetime(parsed_start, "date", user_tz)
                    end_str = format_datetime(parsed_end - timedelta(days=1), "date", user_tz)
                    date_description = f"from {start_str} to {end_str}"
                except Exception as e:
                    raise ToolError(
                        f"Failed to parse date range: {str(e)}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Sort by reminder date
            query = query.order_by(Reminder.reminder_date)
            
            # Execute query and format results
            reminders = query.all()
            reminder_list = [reminder.to_dict() for reminder in reminders]
            
            return {
                "reminders": reminder_list,
                "count": len(reminder_list),
                "date_type": date_type,
                "message": f"Found {len(reminder_list)} reminder(s) {date_description}"
            }

    def _mark_completed(self, reminder_id: str) -> Dict[str, Any]:
        """
        Mark a reminder as completed.
        
        Args:
            reminder_id: ID of the reminder to mark as completed
            
        Returns:
            Dict containing the updated reminder
            
        Raises:
            ToolError: If reminder_id is invalid or not found
        """
        self.logger.info(f"Marking reminder {reminder_id} as completed")
        
        # Get the reminder
        reminder = self.db.get(Reminder, reminder_id)
        if not reminder:
            raise ToolError(
                f"Reminder with ID '{reminder_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Update reminder
        reminder.completed = True
        reminder.completed_at = utc_now()  # Use UTC-aware time
        
        # Save changes
        try:
            self.db.update(reminder)
            self.logger.info(f"Marked reminder {reminder_id} as completed")
        except Exception as e:
            self.logger.error(f"Error updating reminder: {e}")
            raise ToolError(
                f"Failed to update reminder: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        return {
            "reminder": reminder.to_dict(),
            "message": f"Reminder '{reminder.title}' marked as completed"
        }

    def _update_reminder(
        self,
        reminder_id: str,
        title: Optional[str] = None,
        date: Optional[str] = None,
        description: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_email: Optional[str] = None,
        contact_phone: Optional[str] = None,
        additional_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing reminder.
        
        Args:
            reminder_id: ID of the reminder to update
            title: New title (optional)
            date: New date (optional)
            description: New description (optional)
            contact_name: New contact name (optional)
            contact_email: New contact email (optional)
            contact_phone: New contact phone (optional)
            additional_notes: New additional notes (optional)
            
        Returns:
            Dict containing the updated reminder
            
        Raises:
            ToolError: If reminder_id is invalid or not found
        """
        self.logger.info(f"Updating reminder {reminder_id}")
        
        # Get the reminder
        reminder = self.db.get(Reminder, reminder_id)
        if not reminder:
            raise ToolError(
                f"Reminder with ID '{reminder_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Update fields if provided
        changes = []
        
        if title is not None:
            reminder.title = title
            changes.append("title")
            
        if date is not None:
            try:
                reminder.reminder_date = self._parse_date(date)
                changes.append("date")
            except Exception as e:
                raise ToolError(
                    f"Failed to parse date '{date}': {str(e)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
        if description is not None:
            reminder.description = description
            changes.append("description")
        
        # Contact information updates
        if contact_name is not None:
            reminder.contact_name = contact_name
            changes.append("contact_name")
            
            # Lookup customer if name was updated
            customer_info = self._lookup_customer(contact_name)
            if customer_info:
                customer = customer_info.get("customer", {})
                reminder.customer_id = customer.get("id")
                
                # Only update these if not explicitly provided
                if contact_email is None and "email_address" in customer:
                    reminder.contact_email = customer.get("email_address")
                    changes.append("contact_email")
                    
                if contact_phone is None and "phone_number" in customer:
                    reminder.contact_phone = customer.get("phone_number")
                    changes.append("contact_phone")
        
        if contact_email is not None:
            reminder.contact_email = contact_email
            changes.append("contact_email")
            
        if contact_phone is not None:
            reminder.contact_phone = contact_phone
            changes.append("contact_phone")
            
        if additional_notes is not None:
            reminder.additional_notes = additional_notes
            changes.append("additional_notes")
            
        # Save changes
        if changes:
            try:
                self.db.update(reminder)
                self.logger.info(f"Updated reminder {reminder_id} ({', '.join(changes)})")
            except Exception as e:
                self.logger.error(f"Error updating reminder: {e}")
                raise ToolError(
                    f"Failed to update reminder: {str(e)}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
        return {
            "reminder": reminder.to_dict(),
            "updated_fields": changes,
            "message": (
                f"Reminder updated: {', '.join(changes)}" if changes
                else "No changes made to reminder"
            )
        }

    def _delete_reminder(self, reminder_id: str) -> Dict[str, Any]:
        """
        Delete a reminder.
        
        Args:
            reminder_id: ID of the reminder to delete
            
        Returns:
            Dict containing deletion confirmation
            
        Raises:
            ToolError: If reminder_id is invalid or not found
        """
        self.logger.info(f"Deleting reminder {reminder_id}")
        
        # Get the reminder
        reminder = self.db.get(Reminder, reminder_id)
        if not reminder:
            raise ToolError(
                f"Reminder with ID '{reminder_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Store title for confirmation message
        title = reminder.title
        
        # Delete reminder
        try:
            self.db.delete(reminder)
            self.logger.info(f"Deleted reminder {reminder_id}")
        except Exception as e:
            self.logger.error(f"Error deleting reminder: {e}")
            raise ToolError(
                f"Failed to delete reminder: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        return {
            "id": reminder_id,
            "message": f"Reminder '{title}' deleted successfully"
        }

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse a date string into a datetime object, supporting natural language.
        
        Args:
            date_str: Date string in various formats ("tomorrow", "next Friday",
                "2025-05-01", etc.)
            
        Returns:
            timezone-aware datetime object representing the parsed date
            
        Raises:
            ValueError: If date parsing fails
        """
        # Use UTC for internal operations and only convert to user timezone when displaying
        today_utc = utc_now().replace(hour=12, minute=0, second=0, microsecond=0)
        # Store the user's timezone for later display
        user_tz = get_default_timezone()
        
        # Handle some common natural language cases
        date_str = date_str.lower().strip()
        
        if date_str == "today":
            return today_utc
            
        if date_str == "tomorrow":
            return today_utc + timedelta(days=1)
            
        if date_str.startswith("in "):
            # Handle "in X days/weeks/months/years" format
            parts = date_str.split()
            if len(parts) >= 3:
                try:
                    amount = int(parts[1])
                    unit = parts[2].lower()
                    
                    if unit.startswith("day"):
                        return today_utc + timedelta(days=amount)
                    elif unit.startswith("week"):
                        return today_utc + timedelta(weeks=amount)
                    elif unit.startswith("month"):
                        return today_utc + relativedelta(months=amount)
                    elif unit.startswith("year"):
                        return today_utc + relativedelta(years=amount)
                except ValueError:
                    pass  # Fall back to our parse_time_string utility
        
        # Use our timezone-aware parsing utility
        try:
            # Parse using the user's timezone, but then ensure it's converted to UTC for storage
            parsed_dt = parse_time_string(date_str, user_tz, today_utc)
            # Ensure the result is in UTC
            return ensure_utc(parsed_dt)
        except Exception as e:
            raise ValueError(f"Could not parse date: {date_str}. Error: {str(e)}")

    def _lookup_customer(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a customer by name in the customer database.
        
        Args:
            name: Customer name to search for
            
        Returns:
            Dict with customer info or None if not found
        """
        try:
            # Import here to avoid circular imports
            from tools.customerdatabase_tool import CustomerDatabaseTool
            
            # Create database tool instance
            customer_tool = CustomerDatabaseTool()
            
            # Search for customer by name
            return customer_tool.search_customers(query=name, category="name")
        except Exception as e:
            self.logger.warning(f"Customer lookup failed: {e}")
            return None