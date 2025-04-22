"""
Tests for the ReminderTool implementation.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tools.reminder_tool import ReminderTool, Reminder
from errors import ToolError


class TestReminderTool(unittest.TestCase):
    """Tests for the ReminderTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = ReminderTool()
        
        # Mock the database
        self.patcher = patch('tools.reminder_tool.Database')
        self.mock_db_class = self.patcher.start()
        self.mock_db = MagicMock()
        self.mock_db_class.return_value = self.mock_db
        
        # Set up a session mock
        self.mock_session = MagicMock()
        self.mock_db.get_session.return_value.__enter__.return_value = self.mock_session
        
        # Replace the tool's db with our mock
        self.tool.db = self.mock_db

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()

    def test_add_reminder_success(self):
        """Test successful reminder creation."""
        # Set up mocks
        self.mock_db.add.return_value = None
        
        # Call the tool
        result = self.tool.run(
            operation="add_reminder",
            title="Test Reminder",
            date="tomorrow",
            description="Test description"
        )
        
        # Verify results
        self.assertIn("reminder", result)
        self.assertIn("id", result["reminder"])
        self.assertEqual(result["reminder"]["title"], "Test Reminder")
        self.assertEqual(result["reminder"]["description"], "Test description")
        
        # Verify the database add was called
        self.mock_db.add.assert_called_once()

    def test_add_reminder_with_natural_language_date(self):
        """Test reminder creation with natural language date parsing."""
        # Set up mocks
        self.mock_db.add.return_value = None
        
        # Call the tool with natural language date
        result = self.tool.run(
            operation="add_reminder",
            title="Test Reminder",
            date="in 3 weeks",
            description="Test description"
        )
        
        # Verify results
        today = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        expected_date = today + timedelta(weeks=3)
        
        # Get the date from the result and compare with expected
        reminder_date = datetime.fromisoformat(result["reminder"]["reminder_date"])
        date_diff = abs((reminder_date - expected_date).total_seconds())
        
        # Allow a small time difference (a few seconds) due to test execution time
        self.assertLess(date_diff, 5)  # Less than 5 seconds difference
        
        # Verify the database add was called
        self.mock_db.add.assert_called_once()

    def test_add_reminder_with_customer_integration(self):
        """Test reminder creation with customer database integration."""
        # Set up mocks
        self.mock_db.add.return_value = None
        
        # Mock the customer lookup
        customer_data = {
            "customer": {
                "id": "cust_12345",
                "given_name": "John",
                "family_name": "Doe",
                "email_address": "john.doe@example.com",
                "phone_number": "555-123-4567"
            }
        }
        
        with patch.object(self.tool, '_lookup_customer', return_value=customer_data):
            # Call the tool
            result = self.tool.run(
                operation="add_reminder",
                title="Test Reminder",
                date="tomorrow",
                description="Test description",
                contact_name="John Doe"
            )
            
            # Verify results
            self.assertIn("reminder", result)
            self.assertEqual(result["reminder"]["contact_name"], "John Doe")
            self.assertEqual(result["reminder"]["contact_email"], "john.doe@example.com")
            self.assertEqual(result["reminder"]["contact_phone"], "555-123-4567")
            self.assertEqual(result["reminder"]["customer_id"], "cust_12345")
            
            # Verify customer info in response
            self.assertTrue(result["customer_found"])
            self.assertEqual(result["customer_info"]["id"], "cust_12345")

    def test_add_reminder_invalid_date(self):
        """Test error handling with invalid date."""
        # Set up mocks to simulate date parsing failure
        with patch.object(self.tool, '_parse_date', side_effect=ValueError("Invalid date")):
            # Call the tool
            with self.assertRaises(ToolError) as context:
                self.tool.run(
                    operation="add_reminder",
                    title="Test Reminder",
                    date="invalid date",
                    description="Test description"
                )
                
            # Verify error code
            self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")

    def test_get_reminders_today(self):
        """Test retrieving today's reminders."""
        # Create test data
        today = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        reminder1 = Reminder(id="rem_1", title="Reminder 1", reminder_date=today)
        reminder2 = Reminder(id="rem_2", title="Reminder 2", reminder_date=today + timedelta(hours=2))
        
        # Set up query mock
        mock_query = MagicMock()
        self.mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [reminder1, reminder2]
        
        # Call the tool
        result = self.tool.run(
            operation="get_reminders",
            date_type="today"
        )
        
        # Verify results
        self.assertIn("reminders", result)
        self.assertEqual(len(result["reminders"]), 2)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["date_type"], "today")
        
        # Verify the query was called with appropriate filters
        self.mock_session.query.assert_called_with(Reminder)
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called()

    def test_get_reminders_date_range(self):
        """Test retrieving reminders in a date range."""
        # Create test data
        reminder1 = Reminder(id="rem_1", title="Reminder 1", reminder_date=datetime(2025, 5, 1))
        reminder2 = Reminder(id="rem_2", title="Reminder 2", reminder_date=datetime(2025, 5, 5))
        
        # Set up query mock
        mock_query = MagicMock()
        self.mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [reminder1, reminder2]
        
        # Mock date parsing
        with patch.object(self.tool, '_parse_date', side_effect=[
            datetime(2025, 5, 1),  # start_date
            datetime(2025, 5, 10)  # end_date
        ]):
            # Call the tool
            result = self.tool.run(
                operation="get_reminders",
                date_type="range",
                start_date="2025-05-01",
                end_date="2025-05-10"
            )
            
            # Verify results
            self.assertIn("reminders", result)
            self.assertEqual(len(result["reminders"]), 2)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["date_type"], "range")

    def test_mark_completed(self):
        """Test marking a reminder as completed."""
        # Create test data
        reminder = Reminder(id="rem_1", title="Reminder 1", reminder_date=datetime.now())
        
        # Set up mocks
        self.mock_db.get.return_value = reminder
        self.mock_db.update.return_value = None
        
        # Call the tool
        result = self.tool.run(
            operation="mark_completed",
            reminder_id="rem_1"
        )
        
        # Verify results
        self.assertIn("reminder", result)
        self.assertEqual(result["reminder"]["id"], "rem_1")
        self.assertTrue(result["reminder"]["completed"])
        self.assertIsNotNone(result["reminder"]["completed_at"])
        
        # Verify database calls
        self.mock_db.get.assert_called_with(Reminder, "rem_1")
        self.mock_db.update.assert_called_once()

    def test_update_reminder(self):
        """Test updating a reminder."""
        # Create test data
        reminder = Reminder(
            id="rem_1",
            title="Original Title",
            description="Original Description",
            reminder_date=datetime.now(),
            contact_name="Original Contact"
        )
        
        # Set up mocks
        self.mock_db.get.return_value = reminder
        self.mock_db.update.return_value = None
        
        # Call the tool
        result = self.tool.run(
            operation="update_reminder",
            reminder_id="rem_1",
            title="Updated Title",
            description="Updated Description"
        )
        
        # Verify results
        self.assertIn("reminder", result)
        self.assertEqual(result["reminder"]["id"], "rem_1")
        self.assertEqual(result["reminder"]["title"], "Updated Title")
        self.assertEqual(result["reminder"]["description"], "Updated Description")
        self.assertEqual(result["reminder"]["contact_name"], "Original Contact")  # Unchanged
        
        # Verify database calls
        self.mock_db.get.assert_called_with(Reminder, "rem_1")
        self.mock_db.update.assert_called_once()
        
        # Verify updated fields in response
        self.assertIn("updated_fields", result)
        self.assertIn("title", result["updated_fields"])
        self.assertIn("description", result["updated_fields"])

    def test_delete_reminder(self):
        """Test deleting a reminder."""
        # Create test data
        reminder = Reminder(id="rem_1", title="Test Reminder", reminder_date=datetime.now())
        
        # Set up mocks
        self.mock_db.get.return_value = reminder
        self.mock_db.delete.return_value = None
        
        # Call the tool
        result = self.tool.run(
            operation="delete_reminder",
            reminder_id="rem_1"
        )
        
        # Verify results
        self.assertEqual(result["id"], "rem_1")
        self.assertIn("message", result)
        self.assertIn("Test Reminder", result["message"])
        
        # Verify database calls
        self.mock_db.get.assert_called_with(Reminder, "rem_1")
        self.mock_db.delete.assert_called_once()

    def test_parse_date(self):
        """Test date parsing functionality."""
        # Test cases for natural language dates
        test_cases = [
            ("today", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)),
            ("tomorrow", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)),
            ("in 3 days", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=3)),
            ("in 2 weeks", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(weeks=2)),
            ("in 6 months", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + relativedelta(months=6)),
            ("in 1 year", datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + relativedelta(years=1)),
        ]
        
        for date_str, expected_date in test_cases:
            parsed_date = self.tool._parse_date(date_str)
            date_diff = abs((parsed_date - expected_date).total_seconds())
            self.assertLess(date_diff, 5, f"Failed for date string: {date_str}")

    def test_invalid_operation(self):
        """Test error handling for invalid operation."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="non_existent_operation",
                param="value"
            )
            
        self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")
        self.assertIn("Unknown operation", str(context.exception))


if __name__ == '__main__':
    unittest.main()