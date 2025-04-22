"""
Unit tests for the onload_checker module.
"""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from onload_checker import OnLoadChecker, add_stimuli_to_conversation
from stimuli import Stimulus, StimulusType
from conversation import Conversation


class TestOnLoadChecker(unittest.TestCase):
    """Test cases for the OnLoadChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = OnLoadChecker()
        self.mock_conversation = MagicMock(spec=Conversation)

    @patch('onload_checker.ReminderTool')
    def test_check_reminders_today(self, mock_reminder_tool_class):
        """Test checking for today's reminders."""
        # Set up mock reminder tool
        mock_reminder_tool = MagicMock()
        mock_reminder_tool_class.return_value = mock_reminder_tool

        # Set up mock reminder results
        today = datetime.now()
        mock_reminder_tool.run.return_value = {
            "count": 2,
            "reminders": [
                {
                    "id": "rem_12345",
                    "title": "Team meeting",
                    "reminder_date": today.replace(
                        hour=14, minute=0).isoformat()
                },
                {
                    "id": "rem_67890",
                    "title": "Call client",
                    "reminder_date": today.replace(
                        hour=16, minute=30).isoformat()
                }
            ]
        }

        # Run the test
        stimuli = self.checker.check_reminders()

        # Verify results
        self.assertEqual(len(stimuli), 1)
        self.assertEqual(stimuli[0].type, StimulusType.NOTIFICATION)
        self.assertEqual(stimuli[0].source, "reminder_system")
        self.assertIn("2 reminder(s) for today", stimuli[0].content)
        self.assertIn("Team meeting", stimuli[0].content)
        self.assertIn("Call client", stimuli[0].content)

    @patch('onload_checker.ReminderTool')
    def test_check_reminders_upcoming(self, mock_reminder_tool_class):
        """Test checking for upcoming reminders."""
        # Set up mock reminder tool
        mock_reminder_tool = MagicMock()
        mock_reminder_tool_class.return_value = mock_reminder_tool

        # Set up mock return values for get_reminders
        today_result = {"count": 0, "reminders": []}

        # Set up upcoming reminders
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        day_after = today + timedelta(days=2)

        upcoming_result = {
            "count": 2,
            "reminders": [
                {
                    "id": "rem_12345",
                    "title": "Project deadline",
                    "reminder_date": tomorrow.replace(
                        hour=9, minute=0).isoformat()
                },
                {
                    "id": "rem_67890",
                    "title": "Team lunch",
                    "reminder_date": day_after.replace(
                        hour=12, minute=30).isoformat()
                }
            ]
        }

        # Configure mock to return different results for different calls
        mock_reminder_tool.run.side_effect = lambda **kwargs: (
            today_result if kwargs.get('date_type') == 'today' 
            else upcoming_result
        )

        # Run the test
        stimuli = self.checker.check_reminders()

        # Verify results
        # Only upcoming, no today reminders
        self.assertEqual(len(stimuli), 1)
        self.assertEqual(stimuli[0].type, StimulusType.NOTIFICATION)
        self.assertIn("Upcoming reminders", stimuli[0].content)
        self.assertIn("Project deadline", stimuli[0].content)
        self.assertIn("Team lunch", stimuli[0].content)

    @patch('onload_checker.ReminderTool')
    def test_run_all_checks(self, mock_reminder_tool_class):
        """Test running all checks."""
        # Set up mock reminder tool
        mock_reminder_tool = MagicMock()
        mock_reminder_tool_class.return_value = mock_reminder_tool

        # Set up mock reminder results - empty for simplicity
        mock_reminder_tool.run.return_value = {"count": 0, "reminders": []}

        # Mock conversation
        mock_conversation = MagicMock()

        # Run all checks
        stimuli = self.checker.run_all_checks(mock_conversation)

        # Verify no stimuli (since mock returns empty reminders)
        self.assertEqual(len(stimuli), 0)

        # Verify that the reminder check was run
        mock_reminder_tool.run.assert_called()

    def test_add_stimuli_to_conversation(self):
        """Test adding stimuli to conversation."""
        # Create test stimuli
        stimulus1 = Stimulus(
            type=StimulusType.NOTIFICATION,
            content="Test notification 1",
            source="test_source",
            metadata={"priority": "high"}
        )
        stimulus2 = Stimulus(
            type=StimulusType.NOTIFICATION,
            content="Test notification 2",
            source="test_source",
            metadata={"priority": "medium"}
        )

        # Add to mock conversation
        add_stimuli_to_conversation([stimulus1, stimulus2], self.mock_conversation)

        # Verify that add_message was called correctly
        self.assertEqual(self.mock_conversation.add_message.call_count, 2)

        # Check first call
        args1, kwargs1 = self.mock_conversation.add_message.call_args_list[0]
        self.assertEqual(args1[0], "assistant")
        self.assertEqual(args1[1], "Test notification 1")
        self.assertTrue(kwargs1["metadata"]["is_notification"])
        self.assertEqual(kwargs1["metadata"]["priority"], "high")

        # Check second call
        args2, kwargs2 = self.mock_conversation.add_message.call_args_list[1]
        self.assertEqual(args2[0], "assistant")
        self.assertEqual(args2[1], "Test notification 2")
        self.assertTrue(kwargs2["metadata"]["is_notification"])
        self.assertEqual(kwargs2["metadata"]["priority"], "medium")


if __name__ == '__main__':
    unittest.main()
