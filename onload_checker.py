"""
On-load check system for the AI agent.

This module handles checks that should be run when the application first loads,
such as upcoming reminders, important notifications, etc.
"""
import logging
from datetime import datetime, timedelta
from typing import List

from errors import error_context, ErrorCode
from stimuli import Stimulus, StimulusType
from tools.reminder_tool import ReminderTool


class OnLoadChecker:
    """
    Manager for checks that should run when the application starts.

    Provides methods for running various checks and returning
    notifications that should be shown to the user.
    """

    def __init__(self):
        """Initialize the on-load checker."""
        self.logger = logging.getLogger("onload_checker")
        self.check_results = []

    def run_all_checks(self, conversation) -> List[Stimulus]:
        """
        Run all configured on-load checks.

        Args:
            conversation: The conversation to add stimuli to

        Returns:
            List of stimuli generated from the checks
        """
        self.logger.info("Running on-load checks")

        # List to hold all stimuli from checks
        all_stimuli = []

        # Run reminder check
        reminder_stimuli = self.check_reminders()
        if reminder_stimuli:
            all_stimuli.extend(reminder_stimuli)

        # Add more checks here in the future
        # example: system_stimuli = self.check_system_status()

        return all_stimuli

    def check_reminders(self) -> List[Stimulus]:
        """
        Check for upcoming reminders.

        Looks for reminders scheduled for today and the next three days.

        Returns:
            List of notification stimuli for upcoming reminders
        """
        stimuli = []

        with error_context(
            component_name="OnLoadChecker",
            operation="checking reminders",
            error_class=Exception,
            error_code=ErrorCode.UNKNOWN_ERROR,
            logger=self.logger
        ):
            # Create reminder tool instance
            reminder_tool = ReminderTool()

            # Check today's reminders
            today_result = reminder_tool.run(
                operation="get_reminders",
                date_type="today"
            )

            if today_result.get("count", 0) > 0:
                reminders = today_result.get("reminders", [])
                # Filter out completed reminders
                active_reminders = [
                    r for r in reminders if not r.get('completed', False)
                ]

                if active_reminders:
                    content = (f"You have {len(active_reminders)} "
                               f"reminder(s) for today:\n")
                    for idx, reminder in enumerate(active_reminders, 1):
                        date_str = datetime.fromisoformat(
                            reminder['reminder_date']
                        ).strftime('%I:%M %p')
                        content += (f"{idx}. {reminder['title']} "
                                   f"at {date_str}\n")

                # Only add a stimulus if there are active reminders
                if active_reminders:
                    stimuli.append(
                        Stimulus(
                            type=StimulusType.NOTIFICATION,
                            content=content.strip(),
                            source="reminder_system",
                            metadata={
                                "priority": "high",
                                "check_type": "today_reminders"
                            }
                        )
                    )

            # Check upcoming reminders (next 3 days)
            today = datetime.now()
            three_days_later = today + timedelta(days=3)

            upcoming_result = reminder_tool.run(
                operation="get_reminders",
                date_type="range",
                start_date=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=three_days_later.strftime("%Y-%m-%d")
            )

            if upcoming_result.get("count", 0) > 0:
                reminders = upcoming_result.get("reminders", [])

                # Filter out completed reminders
                active_reminders = [
                    r for r in reminders if not r.get('completed', False)
                ]

                # Only continue if there are active reminders
                if active_reminders:
                    # Group reminders by date
                    grouped_reminders = {}
                    for reminder in active_reminders:
                        reminder_date = datetime.fromisoformat(
                            reminder['reminder_date']
                        )
                        date_str = reminder_date.strftime("%A, %B %d")
                        if date_str not in grouped_reminders:
                            grouped_reminders[date_str] = []
                        grouped_reminders[date_str].append(reminder)

                # Only continue with creating the stimulus if we have active reminders
                if active_reminders:
                    # Format content with grouping
                    content = ("Upcoming reminders for the next few "
                               "days:\n")
                    for date_str, date_reminders in grouped_reminders.items():
                        content += f"\n{date_str}:\n"
                        for idx, reminder in enumerate(date_reminders, 1):
                            reminder_date = datetime.fromisoformat(
                                reminder['reminder_date']
                            )
                            time_str = reminder_date.strftime('%I:%M %p')
                            content += (f"{idx}. {reminder['title']} at "
                                        f"{time_str}\n")

                    stimuli.append(
                        Stimulus(
                            type=StimulusType.NOTIFICATION,
                            content=content.strip(),
                            source="reminder_system",
                            metadata={
                                "priority": "medium",
                                "check_type": "upcoming_reminders"
                            }
                        )
                    )

        return stimuli

    # Add more check methods as needed, for example:
    # def check_system_status(self) -> List[Stimulus]:
    #     """Check system status."""
    #     return []


def add_stimuli_to_conversation(stimuli: List[Stimulus], conversation) -> None:
    """
    Add stimuli to a conversation as assistant messages.

    Args:
        stimuli: List of stimuli to add
        conversation: The conversation to add the stimuli to
    """
    for stimulus in stimuli:
        # Format as plain text for user notification
        conversation.add_message(
            "assistant",
            stimulus.content,
            {"is_notification": True, **stimulus.metadata}
        )
