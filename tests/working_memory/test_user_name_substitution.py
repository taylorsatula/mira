"""
Tests for user name substitution in system prompt composition.

Tests the integration of user's first_name into the system prompt,
replacing "The User" with the actual name or gracefully falling back.
"""
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from working_memory.core import WorkingMemory
from cns.core.events import ComposeSystemPromptEvent


class TestUserNameSubstitution:
    """Tests for user name substitution during prompt composition."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        return MagicMock()

    @pytest.fixture
    def working_memory(self, mock_event_bus):
        """Create WorkingMemory instance with mock event bus."""
        return WorkingMemory(mock_event_bus)

    def test_user_name_substitution_with_first_name(self, working_memory, mock_event_bus):
        """USER_NAME is substituted for 'The User' when first_name is provided."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        # Mock get_current_user to return user data with first_name
        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': 'Taylor', 'user_id': user_id}

            # Create prompt with "The User"
            base_prompt = "You are working with The User. The User wants you to help."

            # Mock the composer's set_base_prompt method
            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                # Publish compose event (user_id is retrieved from context automatically)
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                # Verify that composer received prompt with "Taylor" instead of "The User"
                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]  # First positional argument
                assert "Taylor" in composed_prompt
                assert "The User" not in composed_prompt

    def test_user_name_fallback_when_first_name_is_none(self, working_memory, mock_event_bus):
        """'The User' is preserved when first_name is None."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        # Mock get_current_user to return user data without first_name
        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': None, 'user_id': user_id}

            base_prompt = "You are working with The User."

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                # Verify that composer received original prompt with "The User"
                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]
                assert "The User" in composed_prompt

    def test_user_name_fallback_when_first_name_is_empty(self, working_memory, mock_event_bus):
        """'The User' is preserved when first_name is empty string."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': '', 'user_id': user_id}

            base_prompt = "You are working with The User."

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]
                assert "The User" in composed_prompt

    def test_user_name_fallback_when_first_name_is_whitespace(self, working_memory, mock_event_bus):
        """'The User' is preserved when first_name is only whitespace."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': '   ', 'user_id': user_id}

            base_prompt = "You are working with The User."

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]
                assert "The User" in composed_prompt

    def test_all_occurrences_of_the_user_are_replaced(self, working_memory, mock_event_bus):
        """All occurrences of 'The User' are replaced with first_name."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': 'Alex', 'user_id': user_id}

            # Prompt with multiple occurrences (note: "The User's" is also replaced since it contains "The User")
            base_prompt = (
                "Your name is Mira.\n"
                "You're here for The User.\n"
                "The User values direct communication.\n"
                "Working with The User is a collaboration.\n"
                "The User's feedback helps you improve.\n"
            )

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]

                # Count occurrences in composed prompt
                alex_count = composed_prompt.count('Alex')
                user_count = composed_prompt.count('The User')

                # All 4 occurrences of "The User" should be replaced with "Alex"
                # (includes "The User's" where "The User" substring is replaced)
                assert alex_count == 4, f"Expected 4 'Alex' occurrences, got {alex_count}"
                assert user_count == 0, f"Expected 0 'The User' occurrences, got {user_count}"

    def test_first_name_with_special_characters(self, working_memory, mock_event_bus):
        """User names with special characters are handled correctly."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': "O'Brien", 'user_id': user_id}

            base_prompt = "You are working with The User."

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]
                assert "O'Brien" in composed_prompt
                assert "The User" not in composed_prompt

    def test_first_name_with_unicode(self, working_memory, mock_event_bus):
        """User names with unicode characters are handled correctly."""
        from utils.user_context import set_current_user_id

        user_id = str(uuid4())
        set_current_user_id(user_id)

        with patch('working_memory.core.get_current_user') as mock_get_user:
            mock_get_user.return_value = {'first_name': 'José', 'user_id': user_id}

            base_prompt = "You are working with The User."

            with patch.object(working_memory.composer, 'set_base_prompt') as mock_set_base:
                event = ComposeSystemPromptEvent.create(
                    continuum_id=str(uuid4()),
                    base_prompt=base_prompt
                )

                working_memory._handle_compose_prompt(event)

                assert mock_set_base.called
                composed_prompt = mock_set_base.call_args[0][0]
                assert "José" in composed_prompt
                assert "The User" not in composed_prompt


class TestUserInfoTrinketRemoved:
    """Tests verifying UserInfoTrinket has been removed."""

    def test_user_information_not_in_section_order(self):
        """CONTRACT: 'user_information' section is removed from composition order."""
        from working_memory.composer import ComposerConfig

        config = ComposerConfig()
        assert 'user_information' not in config.section_order

    def test_user_info_trinket_file_deleted(self):
        """CONTRACT: UserInfoTrinket module no longer exists."""
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('working_memory.trinkets.user_info_trinket')

    def test_factory_does_not_import_user_info_trinket(self):
        """CONTRACT: Factory does not attempt to import UserInfoTrinket."""
        # This is implicitly tested by the fact that factory.py doesn't have the import
        # If import was still there, importing factory would raise ModuleNotFoundError
        from cns.integration.factory import CNSIntegrationFactory

        # If this imports successfully, the UserInfoTrinket import has been removed
        assert CNSIntegrationFactory is not None
