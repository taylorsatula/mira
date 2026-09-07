"""
Tests for Notification Center implementation.

Focus: TrinketPlacement routing, composer SectionData consolidation,
and notification center content generation.
"""
import pytest

from tests.fixtures.core import TEST_USER_ID


@pytest.fixture
def user_context():
    """Set up user context for event creation tests."""
    from utils.user_context import set_current_user_id, clear_user_context

    set_current_user_id(TEST_USER_ID)
    yield TEST_USER_ID
    clear_user_context()


class TestTrinketPlacement:
    """Tests for TrinketPlacement enum and registry."""

    def test_placement_enum_values(self):
        """CONTRACT: Enum has expected string values."""
        from working_memory.trinkets.base import TrinketPlacement

        assert TrinketPlacement.SYSTEM.value == "system"
        assert TrinketPlacement.NOTIFICATION_CENTER.value == "notification"

    def test_notification_center_registry_contains_expected_trinkets(self):
        """CONTRACT: Registry lists all notification center trinkets."""
        from working_memory.trinkets.base import _NOTIFICATION_CENTER_TRINKETS

        expected = {
            'TimeManager',
            'ManifestTrinket',
            'ReminderManager',
            'GetContextTrinket',
            'ProactiveMemoryTrinket',
        }
        assert _NOTIFICATION_CENTER_TRINKETS == expected

    def test_notification_center_registry_is_frozen(self):
        """CONTRACT: Registry is immutable frozenset."""
        from working_memory.trinkets.base import _NOTIFICATION_CENTER_TRINKETS

        assert isinstance(_NOTIFICATION_CENTER_TRINKETS, frozenset)


class TestEventAwareTrinketPlacement:
    """Tests for placement property on EventAwareTrinket."""

    def test_system_trinket_has_system_placement(self, event_bus):
        """CONTRACT: Trinkets not in registry have SYSTEM placement."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.base import TrinketPlacement
        from working_memory.trinkets.tool_guidance_trinket import ToolGuidanceTrinket

        wm = WorkingMemory(event_bus)
        trinket = ToolGuidanceTrinket(event_bus, wm)

        assert trinket.placement == TrinketPlacement.SYSTEM

    def test_notification_trinket_has_notification_placement(self, event_bus):
        """CONTRACT: Trinkets in registry have NOTIFICATION_CENTER placement."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.base import TrinketPlacement
        from working_memory.trinkets.time_manager import TimeManager

        wm = WorkingMemory(event_bus)
        trinket = TimeManager(event_bus, wm)

        assert trinket.placement == TrinketPlacement.NOTIFICATION_CENTER

    def test_all_notification_trinkets_return_correct_placement(self, event_bus):
        """CONTRACT: All 5 notification center trinkets have correct placement."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.base import TrinketPlacement
        from working_memory.trinkets.time_manager import TimeManager
        from working_memory.trinkets.manifest_trinket import ManifestTrinket
        from working_memory.trinkets.reminder_manager import ReminderManager
        from working_memory.trinkets.getcontext_trinket import GetContextTrinket
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        wm = WorkingMemory(event_bus)

        trinket_classes = [
            TimeManager,
            ManifestTrinket,
            ReminderManager,
            GetContextTrinket,
            ProactiveMemoryTrinket,
        ]

        for cls in trinket_classes:
            trinket = cls(event_bus, wm)
            assert trinket.placement == TrinketPlacement.NOTIFICATION_CENTER, \
                f"{cls.__name__} should have NOTIFICATION_CENTER placement"

    def test_domaindoc_trinket_has_system_placement(self, event_bus):
        """CONTRACT: DomaindocTrinket has SYSTEM placement (cached in system prompt)."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.base import TrinketPlacement
        from working_memory.trinkets.domaindoc_trinket import DomaindocTrinket

        wm = WorkingMemory(event_bus)
        trinket = DomaindocTrinket(event_bus, wm)

        assert trinket.placement == TrinketPlacement.SYSTEM


class TestSectionData:
    """Tests for SectionData NamedTuple."""

    def test_section_data_fields(self):
        """CONTRACT: SectionData has content, cache_policy, placement fields."""
        from working_memory.composer import SectionData

        section = SectionData(
            content="test content",
            cache_policy=True,
            placement="system"
        )

        assert section.content == "test content"
        assert section.cache_policy is True
        assert section.placement == "system"

    def test_section_data_is_immutable(self):
        """CONTRACT: SectionData is a NamedTuple (immutable)."""
        from working_memory.composer import SectionData

        section = SectionData(content="x", cache_policy=False, placement="notification")

        with pytest.raises(AttributeError):
            section.content = "modified"


class TestComposerPlacementConstants:
    """Tests for placement constants."""

    def test_placement_constants_match_enum(self):
        """CONTRACT: Constants match TrinketPlacement enum values."""
        from working_memory.composer import PLACEMENT_SYSTEM, PLACEMENT_NOTIFICATION
        from working_memory.trinkets.base import TrinketPlacement

        assert PLACEMENT_SYSTEM == TrinketPlacement.SYSTEM.value
        assert PLACEMENT_NOTIFICATION == TrinketPlacement.NOTIFICATION_CENTER.value


class TestComposerAddSection:
    """Tests for SystemPromptComposer.add_section()."""

    def test_add_section_stores_section_data(self):
        """CONTRACT: add_section stores SectionData with all fields."""
        from working_memory.composer import SystemPromptComposer, SectionData

        composer = SystemPromptComposer()
        composer.add_section("test", "content", cache_policy=True, placement="notification")

        assert "test" in composer._sections
        section = composer._sections["test"]
        assert isinstance(section, SectionData)
        assert section.content == "content"
        assert section.cache_policy is True
        assert section.placement == "notification"

    def test_add_section_skips_empty_content(self):
        """CONTRACT: Empty content is not stored."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.add_section("empty", "", cache_policy=False, placement="system")
        composer.add_section("whitespace", "   \n  ", cache_policy=False, placement="system")

        assert "empty" not in composer._sections
        assert "whitespace" not in composer._sections

    def test_set_base_prompt_uses_system_placement(self):
        """CONTRACT: Base prompt always has SYSTEM placement and cache_policy=True."""
        from working_memory.composer import SystemPromptComposer, PLACEMENT_SYSTEM

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base prompt content")

        section = composer._sections["base_prompt"]
        assert section.placement == PLACEMENT_SYSTEM
        assert section.cache_policy is True


class TestComposerRouting:
    """Tests for compose() routing by placement."""

    def test_system_cached_sections_go_to_cached_content(self):
        """CONTRACT: System placement + cache_policy=True → cached_content."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base prompt")
        composer.add_section("tool_guidance", "Tool guidance", cache_policy=True, placement="system")

        result = composer.compose()

        assert "Base prompt" in result["cached_content"]
        assert "Tool guidance" in result["cached_content"]
        assert result["notification_center"] == ""

    def test_system_uncached_sections_go_to_non_cached_content(self):
        """CONTRACT: System placement + cache_policy=False → non_cached_content."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base")
        # Use tool_hints which is in section_order and has system placement
        composer.add_section("tool_hints", "Tool hints content", cache_policy=False, placement="system")

        result = composer.compose()

        assert "Tool hints content" in result["non_cached_content"]
        assert "Tool hints content" not in result["cached_content"]

    def test_notification_sections_go_to_notification_center(self):
        """CONTRACT: Notification placement → notification_center."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base")
        composer.add_section("datetime_section", "Today is Friday", placement="notification")
        composer.add_section("active_reminders", "You have 2 reminders", placement="notification")

        result = composer.compose()

        assert "Today is Friday" in result["notification_center"]
        assert "You have 2 reminders" in result["notification_center"]
        assert "Today is Friday" not in result["cached_content"]
        assert "Today is Friday" not in result["non_cached_content"]


class TestNotificationCenterFormatting:
    """Tests for notification center content formatting."""

    def test_notification_center_has_header(self):
        """CONTRACT: Notification center has ═══ delimiters and header."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base")
        composer.add_section("datetime_section", "Time info", placement="notification")

        result = composer.compose()
        nc = result["notification_center"]

        assert "═" * 60 in nc
        assert "NOTIFICATION CENTER" in nc
        assert "front-of-mind" in nc

    def test_empty_notification_center_returns_empty_string(self):
        """CONTRACT: No notification sections → empty string (not formatted header)."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base")
        composer.add_section("tool_guidance", "Tools", cache_policy=True, placement="system")

        result = composer.compose()

        assert result["notification_center"] == ""

    def test_notification_center_preserves_section_order(self):
        """CONTRACT: Sections appear in config.section_order order."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base")
        # Add in reverse order of section_order
        composer.add_section("relevant_memories", "Memories", placement="notification")
        composer.add_section("datetime_section", "Time", placement="notification")

        result = composer.compose()
        nc = result["notification_center"]

        # datetime_section comes before relevant_memories in section_order
        time_pos = nc.find("Time")
        memories_pos = nc.find("Memories")
        assert time_pos < memories_pos


class TestClearSections:
    """Tests for clear_sections() with SectionData."""

    def test_clear_preserves_base_prompt(self):
        """CONTRACT: clear_sections(preserve_base=True) keeps base_prompt."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base prompt")
        composer.add_section("other", "Other content", placement="notification")

        composer.clear_sections(preserve_base=True)

        assert "base_prompt" in composer._sections
        assert "other" not in composer._sections

    def test_clear_removes_base_prompt_when_not_preserved(self):
        """CONTRACT: clear_sections(preserve_base=False) removes everything."""
        from working_memory.composer import SystemPromptComposer

        composer = SystemPromptComposer()
        composer.set_base_prompt("Base prompt")

        composer.clear_sections(preserve_base=False)

        assert "base_prompt" not in composer._sections


class TestTrinketContentEventPlacement:
    """Tests for placement field in TrinketContentEvent."""

    def test_event_has_placement_field(self, user_context):
        """CONTRACT: TrinketContentEvent has placement field."""
        from cns.core.events import TrinketContentEvent

        event = TrinketContentEvent.create(
            continuum_id="test-123",
            variable_name="test_section",
            content="Test content",
            trinket_name="TestTrinket",
            cache_policy=False,
            placement="notification"
        )
        assert event.placement == "notification"

    def test_event_placement_defaults_to_system(self, user_context):
        """CONTRACT: Default placement is 'system'."""
        from cns.core.events import TrinketContentEvent

        event = TrinketContentEvent.create(
            continuum_id="test-123",
            variable_name="test_section",
            content="Test content",
            trinket_name="TestTrinket"
        )
        assert event.placement == "system"


class TestSystemPromptComposedEventNotificationCenter:
    """Tests for notification_center field in SystemPromptComposedEvent."""

    def test_event_has_notification_center_field(self, user_context):
        """CONTRACT: SystemPromptComposedEvent has notification_center field."""
        from cns.core.events import SystemPromptComposedEvent

        event = SystemPromptComposedEvent.create(
            continuum_id="test-123",
            cached_content="cached",
            non_cached_content="non-cached",
            notification_center="notification content"
        )
        assert event.notification_center == "notification content"

    def test_event_notification_center_defaults_to_empty(self, user_context):
        """CONTRACT: notification_center defaults to empty string."""
        from cns.core.events import SystemPromptComposedEvent

        event = SystemPromptComposedEvent.create(
            continuum_id="test-123",
            cached_content="cached",
            non_cached_content="non-cached"
        )
        assert event.notification_center == ""
