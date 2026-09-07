"""
Tests for WorkingMemory trinket access and ProactiveMemoryTrinket cache.

Focus: Testing get_trinket() registry access and get_cached_memories() interface.
"""
import pytest
from unittest.mock import MagicMock


class TestWorkingMemoryTrinketAccess:
    """Tests for WorkingMemory.get_trinket() method."""

    def test_get_trinket_returns_registered_trinket(self):
        """CONTRACT: Returns trinket instance when registered by name."""
        from working_memory.core import WorkingMemory

        # Create WorkingMemory with mock event bus
        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)

        # Create and register a mock trinket
        mock_trinket = MagicMock()
        mock_trinket.__class__.__name__ = "TestTrinket"
        wm._trinkets["TestTrinket"] = mock_trinket

        result = wm.get_trinket("TestTrinket")

        assert result is mock_trinket

    def test_get_trinket_returns_none_for_unknown(self):
        """CONTRACT: Returns None for unregistered trinket name."""
        from working_memory.core import WorkingMemory

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)

        result = wm.get_trinket("NonExistentTrinket")

        assert result is None

    def test_get_trinket_with_proactive_memory_trinket(self):
        """CONTRACT: ProactiveMemoryTrinket can be accessed by name."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)

        # Create real trinket
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # Trinket auto-registers via __init__, so it should be accessible
        result = wm.get_trinket("ProactiveMemoryTrinket")

        assert result is trinket


class TestProactiveMemoryTrinketCache:
    """Tests for ProactiveMemoryTrinket.get_cached_memories() method."""

    def test_cache_empty_initially(self):
        """CONTRACT: New trinket has empty cache."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        assert trinket.get_cached_memories() == []

    def test_get_cached_memories_returns_cache(self):
        """CONTRACT: Returns internal _cached_memories list."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # Manually set cache
        expected_memories = [
            {"id": "1", "text": "Memory A"},
            {"id": "2", "text": "Memory B"},
        ]
        trinket._cached_memories = expected_memories

        result = trinket.get_cached_memories()

        assert result == expected_memories

    def test_cache_updates_on_generate_content_with_memories(self):
        """CONTRACT: generate_content() updates cache when context has 'memories'."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # Initially empty
        assert trinket.get_cached_memories() == []

        # Call generate_content with memories in context
        new_memories = [
            {"id": "1", "text": "New memory", "importance_score": 0.8},
            {"id": "2", "text": "Another memory", "importance_score": 0.7},
            {"id": "3", "text": "Third memory", "importance_score": 0.6},
        ]

        trinket.generate_content({"memories": new_memories})

        # Cache should be updated
        assert len(trinket.get_cached_memories()) == 3
        assert trinket.get_cached_memories() == new_memories

    def test_cache_persists_when_context_has_no_memories(self):
        """CONTRACT: Cache not cleared when context doesn't have 'memories' key."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # Set initial cache
        initial_memories = [{"id": "1", "text": "Initial memory"}]
        trinket._cached_memories = initial_memories

        # Call generate_content without 'memories' key
        trinket.generate_content({"other_key": "value"})

        # Cache should be unchanged
        assert trinket.get_cached_memories() == initial_memories

    def test_returns_same_list_reference(self):
        """CONTRACT: get_cached_memories() returns actual internal list (not copy)."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        trinket._cached_memories = [{"id": "1", "text": "Test"}]

        result1 = trinket.get_cached_memories()
        result2 = trinket.get_cached_memories()

        # Same reference
        assert result1 is result2
        assert result1 is trinket._cached_memories


class TestTrinketCacheIntegration:
    """Integration tests for trinket cache flow."""

    def test_cache_survives_multiple_updates(self):
        """CONTRACT: Cache updates correctly across multiple generate_content calls."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # First update
        memories_v1 = [{"id": "1", "text": "First"}]
        trinket.generate_content({"memories": memories_v1})
        assert trinket.get_cached_memories() == memories_v1

        # Second update
        memories_v2 = [{"id": "2", "text": "Second"}, {"id": "3", "text": "Third"}]
        trinket.generate_content({"memories": memories_v2})
        assert trinket.get_cached_memories() == memories_v2

        # Third update - empty list
        memories_v3 = []
        trinket.generate_content({"memories": memories_v3})
        assert trinket.get_cached_memories() == []

    def test_orchestrator_can_access_cached_memories_through_working_memory(self):
        """CONTRACT: Full flow - orchestrator accesses cache via working_memory.get_trinket()."""
        from working_memory.core import WorkingMemory
        from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket

        mock_event_bus = MagicMock()
        wm = WorkingMemory(mock_event_bus)
        trinket = ProactiveMemoryTrinket(mock_event_bus, wm)

        # Set some cached memories
        memories = [{"id": "1", "text": "Cached memory"}]
        trinket._cached_memories = memories

        # Simulate orchestrator access pattern
        retrieved_trinket = wm.get_trinket("ProactiveMemoryTrinket")
        assert retrieved_trinket is not None
        assert hasattr(retrieved_trinket, 'get_cached_memories')

        cached = retrieved_trinket.get_cached_memories()
        assert cached == memories
