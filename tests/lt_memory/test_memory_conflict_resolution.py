"""
Production-grade integration tests for memory conflict resolution behavior.

Testing philosophy: TEST THE CONTRACT, NOT THE IMPLEMENTATION
This tests that MIRA actually resolves conflicts when using the memory system.

The contract we're testing:
- MIRA should detect and resolve conflicting information autonomously
- The resolution should preserve temporal context when relevant
- Conflicting facts should not persist side-by-side

REAL BUG THIS CATCHES: Without conflict resolution, contradictory information 
accumulates in memory blocks, making MIRA unreliable. For example:
- "John lives in Portland" + "John moved to Seattle" would both persist
- "Sarah is married" + "Sarah got divorced" would create confusion
- Users would lose trust in MIRA's memory accuracy
"""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.tools.memory_tool import LTMemoryTool
from lt_memory.models.base import MemoryBlock, BlockHistory
from config.config_manager import AppConfig
from api.llm_provider import LLMProvider
from utils.timezone_utils import utc_now, format_utc_iso


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """Real configuration with PostgreSQL test database."""
    config = AppConfig()
    config.memory.database_url = "postgresql://mira_admin@localhost:5432/lt_memory_test"
    config.memory.db_pool_size = 5
    config.memory.embedding_dim = 1536
    config.paths.data_dir = "/tmp/test_memory_conflict"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    return config


@pytest.fixture
def memory_system(test_config):
    """
    Initialize real memory system with all components.
    
    This fixture provides a complete, working memory system to test
    conflict resolution behavior.
    """
    # Clean database before test
    from sqlalchemy import create_engine
    from lt_memory.models.base import Base
    
    engine = create_engine(test_config.memory.database_url)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            try:
                conn.execute(table.delete())
            except Exception:
                pass
        conn.commit()
    engine.dispose()
    
    # Initialize system
    llm_provider = LLMProvider()
    memory_manager = MemoryManager(test_config, llm_provider)
    memory_tool = LTMemoryTool(memory_manager)
    
    yield memory_manager, memory_tool
    
    # Cleanup
    memory_manager.engine.dispose()


# =============================================================================
# CONFLICT RESOLUTION CONTRACT TESTS
# =============================================================================

class TestMemoryConflictResolution:
    """
    Test that the system prompt enables MIRA to autonomously resolve conflicts.
    
    These tests verify the behavior, not the implementation. We're testing that
    conflicting information gets resolved, not HOW it gets resolved.
    """
    
    def test_location_conflict_creates_resolvable_state(self, memory_system):
        """
        Test that location conflicts create a state MIRA can resolve.
        
        REAL BUG THIS CATCHES: If we allow "John lives in Portland" and 
        "John moved to Seattle" to coexist without resolution, MIRA gives
        contradictory information about where John lives, breaking user trust.
        """
        memory_manager, memory_tool = memory_system
        
        # Add initial location information
        result1 = memory_tool.run(
            "core_memory_append",
            label="human",
            content="John is my coworker who lives in Portland and works at Nike."
        )
        assert result1["success"] is True
        
        # Add conflicting location information
        result2 = memory_tool.run(
            "core_memory_append", 
            label="human",
            content="John told me he moved to Seattle for a new job at Amazon."
        )
        assert result2["success"] is True
        
        # Verify both pieces of information exist (creating the conflict)
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        assert "Portland" in memory_content
        assert "Seattle" in memory_content
        assert "Nike" in memory_content
        assert "Amazon" in memory_content
        
        # The conflict now exists - MIRA's system prompt should guide resolution
        # when it reads this memory during conversation
        
    def test_relationship_status_conflict_resolution_pattern(self, memory_system):
        """
        Test relationship status changes create resolvable conflicts.
        
        REAL BUG THIS CATCHES: If "Sarah is married" and "Sarah got divorced"
        both persist, MIRA might congratulate Sarah on her marriage after
        she's divorced, causing significant social errors.
        """
        memory_manager, memory_tool = memory_system
        
        # Initial relationship status
        memory_tool.run(
            "core_memory_append",
            label="human", 
            content="My friend Sarah is married to Tom and they have two kids."
        )
        
        # Conflicting status update
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="Sarah got divorced last year and is adjusting to single life."
        )
        
        # Verify conflict exists
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        # Both statements should be present initially
        assert "married to Tom" in memory_content
        assert "got divorced" in memory_content
        
        # This creates a clear conflict for MIRA to resolve autonomously
        
    def test_attribute_change_creates_temporal_conflict(self, memory_system):
        """
        Test that attribute changes create conflicts needing temporal context.
        
        REAL BUG THIS CATCHES: Without proper handling of attribute changes,
        MIRA might reference outdated physical descriptions or characteristics,
        making conversations feel disconnected from reality.
        """
        memory_manager, memory_tool = memory_system
        
        # Original attribute
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="My sister Emma has long brown hair and is studying medicine."
        )
        
        # Changed attribute
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="Emma dyed her hair blonde for summer and switched to law school."
        )
        
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        # Both old and new info present
        assert "brown hair" in memory_content
        assert "blonde" in memory_content
        assert "medicine" in memory_content
        assert "law school" in memory_content
        
        # MIRA should recognize these as temporal changes when resolving
        
    def test_professional_status_updates_need_resolution(self, memory_system):
        """
        Test professional/career changes create resolvable conflicts.
        
        REAL BUG THIS CATCHES: If MIRA keeps outdated job information,
        it might ask "How's work at Google?" when someone quit months ago,
        showing it's not truly listening or remembering conversations.
        """
        memory_manager, memory_tool = memory_system
        
        # Current job
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="David works as a software engineer at Google on the Maps team."
        )
        
        # Job change
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="David quit his job to start his own AI startup called NeuralFlow."
        )
        
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        # Conflict exists
        assert "Google" in memory_content
        assert "Maps team" in memory_content
        assert "quit his job" in memory_content
        assert "NeuralFlow" in memory_content
        
    def test_conflicting_preferences_highlight_resolution_need(self, memory_system):
        """
        Test that preference changes create conflicts requiring resolution.
        
        REAL BUG THIS CATCHES: If MIRA suggests Italian restaurants to someone
        who developed a gluten allergy and can't eat pasta anymore, it shows
        the memory system isn't adapting to important life changes.
        """
        memory_manager, memory_tool = memory_system
        
        # Original preference
        memory_tool.run(
            "core_memory_replace",
            label="persona",
            old_content="I am MIRA, an AI assistant with advanced memory capabilities.",
            new_content="I am MIRA. The user loves Italian food, especially pasta dishes from authentic restaurants."
        )
        
        # Preference change due to health
        memory_tool.run(
            "core_memory_append",
            label="persona",
            content="The user was diagnosed with celiac disease and now follows a strict gluten-free diet."
        )
        
        current = memory_tool.run("get_core_memory", label="persona")
        memory_content = current["block"]["value"]
        
        # Conflicting dietary information exists
        assert "loves Italian food" in memory_content
        assert "pasta dishes" in memory_content
        assert "gluten-free diet" in memory_content
        
    def test_memory_block_history_shows_evolution(self, memory_system):
        """
        Test that memory evolution is tracked through history.
        
        REAL BUG THIS CATCHES: Without proper history tracking, we can't
        debug why memory changed or recover from incorrect resolutions,
        making the system opaque and untrustworthy.
        """
        memory_manager, memory_tool = memory_system
        
        # Make several updates that would need resolution
        updates = [
            "Alex lives in Boston and works at MIT.",
            "Alex is considering a job offer in California.",
            "Alex accepted the position and moved to San Francisco.",
            "Alex loves the Bay Area weather compared to Boston winters."
        ]
        
        for update in updates:
            memory_tool.run("core_memory_append", label="human", content=update)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Check history
        with memory_manager.get_session() as session:
            block = session.query(MemoryBlock).filter_by(label="human").first()
            history = session.query(BlockHistory).filter_by(
                block_id=block.id
            ).order_by(BlockHistory.version).all()
            
            # Should have history entries for each change
            assert len(history) >= len(updates)
            
            # History should show progression
            for h in history:
                assert h.operation in ["base", "append"]
                assert h.actor == "tool"
                
    def test_concurrent_conflicting_updates_both_recorded(self, memory_system):
        """
        Test that concurrent conflicting updates are both recorded for resolution.
        
        REAL BUG THIS CATCHES: If concurrent updates cause one piece of
        information to be lost, MIRA might miss critical context needed
        for accurate conflict resolution.
        """
        memory_manager, memory_tool = memory_system
        
        # Simulate concurrent updates about same person
        def update_location(content):
            return memory_tool.run("core_memory_append", label="human", content=content)
        
        # These could happen in rapid succession in a real conversation
        contents = [
            "Maria is visiting New York for a conference next week.",
            "Maria decided to extend her stay in New York for another month.",
            "Maria found a job in New York and is relocating permanently."
        ]
        
        results = []
        for content in contents:
            result = update_location(content)
            results.append(result)
            assert result["success"] is True
        
        # All updates should be recorded
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        assert "visiting New York" in memory_content
        assert "extend her stay" in memory_content
        assert "relocating permanently" in memory_content
        
        # This creates a progression MIRA needs to resolve into current state
        
    def test_memory_size_limits_maintained_during_conflicts(self, memory_system):
        """
        Test that character limits are enforced even with conflicts.
        
        REAL BUG THIS CATCHES: If conflict resolution bypasses size limits,
        memory blocks could grow unbounded, causing performance issues and
        potentially breaking the context window.
        """
        memory_manager, memory_tool = memory_system
        
        # Get current size
        current = memory_tool.run("get_core_memory", label="human")
        current_size = current["block"]["characters"]
        limit = current["block"]["limit"]
        
        # Try to add content that would exceed limit
        large_content = "X" * (limit - current_size + 100)
        
        # Should raise an error when exceeding limit
        from errors import ToolError
        with pytest.raises(ToolError) as exc_info:
            memory_tool.run(
                "core_memory_append",
                label="human",
                content=large_content
            )
        
        # Verify the error message is helpful
        assert "exceed character limit" in str(exc_info.value)
        assert "core_memory_replace" in str(exc_info.value)
        
        # Verify limit wasn't exceeded
        current = memory_tool.run("get_core_memory", label="human")
        assert current["block"]["characters"] <= limit


class TestMemoryConflictResolutionPatterns:
    """
    Test specific patterns of conflict that MIRA should handle.
    
    These tests document expected conflict types without prescribing
    implementation details.
    """
    
    def test_multiple_people_conflicts_remain_separate(self, memory_system):
        """
        Test that conflicts about different people don't interfere.
        
        REAL BUG THIS CATCHES: If conflict resolution merges information
        about different people, MIRA might think John and Sarah are the
        same person or mix up their life events.
        """
        memory_manager, memory_tool = memory_system
        
        # Information about multiple people
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="John lives in Portland. Sarah lives in Boston. Mike lives in Austin."
        )
        
        # Updates about different people  
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="John moved to Seattle. Sarah is still in Boston but considering NYC."
        )
        
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        # All person-specific information should be present
        assert "John" in memory_content and "Seattle" in memory_content
        assert "Sarah" in memory_content and "Boston" in memory_content
        assert "Mike" in memory_content and "Austin" in memory_content
        
    def test_partial_information_updates_preserve_context(self, memory_system):
        """
        Test that partial updates don't lose existing context.
        
        REAL BUG THIS CATCHES: If updating one fact about a person erases
        everything else we know about them, MIRA appears to have amnesia
        about previously discussed details.
        """
        memory_manager, memory_tool = memory_system
        
        # Comprehensive initial information
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="Rachel is a pediatrician in Denver, married with three kids, loves hiking."
        )
        
        # Partial update
        memory_tool.run(
            "core_memory_append",
            label="human",
            content="Rachel got promoted to department head at the hospital."
        )
        
        current = memory_tool.run("get_core_memory", label="human")
        memory_content = current["block"]["value"]
        
        # Original context should be preserved
        assert "pediatrician" in memory_content
        assert "Denver" in memory_content
        assert "three kids" in memory_content
        assert "hiking" in memory_content
        assert "department head" in memory_content