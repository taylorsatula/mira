"""
Tests for lt_memory.timeline_manager module.

Tests the timeline manager and memory bridge components that connect LT_Memory with working memory.

RENAMED CLASSES AND FILES (2025-06-10):
- File: bridge.py → timeline_manager.py 
- File: test_bridge.py → test_timeline_manager.py
- ConversationArchiveBridge → ConversationTimelineManager
- TestConversationArchiveBridge → TestConversationTimelineManager
- conversation_bridge fixture → conversation_timeline_manager fixture

All test methods and imports updated to use new naming convention.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta
from unittest.mock import Mock

from lt_memory.timeline_manager import MemoryBridge, ConversationTimelineManager
from working_memory import WorkingMemory
from config.config_manager import AppConfig
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import ArchivedConversation
from utils.timezone_utils import utc_now, ensure_utc


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = AppConfig()
    # Use test database
    config.memory.database_url = "postgresql://mira_app@localhost/lt_memory_test"
    config.memory.db_pool_size = 5
    config.memory.embedding_dim = 1024
    
    # Ensure paths exist
    config.paths.data_dir = "/tmp/test_bridge_cache"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    
    return config


@pytest.fixture
def clean_test_database(test_config):
    """Provides a clean test database for each test."""
    yield test_config
    
    # Clean up data after test
    from sqlalchemy import create_engine
    from lt_memory.models.base import Base
    
    engine = create_engine(test_config.memory.database_url)
    
    # Delete data in reverse dependency order
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            try:
                conn.execute(table.delete())
            except Exception:
                # Table might not exist yet - that's fine
                pass
        conn.commit()
    
    engine.dispose()


@pytest.fixture
def memory_manager(clean_test_database):
    """Create real memory manager with clean test database."""
    return MemoryManager(clean_test_database)


@pytest.fixture
def working_memory():
    """Create real working memory instance."""
    return WorkingMemory()


@pytest.fixture
def memory_bridge(working_memory, memory_manager):
    """Create memory bridge with real components."""
    return MemoryBridge(working_memory, memory_manager)


@pytest.fixture
def conversation_timeline_manager(memory_manager):
    """Create conversation timeline manager with real components."""
    return ConversationTimelineManager(memory_manager)


class TestMemoryBridge:
    """Test MemoryBridge core memory integration."""

    def test_update_working_memory_with_core_blocks(self, memory_bridge, memory_manager):
        """
        Test that core memory blocks are properly formatted and added to working memory.
        
        REAL BUG THIS CATCHES: If update_working_memory() fails to retrieve core blocks
        or format them correctly, the system prompt won't include user's core memory,
        breaking personalization and context continuity.
        """
        # Add some real core memory blocks (use unique test names)
        memory_manager.block_manager.create_block(
            label="test_user_preferences",
            content="Prefers concise responses, dislikes verbose explanations"
        )
        memory_manager.block_manager.create_block(
            label="test_current_projects", 
            content="Working on Python testing framework, learning PostgreSQL"
        )
        
        # Update working memory
        memory_bridge.update_working_memory()
        
        # Verify core memory was added to working memory
        core_memory_items = memory_bridge.working_memory.get_items_by_category("core_memory")
        
        assert len(core_memory_items) == 1
        content = core_memory_items[0]["content"]
        
        # Verify formatted content structure
        assert "# Core Memory" in content
        assert "## Test_User_Preferences" in content or "## Test_user_preferences" in content
        assert "Prefers concise responses" in content
        assert "## Test_Current_Projects" in content or "## Test_current_projects" in content
        assert "Python testing framework" in content


class TestConversationTimelineManager:
    """Test ConversationTimelineManager conversation retrieval and injection."""

    def _create_test_conversation(self, memory_manager, target_date, summary="Test conversation summary"):
        """Helper to create a test archived conversation."""
        test_messages = [
            {"role": "user", "content": "Hello, how are you?", "timestamp": utc_now().isoformat()},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?", "timestamp": utc_now().isoformat()},
            {"role": "user", "content": "Can you explain quantum computing?", "timestamp": utc_now().isoformat()},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.", "timestamp": utc_now().isoformat()}
        ]
        
        with memory_manager.get_session() as session:
            archived_conv = ArchivedConversation(
                conversation_date=ensure_utc(datetime.combine(target_date, datetime.min.time())),
                message_count=len(test_messages),
                messages=test_messages,
                summary=summary,
                conversation_metadata={"created_by": "test", "test_data": True}
            )
            session.add(archived_conv)
            session.commit()
            return str(archived_conv.id)

    def test_inject_conversation_single_date(self, conversation_timeline_manager, memory_manager):
        """
        Test that conversation injection retrieves and formats archived conversation correctly.
        
        REAL BUG THIS CATCHES: If inject_conversation() fails to retrieve the correct conversation
        or format it properly, users won't get the historical context they requested, breaking
        conversation continuity and reference functionality.
        """
        # Create a test conversation for yesterday
        target_date = (utc_now() - timedelta(days=1)).date()
        conversation_id = self._create_test_conversation(
            memory_manager, 
            target_date, 
            "Discussion about quantum computing fundamentals"
        )
        
        # Inject the conversation
        result = conversation_timeline_manager.inject_conversation(target_date, include_full_messages=True)
        
        # Verify injection was successful
        assert result["success"] is True
        assert result["date"] == target_date.isoformat()
        assert result["message_count"] == 4
        assert result["include_full_messages"] is True
        
        # Verify content formatting
        content = result["content"]
        assert f"## Referenced Conversation: {target_date}" in content
        assert "**Messages:** 4" in content
        assert "Discussion about quantum computing fundamentals" in content
        assert "**Full Conversation:**" in content
        assert "Hello, how are you?" in content
        assert "quantum computing" in content
        
        # Verify injection tracking
        assert target_date.isoformat() in conversation_timeline_manager.injected_dates

    def test_inject_date_range_multiple_conversations(self, conversation_timeline_manager, memory_manager):
        """
        Test that date range injection retrieves and formats multiple conversations correctly.
        
        REAL BUG THIS CATCHES: If inject_date_range() fails to retrieve all conversations
        in a range or aggregate them incorrectly, users get incomplete historical context,
        breaking their ability to review conversation patterns over time.
        """
        # Create multiple test conversations over 3 different days
        base_date = (utc_now() - timedelta(days=5)).date()
        conversation_dates = []
        
        for i in range(3):
            conv_date = base_date + timedelta(days=i)
            conversation_dates.append(conv_date)
            # Create conversation with messages timestamped for that specific day
            day_datetime = ensure_utc(datetime.combine(conv_date, datetime.min.time()))
            test_messages = [
                {"role": "user", "content": f"Question from day {i+1}", "timestamp": day_datetime.isoformat()},
                {"role": "assistant", "content": f"Answer from day {i+1}", "timestamp": day_datetime.isoformat()}
            ]
            
            with memory_manager.get_session() as session:
                archived_conv = ArchivedConversation(
                    conversation_date=day_datetime,
                    message_count=len(test_messages),
                    messages=test_messages,
                    summary=f"Day {i+1}: Discussion about programming topic {i+1}",
                    conversation_metadata={"created_by": "test", "day": i+1}
                )
                session.add(archived_conv)
                session.commit()
        
        # Inject the date range
        start_date = conversation_dates[0]
        end_date = conversation_dates[-1]
        result = conversation_timeline_manager.inject_date_range(start_date, end_date, summary_only=False)
        
        # Verify injection was successful
        assert result["success"] is True
        assert result["injected_count"] == 3
        assert result["summary_only"] is False
        
        # Verify content formatting
        content = result["content"]
        assert "# Referenced Archived Conversations" in content
        assert "**Total conversations:** 3" in content
        assert f"**Date range:** {start_date} to {end_date}" in content
        
        # Verify all conversations are included
        for i, conv_date in enumerate(conversation_dates):
            assert f"## {conv_date}" in content
            assert f"Day {i+1}: Discussion about programming topic {i+1}" in content
            assert "**Messages:**" in content  # Full messages included since summary_only=False
            assert f"Question from day {i+1}" in content
            assert f"Answer from day {i+1}" in content
        
        # Verify injection tracking - all dates should be tracked
        for conv_date in conversation_dates:
            assert conv_date.isoformat() in conversation_timeline_manager.injected_dates
        
        assert len(conversation_timeline_manager.injected_dates) == 3
        
        # Test subset retrieval - clear injections and test narrower range
        conversation_timeline_manager.clear_all_injections()
        
        # Request only middle 2 days (subset)
        subset_start = conversation_dates[0] + timedelta(days=1)  # Skip first day
        subset_end = conversation_dates[-1]  # Include last day
        subset_result = conversation_timeline_manager.inject_date_range(subset_start, subset_end, summary_only=True)
        
        # Verify subset injection
        assert subset_result["success"] is True
        assert subset_result["injected_count"] == 2  # Only 2 conversations
        assert subset_result["summary_only"] is True
        
        # Verify subset content formatting
        subset_content = subset_result["content"]
        assert "**Total conversations:** 2" in subset_content
        assert f"**Date range:** {subset_start} to {subset_end}" in subset_content
        
        # Verify only subset conversations are included
        assert f"Day 1: Discussion about programming topic 1" not in subset_content  # First day excluded
        assert f"Day 2: Discussion about programming topic 2" in subset_content     # Second day included
        assert f"Day 3: Discussion about programming topic 3" in subset_content     # Third day included
        
        # Verify summary-only mode (no full messages)
        assert "**Messages:**" not in subset_content
        assert "*Full messages available on request.*" in subset_content
        
        # Verify subset injection tracking
        assert len(conversation_timeline_manager.injected_dates) == 2
        assert conversation_dates[0].isoformat() not in conversation_timeline_manager.injected_dates  # First day not tracked
        assert conversation_dates[1].isoformat() in conversation_timeline_manager.injected_dates      # Second day tracked
        assert conversation_dates[2].isoformat() in conversation_timeline_manager.injected_dates      # Third day tracked

    def test_error_handling_missing_conversations(self, conversation_timeline_manager):
        """
        Test that bridge handles missing conversations gracefully without crashes.
        
        REAL BUG THIS CATCHES: If inject_conversation() or inject_date_range() crash when
        conversations don't exist, users get unhelpful error messages instead of clear
        feedback about what went wrong, breaking the user experience.
        """
        # Test single conversation that doesn't exist
        nonexistent_date = (utc_now() - timedelta(days=365)).date()  # Very old date
        result = conversation_timeline_manager.inject_conversation(nonexistent_date)
        
        # Should fail gracefully with helpful message
        assert result["success"] is False
        assert "No archived conversation found" in result["message"]
        assert nonexistent_date.isoformat() in result["message"]
        assert result["content"] is None
        
        # Should not add to injection tracking
        assert nonexistent_date.isoformat() not in conversation_timeline_manager.injected_dates
        
        # Test date range with no conversations
        start_date = (utc_now() - timedelta(days=400)).date()
        end_date = (utc_now() - timedelta(days=390)).date()
        range_result = conversation_timeline_manager.inject_date_range(start_date, end_date)
        
        # Should fail gracefully with helpful message
        assert range_result["success"] is False
        assert "No conversations found in range" in range_result["message"]
        assert str(start_date) in range_result["message"]
        assert str(end_date) in range_result["message"]
        assert range_result["content"] is None
        
        # Should not affect injection tracking
        assert len(conversation_timeline_manager.injected_dates) == 0

    def test_remove_and_clear_injection_management(self, conversation_timeline_manager, memory_manager):
        """
        Test injection management operations work correctly.
        
        REAL BUG THIS CATCHES: If remove_conversation() or clear_all_injections() have
        logic errors, users can't manage their injected conversations properly, leading
        to confusing state where they think conversations are injected but they're not.
        """
        # Create and inject a test conversation
        target_date = (utc_now() - timedelta(days=2)).date()
        self._create_test_conversation(memory_manager, target_date, "Test conversation for removal")
        
        inject_result = conversation_timeline_manager.inject_conversation(target_date)
        assert inject_result["success"] is True
        assert target_date.isoformat() in conversation_timeline_manager.injected_dates
        
        # Test removing injected conversation
        remove_result = conversation_timeline_manager.remove_conversation(target_date)
        assert remove_result["success"] is True
        assert target_date.isoformat() in remove_result["message"]
        assert remove_result["remaining_count"] == 0
        
        # Should be removed from tracking
        assert target_date.isoformat() not in conversation_timeline_manager.injected_dates
        
        # Test removing conversation that isn't injected
        remove_again_result = conversation_timeline_manager.remove_conversation(target_date)
        assert remove_again_result["success"] is False
        assert "not currently injected" in remove_again_result["message"]
        
        # Inject multiple conversations and test clear all
        dates = []
        for i in range(3):
            conv_date = target_date + timedelta(days=i+1)
            dates.append(conv_date)
            self._create_test_conversation(memory_manager, conv_date, f"Conversation {i+1}")
            conversation_timeline_manager.inject_conversation(conv_date)
        
        assert len(conversation_timeline_manager.injected_dates) == 3
        
        # Clear all injections
        clear_result = conversation_timeline_manager.clear_all_injections()
        assert clear_result["success"] is True
        assert clear_result["cleared_count"] == 3
        assert len(conversation_timeline_manager.injected_dates) == 0

    def test_suggest_relevant_dates_query_matching(self, conversation_timeline_manager, memory_manager):
        """
        Test that suggest_relevant_dates correctly filters by query and returns recent dates.
        
        REAL BUG THIS CATCHES: If suggest_relevant_dates() has broken query matching or
        date sorting, users get irrelevant conversation suggestions, making it hard to
        find the historical context they're looking for.
        """
        # Create conversations with different topics over recent days
        base_date = (utc_now() - timedelta(days=10)).date()
        
        # Conversations with different summaries
        conversations_data = [
            (base_date + timedelta(days=1), "Discussion about Python programming and debugging techniques"),
            (base_date + timedelta(days=3), "Planning weekend activities and restaurant recommendations"), 
            (base_date + timedelta(days=5), "Python code review and optimization strategies"),
            (base_date + timedelta(days=7), "Meeting notes about project deadlines and team coordination"),
            (base_date + timedelta(days=9), "Python testing frameworks and best practices discussion")
        ]
        
        for conv_date, summary in conversations_data:
            self._create_test_conversation(memory_manager, conv_date, summary)
        
        # Test query matching for "Python" - should return 3 conversations
        python_dates = conversation_timeline_manager.suggest_relevant_dates("Python")
        assert len(python_dates) == 3
        
        # Should be sorted by date (most recent first)
        expected_python_dates = [
            (base_date + timedelta(days=9)).isoformat(),  # Most recent Python discussion
            (base_date + timedelta(days=5)).isoformat(),  # Middle Python discussion  
            (base_date + timedelta(days=1)).isoformat()   # Oldest Python discussion
        ]
        assert python_dates == expected_python_dates
        
        # Test query matching for "project" - should return 1 conversation
        project_dates = conversation_timeline_manager.suggest_relevant_dates("project")
        assert len(project_dates) == 1
        assert project_dates[0] == (base_date + timedelta(days=7)).isoformat()
        
        # Test case-insensitive matching
        python_lower_dates = conversation_timeline_manager.suggest_relevant_dates("python")
        assert python_lower_dates == python_dates  # Should match same conversations
        
        # Test no query (should return all conversations)
        all_dates = conversation_timeline_manager.suggest_relevant_dates()
        assert len(all_dates) == 5
        
        # Should be sorted most recent first
        expected_all_dates = [
            (base_date + timedelta(days=9)).isoformat(),
            (base_date + timedelta(days=7)).isoformat(),
            (base_date + timedelta(days=5)).isoformat(),
            (base_date + timedelta(days=3)).isoformat(),
            (base_date + timedelta(days=1)).isoformat()
        ]
        assert all_dates == expected_all_dates
        
        # Test query with no matches
        no_match_dates = conversation_timeline_manager.suggest_relevant_dates("quantum physics")
        assert len(no_match_dates) == 0
        
        # Test days_back parameter (only recent conversations)
        recent_dates = conversation_timeline_manager.suggest_relevant_dates(days_back=5)
        assert len(recent_dates) == 3  # Conversations within 5 days (1, 3, 5 days ago)
        assert recent_dates == [
            (base_date + timedelta(days=9)).isoformat(),  # 1 day ago
            (base_date + timedelta(days=7)).isoformat(),  # 3 days ago  
            (base_date + timedelta(days=5)).isoformat()   # 5 days ago
        ]