"""
Tests for working_memory.py module.

These tests validate the core contracts of WorkingMemory and its trinkets
to catch real production bugs that would break system prompt generation
and memory management.
"""
import pytest
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from working_memory import (
    WorkingMemory, 
    BaseTrinket,
    TimeManager,
    SystemStatusManager,
    ReminderManager
)
from utils.timezone_utils import utc_now


class TestWorkingMemoryCore:
    """Test core WorkingMemory functionality."""
    
    @pytest.fixture
    def memory(self):
        """Fresh working memory instance for each test."""
        return WorkingMemory()
    
    def test_add_and_retrieve_content(self, memory):
        """
        Test that content added to working memory can be retrieved in system prompt.
        
        REAL BUG THIS CATCHES: If add() or get_prompt_content() has bugs in storage
        or retrieval, system prompts will be missing critical context, causing the
        assistant to give incorrect or incomplete responses to users.
        """
        # Add real content that would appear in production
        content1 = "# Active Reminders\nUser has meeting at 3pm today"
        content2 = "# System Status\nAll systems operational"
        
        id1 = memory.add(content1, "reminders")
        id2 = memory.add(content2, "status")
        
        # Verify content is properly stored and retrievable
        prompt_content = memory.get_prompt_content()
        
        assert isinstance(id1, str) and len(id1) > 0
        assert isinstance(id2, str) and len(id2) > 0
        assert id1 != id2  # IDs should be unique
        
        # Verify both pieces of content appear in prompt
        assert content1 in prompt_content
        assert content2 in prompt_content
        assert "\n\n" in prompt_content  # Content should be separated
    
    def test_concurrent_memory_access_maintains_integrity(self, memory):
        """
        Test that concurrent access to working memory doesn't corrupt data.
        
        REAL BUG THIS CATCHES: If working memory has race conditions, concurrent
        requests could corrupt the system prompt content, causing inconsistent
        or broken assistant responses during high traffic.
        """
        results = []
        
        def add_content_worker(worker_id):
            """Worker that adds content and verifies it's retrievable."""
            content = f"Worker {worker_id} data: {utc_now().isoformat()}"
            category = f"worker_{worker_id}"
            
            item_id = memory.add(content, category)
            
            # Verify content is immediately retrievable
            prompt_content = memory.get_prompt_content()
            if content in prompt_content:
                results.append((worker_id, item_id, True))
            else:
                results.append((worker_id, item_id, False))
        
        # Launch concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_content_worker, i) for i in range(20)]
            for future in futures:
                future.result()
        
        # Verify all operations succeeded
        assert len(results) == 20
        assert all(success for _, _, success in results)
        
        # Verify no data corruption - all content should be present
        final_prompt = memory.get_prompt_content()
        for worker_id, _, _ in results:
            assert f"Worker {worker_id} data:" in final_prompt

    def test_remove_content_and_category_operations(self, memory):
        """
        Test that content removal operations work correctly.
        
        REAL BUG THIS CATCHES: If remove() or remove_by_category() has bugs,
        stale content will accumulate in system prompts, causing the assistant
        to reference outdated information or hit token limits.
        """
        # Add content in different categories
        reminder_id = memory.add("Meeting at 3pm", "reminders")
        status_id1 = memory.add("System starting up", "status")
        status_id2 = memory.add("Database connected", "status")
        other_id = memory.add("User preferences loaded", "preferences")
        
        # Verify all content is present
        prompt = memory.get_prompt_content()
        assert "Meeting at 3pm" in prompt
        assert "System starting up" in prompt
        assert "Database connected" in prompt
        assert "User preferences loaded" in prompt
        
        # Remove specific item
        removed = memory.remove(reminder_id)
        assert removed is True
        
        prompt = memory.get_prompt_content()
        assert "Meeting at 3pm" not in prompt
        assert "System starting up" in prompt  # Others should remain
        
        # Remove entire category
        removed_count = memory.remove_by_category("status")
        assert removed_count == 2
        
        prompt = memory.get_prompt_content()
        assert "System starting up" not in prompt
        assert "Database connected" not in prompt
        assert "User preferences loaded" in prompt  # Different category should remain
        
        # Test removing non-existent item
        fake_removed = memory.remove("nonexistent-id")
        assert fake_removed is False

    def test_empty_and_invalid_input_validation(self, memory):
        """
        Test that empty and invalid inputs are handled properly.
        
        REAL BUG THIS CATCHES: If validation fails, empty content could
        corrupt system prompts or cause crashes when generating responses,
        breaking the assistant for users.
        """
        # Test empty content
        with pytest.raises(ValueError) as exc_info:
            memory.add("", "test_category")
        assert "Content and category cannot be empty" in str(exc_info.value)
        
        # Test empty category
        with pytest.raises(ValueError) as exc_info:
            memory.add("Some content", "")
        assert "Content and category cannot be empty" in str(exc_info.value)
        
        # Test None values
        with pytest.raises(ValueError):
            memory.add(None, "test_category")
        
        with pytest.raises(ValueError):
            memory.add("Some content", None)
        
        # Verify empty memory returns empty string
        empty_prompt = memory.get_prompt_content()
        assert empty_prompt == ""
        
        # Test get_items_by_category with non-existent category
        items = memory.get_items_by_category("nonexistent")
        assert items == []

    def test_memory_items_isolation_and_metadata(self, memory):
        """
        Test that memory items are properly isolated and maintain metadata.
        
        REAL BUG THIS CATCHES: If get_items_by_category() returns references 
        instead of copies, external code could modify working memory content,
        corrupting system prompts for all subsequent requests.
        """
        # Add content
        content_id = memory.add("Test content", "test_category")
        
        # Get items by category
        items = memory.get_items_by_category("test_category")
        assert len(items) == 1
        
        item = items[0]
        assert item["content"] == "Test content"
        assert item["category"] == "test_category"
        assert "metadata" in item
        
        # Modify returned item (should not affect original)
        item["content"] = "Modified content"
        item["category"] = "modified_category"
        item["metadata"]["new_key"] = "new_value"
        
        # Verify original is unchanged
        original_prompt = memory.get_prompt_content()
        assert "Test content" in original_prompt
        assert "Modified content" not in original_prompt
        
        # Verify original item data unchanged
        fresh_items = memory.get_items_by_category("test_category")
        assert len(fresh_items) == 1
        assert fresh_items[0]["content"] == "Test content"
        assert fresh_items[0]["category"] == "test_category"


class TestManagerRegistration:
    """Test manager registration and update functionality."""
    
    @pytest.fixture
    def memory(self):
        """Fresh working memory instance for each test."""
        return WorkingMemory()
    
    class MockManager:
        """Mock manager for testing registration functionality."""
        
        def __init__(self, name):
            self.name = name
            self.update_called = False
            self.cleanup_called = False
            
        def update_working_memory(self):
            """Mock update method."""
            self.update_called = True
            
        def cleanup(self):
            """Mock cleanup method."""
            self.cleanup_called = True
    
    class BadManager:
        """Manager without required methods."""
        pass
    
    def test_manager_registration_and_updates(self, memory):
        """
        Test that managers can be registered and their updates are called.
        
        REAL BUG THIS CATCHES: If manager registration or update_all_managers()
        has bugs, components like TimeManager won't update system prompts,
        causing the assistant to reference stale datetime information.
        """
        manager1 = self.MockManager("manager1")
        manager2 = self.MockManager("manager2")
        
        # Register managers
        memory.register_manager(manager1)
        memory.register_manager(manager2)
        
        # Verify managers are not called during registration
        assert manager1.update_called is False
        assert manager2.update_called is False
        
        # Call update_all_managers
        memory.update_all_managers()
        
        # Verify both managers were updated
        assert manager1.update_called is True
        assert manager2.update_called is True
    
    def test_manager_registration_without_update_method(self, memory):
        """
        Test that managers without update_working_memory method are handled gracefully.
        
        REAL BUG THIS CATCHES: If manager validation is broken, registering
        incompatible managers could crash the system when trying to update
        working memory before generating responses.
        """
        bad_manager = self.BadManager()
        
        # Should register without error (only warns)
        memory.register_manager(bad_manager)
        
        # Should not crash when calling update_all_managers
        memory.update_all_managers()  # Should complete without error
    
    def test_cleanup_all_managers(self, memory):
        """
        Test that cleanup properly clears managers and memory items.
        
        REAL BUG THIS CATCHES: If cleanup_all_managers() has bugs, memory
        won't be properly cleared on system shutdown, causing memory leaks
        in long-running processes.
        """
        manager1 = self.MockManager("manager1")
        manager2 = self.MockManager("manager2")
        
        # Register managers and add some content
        memory.register_manager(manager1)
        memory.register_manager(manager2)
        memory.add("Test content", "test_category")
        
        # Verify content exists
        assert memory.get_prompt_content() != ""
        
        # Cleanup
        memory.cleanup_all_managers()
        
        # Verify managers cleanup was called
        assert manager1.cleanup_called is True
        assert manager2.cleanup_called is True
        
        # Verify memory items cleared
        assert memory.get_prompt_content() == ""
        
        # Verify managers list cleared
        assert len(memory._managers) == 0
        
        # Verify update_all_managers works with empty manager list
        memory.update_all_managers()  # Should not crash


class TestBaseTrinket:
    """Test BaseTrinket functionality."""
    
    @pytest.fixture
    def memory(self):
        """Fresh working memory instance for each test."""
        return WorkingMemory()
    
    def test_base_trinket_content_tracking(self, memory):
        """
        Test that BaseTrinket properly tracks and cleans up its content.
        
        REAL BUG THIS CATCHES: If BaseTrinket tracking is broken, trinkets
        won't properly clean up their content on shutdown, causing memory
        leaks and stale content in long-running processes.
        """
        trinket = BaseTrinket(memory)
        
        # Add content through trinket
        content_id1 = trinket._add_content("Content 1", "category1")
        content_id2 = trinket._add_content("Content 2", "category2")
        
        # Verify content is in working memory
        prompt = memory.get_prompt_content()
        assert "Content 1" in prompt
        assert "Content 2" in prompt
        
        # Verify trinket is tracking the content
        assert len(trinket._content_ids) == 2
        assert content_id1 in trinket._content_ids
        assert content_id2 in trinket._content_ids
        
        # Remove one item manually
        success = trinket._remove_content(content_id1)
        assert success is True
        
        # Verify tracking updated
        assert len(trinket._content_ids) == 1
        assert content_id1 not in trinket._content_ids
        assert content_id2 in trinket._content_ids
        
        # Verify content removed from memory
        prompt = memory.get_prompt_content()
        assert "Content 1" not in prompt
        assert "Content 2" in prompt
        
        # Cleanup trinket
        trinket.cleanup()
        
        # Verify all content removed
        assert memory.get_prompt_content() == ""
        assert len(trinket._content_ids) == 0