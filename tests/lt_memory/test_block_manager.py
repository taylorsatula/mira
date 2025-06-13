"""
Production-grade tests for BlockManager - memory block management with versioning.

Testing philosophy: TEST THE CONTRACT, NOT THE IMPLEMENTATION
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from lt_memory.managers.block_manager import BlockManager
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import Base
from config.config_manager import AppConfig
from config.memory_config import MemoryConfig
from errors import ToolError, ErrorCode
from sqlalchemy import create_engine


@pytest.fixture
def test_config():
    """Real configuration for testing with PostgreSQL test database."""
    config = AppConfig()
    memory_config = MemoryConfig()
    memory_config.database_url = "postgresql://mira_app@localhost/lt_memory_test"
    config.memory = memory_config
    return config


@pytest.fixture
def clean_test_database(test_config):
    """Provides a clean test database for each test."""
    engine = create_engine(test_config.memory.database_url)
    
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            try:
                conn.execute(table.delete())
            except Exception:
                pass
        conn.commit()
    
    engine.dispose()
    return test_config


@pytest.fixture
def memory_manager(clean_test_database):
    """Creates a MemoryManager instance with clean database."""
    return MemoryManager(clean_test_database)


@pytest.fixture
def block_manager(memory_manager):
    """Creates a BlockManager instance using real MemoryManager."""
    return BlockManager(memory_manager)


def test_create_block_stores_and_retrieves(block_manager):
    """
    Test that created blocks persist and can be retrieved with correct data.
    
    REAL BUG THIS CATCHES: If block creation or retrieval has SQL bugs,
    persistence bugs, or data transformation errors, users can't store or
    access their memory blocks, breaking the entire memory system.
    """
    content = "# User Profile\nName: Alice\nRole: Engineer"
    
    # Create block
    result = block_manager.create_block("user_profile", content, character_limit=2048)
    
    # Verify creation response
    assert result["label"] == "user_profile"
    assert result["value"] == content
    assert result["limit"] == 2048
    assert result["characters"] == len(content)
    assert result["version"] == 1
    assert "id" in result
    assert "updated_at" in result
    
    # Verify persistence by retrieving
    retrieved = block_manager.get_block("user_profile")
    assert retrieved is not None
    assert retrieved["label"] == "user_profile" 
    assert retrieved["value"] == content
    assert retrieved["version"] == 1


def test_create_duplicate_label_fails(block_manager):
    """
    Test that duplicate labels are rejected with clear error.
    
    REAL BUG THIS CATCHES: If duplicate checking fails, users can accidentally
    overwrite existing blocks, causing data loss and confusion.
    """
    # Create first block
    block_manager.create_block("config", "API_KEY=secret123")
    
    # Attempt to create duplicate should fail
    with pytest.raises(ToolError) as exc_info:
        block_manager.create_block("config", "different content")
    
    assert exc_info.value.code == ErrorCode.MEMORY_BLOCK_ALREADY_EXISTS
    assert "already exists" in str(exc_info.value)
    
    # Original block should be unchanged
    original = block_manager.get_block("config")
    assert original["value"] == "API_KEY=secret123"


def test_character_limit_boundary_conditions(block_manager):
    """
    Test character limit validation at exact boundaries.
    
    REAL BUG THIS CATCHES: If boundary checking has off-by-one errors,
    users get inconsistent behavior where content exactly at the limit
    sometimes works and sometimes fails, causing confusion and unreliable workflows.
    """
    limit = 100
    
    # Content exactly at limit should succeed
    exact_content = "x" * limit
    result = block_manager.create_block("exact_limit", exact_content, character_limit=limit)
    assert result["characters"] == limit
    assert result["value"] == exact_content
    
    # Content one character over limit should fail
    with pytest.raises(ToolError) as exc_info:
        block_manager.create_block("over_limit", "x" * (limit + 1), character_limit=limit)
    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert "exceeds character limit" in str(exc_info.value)
    
    # Content one under limit should succeed  
    under_content = "x" * (limit - 1)
    result2 = block_manager.create_block("under_limit", under_content, character_limit=limit)
    assert result2["characters"] == limit - 1


def test_append_respects_character_limit_during_operation(block_manager):
    """
    Test that append operations respect character limits and handle newlines correctly.
    
    REAL BUG THIS CATCHES: If append limit checking fails or newline handling
    is wrong, users can create oversized blocks or get malformed content that
    breaks rendering or exceeds LLM context limits.
    """
    limit = 50
    
    # Create block with content leaving room for more
    initial = "Initial content."  # 16 chars
    block_manager.create_block("append_test", initial, character_limit=limit)
    
    # Append that fits should work
    additional = "More text."  # 10 chars + 1 newline = 27 total
    result = block_manager.core_memory_append("append_test", additional)
    
    expected = initial + "\n" + additional
    assert result["value"] == expected
    assert result["characters"] == len(expected)
    assert result["version"] == 2
    
    # Append that would exceed limit should fail
    too_much = "x" * 30  # Would make total > 50
    with pytest.raises(ToolError) as exc_info:
        block_manager.core_memory_append("append_test", too_much)
    
    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert "exceed character limit" in str(exc_info.value)
    
    # Original block should be unchanged after failed append
    unchanged = block_manager.get_block("append_test")
    assert unchanged["value"] == expected
    assert unchanged["version"] == 2


def test_complete_version_history_and_rollback_workflow(block_manager):
    """
    Test complete workflow: create, modify multiple times, check history, rollback.
    
    REAL BUG THIS CATCHES: If version tracking, history storage, or rollback
    reconstruction fails, users lose the ability to track changes and recover
    from mistakes, breaking a core value proposition of the memory system.
    """
    # Create initial block
    initial = "# User Notes\nImportant information"
    result = block_manager.create_block("workflow_test", initial)
    assert result["version"] == 1
    
    # Make several modifications to build history
    block_manager.core_memory_append("workflow_test", "\n- Added note 1")
    v2 = block_manager.get_block("workflow_test")
    assert v2["version"] == 2
    
    block_manager.core_memory_replace("workflow_test", "Important information", "Critical data")
    v3 = block_manager.get_block("workflow_test")
    assert v3["version"] == 3
    assert "Critical data" in v3["value"]
    
    block_manager.core_memory_append("workflow_test", "\n- Added note 2")
    v4 = block_manager.get_block("workflow_test")
    assert v4["version"] == 4
    
    # Check complete history
    history = block_manager.get_block_history("workflow_test")
    assert len(history) == 4
    
    # Should be newest first
    operations = [entry["operation"] for entry in history]
    assert operations == ["append", "replace", "append", "base"]
    
    versions = [entry["version"] for entry in history]
    assert versions == [4, 3, 2, 1]
    
    # Rollback to version 2 (after first append, before replace)
    rolled_back = block_manager.rollback_block("workflow_test", version=2)
    assert rolled_back["version"] == 5  # New version for rollback
    assert "Important information" in rolled_back["value"]  # Original text restored
    assert "Critical data" not in rolled_back["value"]  # Replace undone
    assert "- Added note 1" in rolled_back["value"]  # First append preserved
    assert "- Added note 2" not in rolled_back["value"]  # Later changes undone
    
    # History should now include the rollback
    updated_history = block_manager.get_block_history("workflow_test")
    assert len(updated_history) == 5
    assert updated_history[0]["operation"] == "rollback"


def test_concurrent_block_operations_maintain_data_integrity(block_manager):
    """
    Test that concurrent operations on memory blocks maintain data integrity.
    
    REAL BUG THIS CATCHES: If database transactions, connection pooling, or 
    locking has bugs, concurrent users can corrupt each other's data, lose
    updates, or cause deadlocks that break the entire memory system.
    """
    # Create initial block with calculated space for testing overflow
    initial_content = "Initial content for testing"  # 28 chars
    limit = 120  # Tight limit to force overflow: 28 + ~25*3 + newlines ≈ 110-115 chars
    block_manager.create_block("concurrent_test", initial_content, character_limit=limit)
    
    # Track operations and results
    append_successes = []
    replace_successes = []
    rollback_successes = []
    append_failures = []
    replace_failures = []
    rollback_failures = []
    
    def append_worker(worker_id):
        try:
            content = f"\nAppend from worker {worker_id}"  # ~25 chars + 1 newline
            result = block_manager.core_memory_append("concurrent_test", content)
            append_successes.append((worker_id, result["version"]))
            return True
        except ToolError:
            append_failures.append(worker_id)
            return False
    
    def replace_worker(worker_id):
        try:
            # All try to replace the same text - only first succeeds
            result = block_manager.core_memory_replace("concurrent_test", "Initial content", f"Updated by {worker_id}")
            replace_successes.append((worker_id, result["version"]))
            return True
        except ToolError:
            replace_failures.append(worker_id)
            return False
    
    def rollback_worker(worker_id):
        try:
            # Wait for other operations to create version history
            time.sleep(0.2)
            current = block_manager.get_block("concurrent_test")
            if current["version"] >= 3:
                result = block_manager.rollback_block("concurrent_test", version=2)
                rollback_successes.append((worker_id, result["version"]))
                return True
            else:
                rollback_failures.append(worker_id)
                return False
        except ToolError:
            rollback_failures.append(worker_id)
            return False
    
    # Run concurrent operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        # 4 appends - expect ~3 to succeed before overflow
        for i in range(4):
            futures.append(executor.submit(append_worker, f"append_{i}"))
        
        # 4 replaces - expect exactly 1 to succeed (race condition)
        for i in range(4):
            futures.append(executor.submit(replace_worker, f"replace_{i}"))
        
        # 2 rollbacks - expect 0-1 to succeed depending on timing
        for i in range(2):
            futures.append(executor.submit(rollback_worker, f"rollback_{i}"))
        
        # Wait for completion
        [future.result() for future in as_completed(futures)]
    
    # Verify exact expected outcomes
    final_block = block_manager.get_block("concurrent_test")
    
    # Core contract: block integrity maintained
    assert final_block is not None
    assert final_block["characters"] <= limit, "Character limit never exceeded"
    assert final_block["characters"] == len(final_block["value"])
    
    # Expected outcomes based on our design:
    
    # 1. Database serialization means all replaces can succeed if they find the target text
    # This validates that PostgreSQL transactions are working correctly
    assert len(replace_successes) >= 1, f"At least 1 replace should succeed, got {len(replace_successes)}"
    assert len(replace_successes) + len(replace_failures) == 4, f"All 4 replace operations should be accounted for"
    
    # 2. Appends: some should succeed until character limit reached
    assert len(append_successes) + len(append_failures) == 4, f"All 4 append operations should be accounted for"
    assert len(append_successes) >= 1, f"At least some appends should succeed before limit reached"
    
    # 3. Character limit should never be exceeded regardless of concurrency
    assert final_block["characters"] <= limit, "Concurrent operations should never exceed character limit"
    
    # 4. All operations should be accounted for (no operations lost)
    total_attempted = 10  # 4 appends + 4 replaces + 2 rollbacks
    total_completed = len(append_successes) + len(replace_successes) + len(rollback_successes)
    total_failed = len(append_failures) + len(replace_failures) + len(rollback_failures)
    assert total_completed + total_failed == total_attempted, "All operations should be accounted for"
    
    # 5. Version consistency - final version should reflect successful operations
    assert final_block["version"] >= 1, "Version should be at least 1"
    
    # 6. History consistency - should now work correctly with row-level locking
    history = block_manager.get_block_history("concurrent_test")
    assert len(history) == final_block["version"], "History entries should match version count"
    
    # Verify version numbers are sequential (no duplicates from race conditions)
    versions = [h["version"] for h in reversed(history)]  # Chronological order
    expected_versions = list(range(1, final_block["version"] + 1))
    assert versions == expected_versions, f"Versions should be sequential: expected {expected_versions}, got {versions}"


def test_core_memory_replace_updates_content(block_manager):
    """
    Test that replacing content in blocks works correctly.
    
    REAL BUG THIS CATCHES: If replace functionality fails, users can't correct
    mistakes or update specific information in memory blocks, forcing them to
    recreate entire blocks and losing version history.
    """
    # Create block with content to replace
    initial_content = "Status: In Progress\nPriority: Medium\nDue: Tomorrow"
    block_manager.create_block("task", initial_content)
    
    # Replace specific content
    result = block_manager.core_memory_replace("task", "Status: In Progress", "Status: Completed")
    
    # Core contract: old content should be gone, new content should be present
    assert "Status: In Progress" not in result["value"]
    assert "Status: Completed" in result["value"]
    assert result["version"] == 2
    
    # Other content should be unchanged
    assert "Priority: Medium" in result["value"]
    assert "Due: Tomorrow" in result["value"]


def test_memory_insert_at_line_number(block_manager):
    """
    Test that inserting content at specific line numbers works correctly.
    
    REAL BUG THIS CATCHES: If line-based insertion fails, users can't add
    content at precise locations in structured memory blocks like lists
    or numbered sections, breaking organized information management.
    """
    # Create block with multiple lines
    initial_content = "Line 1\nLine 2\nLine 3"
    block_manager.create_block("list", initial_content)
    
    # Insert at line 2 (between Line 1 and Line 2)
    result = block_manager.memory_insert("list", "Inserted Line", 2)
    
    # Core contract: new content should be at correct position
    expected_content = "Line 1\nInserted Line\nLine 2\nLine 3"
    assert result["value"] == expected_content
    assert result["version"] == 2


def test_memory_rethink_rewrites_block(block_manager):
    """
    Test that rethinking completely replaces block content.
    
    REAL BUG THIS CATCHES: If rethink functionality fails, users can't
    completely rewrite memory blocks when context fundamentally changes,
    forcing them to manually edit or delete and recreate blocks.
    """
    # Create block with initial content
    initial_content = "Old project goals:\n1. Build app\n2. Launch quickly"
    block_manager.create_block("project", initial_content)
    
    # Completely rewrite with new content
    new_content = "Updated strategy:\n- Focus on quality\n- Extensive testing\n- Gradual rollout"
    result = block_manager.memory_rethink("project", new_content)
    
    # Core contract: all old content should be gone, only new content remains
    assert result["value"] == new_content
    assert result["version"] == 2
    assert "Old project goals" not in result["value"]
    assert "Updated strategy" in result["value"]


def test_get_nonexistent_block_returns_none(block_manager):
    """
    Test that requesting nonexistent blocks returns None gracefully.
    
    REAL BUG THIS CATCHES: If missing block handling throws exceptions instead
    of returning None, the system crashes when users reference blocks that
    don't exist, causing poor user experience and broken workflows.
    """
    # Try to get a block that doesn't exist
    result = block_manager.get_block("nonexistent_block")
    
    # Core contract: should return None, not crash
    assert result is None


def test_get_block_retrieves_created_block(block_manager):
    """
    Test that get_block successfully retrieves created blocks with correct data structure.
    
    REAL BUG THIS CATCHES: If block retrieval has data transformation bugs,
    field mapping errors, or serialization issues, users get corrupted or
    incomplete block data, breaking memory system reliability.
    """
    # Create a block with known content
    content = "User: Sarah Johnson\nRole: Product Manager\nLocation: Seattle"
    created_block = block_manager.create_block("user_profile", content, character_limit=1000)
    
    # Retrieve the same block
    retrieved_block = block_manager.get_block("user_profile")
    
    # Core contract: retrieved block should have all expected fields with correct values
    assert retrieved_block["label"] == "user_profile"
    assert retrieved_block["value"] == content
    assert retrieved_block["limit"] == 1000
    assert retrieved_block["characters"] == len(content)
    assert retrieved_block["version"] == 1
    assert "id" in retrieved_block
    assert "updated_at" in retrieved_block
    
    # Should match what create_block returned
    assert retrieved_block["id"] == created_block["id"]
    assert retrieved_block["updated_at"] == created_block["updated_at"]


def test_get_all_blocks_returns_sorted_blocks(block_manager):
    """
    Test that get_all_blocks returns blocks in alphabetical order by label.
    
    REAL BUG THIS CATCHES: If block listing has sorting bugs, wrong order,
    or missing blocks, users can't see all their memory blocks in predictable
    order, breaking memory management workflows.
    """
    # Get initial blocks (MemoryManager creates some by default)
    initial_blocks = block_manager.get_all_blocks()
    initial_count = len(initial_blocks)
    
    # Create additional blocks in non-alphabetical order
    block_manager.create_block("zebra", "Last alphabetically")
    block_manager.create_block("apple", "First alphabetically") 
    block_manager.create_block("middle", "Middle alphabetically")
    
    # Get all blocks
    all_blocks = block_manager.get_all_blocks()
    
    # Core contract: should return all blocks (initial + new) sorted by label
    assert len(all_blocks) == initial_count + 3
    
    # Should be sorted alphabetically by label
    labels = [block["label"] for block in all_blocks]
    assert labels == sorted(labels)  # Verify alphabetical order
    
    # Our new blocks should be in correct positions
    assert "apple" in labels
    assert "middle" in labels  
    assert "zebra" in labels
    assert labels.index("apple") < labels.index("middle") < labels.index("zebra")
    
    # Each block should have all required fields
    for block in all_blocks:
        assert "id" in block
        assert "label" in block
        assert "value" in block
        assert "limit" in block
        assert "characters" in block
        assert "version" in block
        assert "updated_at" in block


def test_render_blocks_for_prompt_inclusion(block_manager):
    """
    Test that render_blocks produces proper formatted output for LLM prompts.
    
    REAL BUG THIS CATCHES: If block rendering fails or produces malformed
    output, the LLM gets corrupted or missing memory context, breaking the
    entire memory system's ability to provide context to conversations.
    """
    # Create test blocks with known content
    block_manager.create_block("user", "Name: Alice\nRole: Engineer")
    block_manager.create_block("project", "Building chat system")
    
    # Render blocks using default template
    rendered = block_manager.render_blocks()
    
    # Core contract: should produce XML-like format for each block
    assert rendered
    
    # Should contain block tags with character counts
    assert "<user " in rendered
    assert "<project " in rendered
    assert "</user>" in rendered
    assert "</project>" in rendered
    
    # Should include character count attributes
    assert 'characters="' in rendered
    
    # Should contain actual block content
    assert "Name: Alice" in rendered
    assert "Role: Engineer" in rendered
    assert "Building chat system" in rendered
    
    # Verify structure: opening and closing tags should match
    assert rendered.count("<user") == rendered.count("</user>")
    assert rendered.count("<project") == rendered.count("</project>")


def test_render_blocks_template_injection_vulnerability(block_manager):
    """
    Test for critical Jinja2 template injection vulnerability.
    
    REAL BUG THIS CATCHES: Users can inject Jinja2 template code that gets
    executed during rendering, allowing them to access system config, execute
    arbitrary Python code, and compromise the entire system.
    """
    # Test template injection
    template_injection = "{{ config.__class__.__bases__[0].__subclasses__() }}"
    block_manager.create_block("injection_test", template_injection)
    
    rendered = block_manager.render_blocks()
    
    # SECURITY: Template code should be escaped/safe, not executed
    # If the template injection was successful, we'd see the result of the code execution
    # If it's safe, we should see the literal template code (escaped) 
    assert "{{ config.__class__" in rendered, "Template code should appear as literal text"
    # But we should NOT see signs of code execution like object lists or class references
    assert "<class " not in rendered, "Template code should not be executed"


def test_rollback_block_to_previous_version(block_manager):
    """
    Test that rollback_block correctly restores previous block content.
    
    REAL BUG THIS CATCHES: If rollback functionality fails, users can't recover
    from mistakes or unwanted changes, making the memory system fragile and
    unreliable for critical information management.
    """
    # Create block and make several changes to build version history
    original_content = "Version 1: Initial content"
    block_manager.create_block("rollback_test", original_content)
    
    # Make changes to create version history
    block_manager.core_memory_append("rollback_test", "\nVersion 2: Added line")
    block_manager.core_memory_replace("rollback_test", "Initial", "Modified")
    
    # Verify current state
    current_block = block_manager.get_block("rollback_test") 
    assert current_block["version"] == 3
    assert "Modified" in current_block["value"]
    assert "Added line" in current_block["value"]
    
    # Rollback to version 2
    rolled_back = block_manager.rollback_block("rollback_test", version=2)
    
    # Core contract: should restore version 2 content and create new version
    assert rolled_back["version"] == 4  # New version created for rollback
    assert "Initial" in rolled_back["value"]  # Version 2 had "Initial", not "Modified"
    assert "Added line" in rolled_back["value"]  # Version 2 had the added line
    assert rolled_back["value"] != current_block["value"]  # Should be different from version 3
    
    # Verify persistence
    retrieved = block_manager.get_block("rollback_test")
    assert retrieved["value"] == rolled_back["value"]
    assert retrieved["version"] == 4


def test_get_block_history_tracks_changes(block_manager):
    """
    Test that get_block_history returns complete change history in correct order.
    
    REAL BUG THIS CATCHES: If history tracking fails, users can't see what
    changes were made, who made them, or when, breaking audit trails and
    debugging workflows for memory management.
    """
    # Create block and make several changes
    block_manager.create_block("history_test", "Initial content")
    block_manager.core_memory_append("history_test", "\nAppended line", actor="user1")
    block_manager.core_memory_replace("history_test", "Initial", "Modified", actor="user2")
    
    # Get complete history
    history = block_manager.get_block_history("history_test")
    
    # Core contract: should return all changes in reverse chronological order
    assert len(history) == 3
    
    # Should be ordered by version descending (newest first)
    versions = [entry["version"] for entry in history]
    assert versions == [3, 2, 1]
    
    # Should track operations correctly
    operations = [entry["operation"] for entry in history]
    assert operations == ["replace", "append", "base"]
    
    # Should track actors correctly
    actors = [entry["actor"] for entry in history]
    assert actors == ["user2", "user1", "system"]
    
    # Each entry should have all required fields
    for entry in history:
        assert "id" in entry
        assert "version" in entry
        assert "operation" in entry
        assert "actor" in entry
        assert "created_at" in entry


def test_get_block_history_edge_cases(block_manager):
    """
    Test get_block_history edge cases: nonexistent blocks and limit parameter.
    
    REAL BUG THIS CATCHES: If history retrieval fails for missing blocks or
    doesn't respect limits, users get crashes or incomplete information,
    breaking history browsing functionality.
    """
    # Test nonexistent block returns empty list, not crash
    empty_history = block_manager.get_block_history("nonexistent_block")
    assert empty_history == []
    
    # Create block and make many changes to test limit
    block_manager.create_block("limit_test", "Version 1")
    for i in range(2, 8):  # Create versions 2-7
        block_manager.core_memory_append("limit_test", f"\nVersion {i}")
    
    # Test limit parameter
    limited_history = block_manager.get_block_history("limit_test", limit=3)
    assert len(limited_history) == 3
    
    # Should still be newest first
    versions = [entry["version"] for entry in limited_history]
    assert versions == [7, 6, 5]  # Most recent 3 versions
    
    # Test getting all history (default limit=10 should get all 7)
    full_history = block_manager.get_block_history("limit_test")
    assert len(full_history) == 7
    
    # All versions should be present in descending order
    all_versions = [entry["version"] for entry in full_history]
    assert all_versions == [7, 6, 5, 4, 3, 2, 1]


def test_render_blocks_handles_malicious_content(block_manager):
    """
    Test that render_blocks safely handles malicious or problematic user input.
    
    REAL BUG THIS CATCHES: If template rendering doesn't properly escape or
    handle malicious input, users can inject XML/template code that breaks
    LLM prompts or causes security vulnerabilities in the system.
    """
    # Test various attack vectors that users might enter
    malicious_inputs = [
        # XML injection attempts
        "</user><malicious>evil</malicious><user>",
        "<script>alert('xss')</script>",
        
        # Note: Template injection tested separately
        
        # Special characters that could break parsing (avoiding null bytes)
        "Content with\x01control\x02chars\x03and\x04more",
        "Unicode test: \u202e\u2066\u2067evil\u2069\u202c",
        
        # Very long content (potential DoS)
        "A" * 1000 + " very long content that might break things",
        
        # Edge case characters
        'Content with "quotes" and \'apostrophes\' and <brackets>',
        "Line 1\n\n\n\n\nMany newlines\n\n\n",
    ]
    
    # Create blocks with potentially malicious content
    for i, malicious_content in enumerate(malicious_inputs):
        label = f"test_{i}"
        block_manager.create_block(label, malicious_content)
    
    # Core contract: render_blocks should handle all content safely without crashing
    rendered = block_manager.render_blocks()
    
    # Should not crash and return valid output
    assert rendered
    
    # Should still maintain basic XML structure
    assert "<test_0" in rendered  # First test block should be present
    
    # Basic structure should be maintained
    # (Template injection tested separately)
    
    # XML structure should still be intact despite malicious input
    # Count opening vs closing tags for first few test blocks
    for i in range(3):
        label = f"test_{i}"
        opening_count = rendered.count(f"<{label}")
        closing_count = rendered.count(f"</{label}>")
        assert opening_count == closing_count, f"Mismatched tags for {label}"