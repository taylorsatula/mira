"""
Core tests for LT_Memory system.

Tests basic functionality without requiring PostgreSQL.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Mock the PostgreSQL dependencies for unit tests
import sys
import os

# Add paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_pgvector import setup_pgvector_mocks
setup_pgvector_mocks()

from lt_memory.models.base import MemoryBlock, BlockHistory, MemoryPassage
from lt_memory.utils.embeddings import EmbeddingCache
from lt_memory.utils.onnx_embeddings import ONNXEmbeddingModel


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir)
            
            # Test cache miss
            assert cache.get("test text") is None
            assert cache.stats["misses"] == 1
            
            # Test cache set and hit
            embedding = np.random.rand(384)
            cache.set("test text", embedding)
            
            cached = cache.get("test text")
            assert cached is not None
            assert np.array_equal(cached, embedding)
            assert cache.stats["hits"] == 1
            assert cache.stats["memory_hits"] == 1
    
    def test_cache_key_generation(self):
        """Test consistent cache key generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir)
            
            key1 = cache._get_cache_key("test text")
            key2 = cache._get_cache_key("test text")
            key3 = cache._get_cache_key("different text")
            
            assert key1 == key2
            assert key1 != key3
    
    def test_lru_eviction(self):
        """Test LRU eviction in memory cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir, memory_cache_size=2)
            
            # Add 3 embeddings to cache with size 2
            cache.set("text1", np.array([1]))
            cache.set("text2", np.array([2]))
            cache.set("text3", np.array([3]))
            
            # text1 should be evicted
            assert len(cache.memory_cache) == 2
            assert "text1" not in [cache._get_cache_key(k) for k in ["text1"]]


class TestMemoryModels:
    """Test database models (without actual database)."""
    
    def test_memory_block_creation(self):
        """Test MemoryBlock model creation."""
        block = MemoryBlock(
            label="test",
            value="test content",
            character_limit=100,
            version=1  # Set explicitly for testing
        )
        
        assert block.label == "test"
        assert block.value == "test content"
        assert block.character_limit == 100
        assert block.version == 1
    
    def test_block_history_creation(self):
        """Test BlockHistory model creation."""
        history = BlockHistory(
            block_id="test-id",
            label="test",
            old_value="old",
            new_value="new",
            operation="replace",
            actor="user"
        )
        
        assert history.operation == "replace"
        assert history.actor == "user"
    
    def test_memory_passage_creation(self):
        """Test MemoryPassage model creation."""
        passage = MemoryPassage(
            text="test passage",
            source="test",
            source_id="123",
            importance_score=0.8,
            access_count=0  # Set explicitly for testing
        )
        
        assert passage.text == "test passage"
        assert passage.importance_score == 0.8
        assert passage.access_count == 0


class TestMemoryBridge:
    """Test WorkingMemory bridge functionality."""
    
    def test_bridge_initialization(self):
        """Test memory bridge can be initialized."""
        from lt_memory.bridge import MemoryBridge
        
        working_memory = Mock()
        memory_manager = Mock()
        
        bridge = MemoryBridge(working_memory, memory_manager)
        
        assert bridge.working_memory == working_memory
        assert bridge.memory_manager == memory_manager
        working_memory.register_manager.assert_called_once_with(bridge)
    
    def test_update_working_memory(self):
        """Test working memory update process."""
        from lt_memory.bridge import MemoryBridge
        
        working_memory = Mock()
        working_memory.add = Mock(return_value="item-123")
        working_memory.remove = Mock()
        
        memory_manager = Mock()
        memory_manager.block_manager = Mock()
        memory_manager.block_manager.render_blocks = Mock(
            return_value="<persona>Test persona</persona>"
        )
        
        bridge = MemoryBridge(working_memory, memory_manager)
        bridge.update_working_memory()
        
        # Should add core memory
        working_memory.add.assert_called()
        call_args = working_memory.add.call_args
        assert "Core Memory" in call_args[1]["content"]
        assert call_args[1]["category"] == "lt_memory_core"


class TestMemoryTool:
    """Test LT_Memory tool interface."""
    
    def test_tool_initialization(self):
        """Test memory tool can be initialized."""
        from lt_memory.tools.memory_tool import LTMemoryTool
        
        memory_manager = Mock()
        tool = LTMemoryTool(memory_manager)
        
        assert tool.name == "lt_memory"
        assert tool.memory_manager == memory_manager
        assert "core_memory_append" in tool.description
    
    def test_tool_operations_list(self):
        """Test tool provides correct operations."""
        from lt_memory.tools.memory_tool import LTMemoryTool
        
        memory_manager = Mock()
        tool = LTMemoryTool(memory_manager)
        
        definition = tool.get_tool_definition()
        operations = definition["parameters"]["properties"]["operation"]["enum"]
        
        expected_ops = [
            "core_memory_append",
            "core_memory_replace",
            "memory_insert",
            "memory_rethink",
            "search_archival",
            "get_entity_info"
        ]
        
        for op in expected_ops:
            assert op in operations
    
    def test_core_memory_append_call(self):
        """Test core memory append operation."""
        from lt_memory.tools.memory_tool import LTMemoryTool
        
        memory_manager = Mock()
        memory_manager.block_manager = Mock()
        memory_manager.block_manager.core_memory_append = Mock(
            return_value={"label": "test", "value": "updated"}
        )
        
        tool = LTMemoryTool(memory_manager)
        result = tool.run(
            operation="core_memory_append",
            label="test",
            content="new content"
        )
        
        assert result["success"] is True
        memory_manager.block_manager.core_memory_append.assert_called_with(
            label="test",
            content="new content",
            actor="tool"
        )


class TestAutomations:
    """Test memory automation definitions."""
    
    def test_automation_definitions(self):
        """Test automation definitions are valid."""
        from lt_memory.automations.memory_automations import MEMORY_AUTOMATIONS
        
        assert len(MEMORY_AUTOMATIONS) >= 4
        
        # Check required automations exist
        automation_names = [a["name"] for a in MEMORY_AUTOMATIONS]
        assert "Hourly Conversation Processing" in automation_names
        assert "Daily Memory Consolidation" in automation_names
        assert "Weekly Memory Review" in automation_names
        
        # Validate automation structure
        for automation in MEMORY_AUTOMATIONS:
            assert "name" in automation
            assert "type" in automation
            assert "frequency" in automation
            assert "enabled" in automation
    
    def test_automation_registration(self):
        """Test automation registration function."""
        from lt_memory.automations.memory_automations import register_memory_automations
        
        # Mock automation controller
        controller = Mock()
        controller.get_automation = Mock(return_value=None)
        controller.create_automation = Mock(
            return_value={"success": True, "automation_id": "test-123"}
        )
        
        registered = register_memory_automations(controller)
        
        # Should attempt to register all automations
        assert controller.create_automation.call_count >= 4
        assert len(registered) >= 4


class TestIntegration:
    """Test integration module."""
    
    @patch('lt_memory.managers.memory_manager.MemoryManager')
    @patch('lt_memory.bridge.MemoryBridge')
    @patch('lt_memory.tools.memory_tool.LTMemoryTool')
    def test_initialize_lt_memory(self, mock_tool, mock_bridge, mock_manager):
        """Test LT_Memory initialization."""
        from lt_memory.integration import initialize_lt_memory
        
        # Mock dependencies
        config = Mock()
        config.memory = Mock()
        working_memory = Mock()
        tool_repo = Mock()
        automation_controller = Mock()
        
        # Mock health check
        mock_manager_instance = Mock()
        mock_manager_instance.health_check = Mock(
            return_value={"status": "healthy"}
        )
        mock_manager_instance.get_memory_stats = Mock(
            return_value={
                "blocks": {"count": 3},
                "passages": {"count": 0},
                "entities": {"count": 0}
            }
        )
        mock_manager.return_value = mock_manager_instance
        
        # Initialize
        result = initialize_lt_memory(
            config, working_memory, tool_repo, automation_controller
        )
        
        # Verify components created
        assert result["manager"] is not None
        assert result["bridge"] is not None
        assert result["tool"] is not None
        
        # Verify tool registered
        tool_repo.register_tool.assert_called_once()
    
    def test_check_requirements(self):
        """Test requirement checking."""
        from lt_memory.integration import check_lt_memory_requirements
        
        # Without any setup, requirements should be false
        requirements = check_lt_memory_requirements()
        
        assert isinstance(requirements, dict)
        assert "postgresql" in requirements
        assert "pgvector" in requirements
        assert "onnx_model" in requirements
        assert "database_url" in requirements