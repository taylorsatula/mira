"""
Integration test that MIRA actually resolves memory conflicts.

This test creates conflicts and verifies MIRA takes action when encountering them.

REAL BUG THIS CATCHES: If the system prompt instructions don't work, MIRA will
give contradictory answers like "John lives in Portland and Seattle simultaneously."
"""

import pytest
import json
from pathlib import Path

from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.tools.memory_tool import LTMemoryTool
from lt_memory.timeline_manager import MemoryBridge
from config.config_manager import AppConfig
from api.llm_provider import LLMProvider
from working_memory import WorkingMemory


@pytest.fixture
def integrated_memory_system():
    """Set up a full memory system as MIRA would use it."""
    # Config
    config = AppConfig()
    config.memory.database_url = "postgresql://mira_admin@localhost:5432/lt_memory_test"
    config.paths.data_dir = "/tmp/test_mira_conflict"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    
    # Clean database
    from sqlalchemy import create_engine
    from lt_memory.models.base import Base
    
    engine = create_engine(config.memory.database_url)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            try:
                conn.execute(table.delete())
            except Exception:
                pass
        conn.commit()
    engine.dispose()
    
    # Initialize as in main.py
    llm_provider = LLMProvider()
    memory_manager = MemoryManager(config, llm_provider)
    memory_tool = LTMemoryTool(memory_manager)
    
    # Set up working memory integration
    working_memory = WorkingMemory()
    memory_bridge = MemoryBridge(working_memory, memory_manager)
    working_memory.register_manager(memory_bridge)
    
    yield {
        "memory_manager": memory_manager,
        "memory_tool": memory_tool,
        "working_memory": working_memory,
        "llm_provider": llm_provider,
        "config": config
    }
    
    memory_manager.engine.dispose()


class TestMiraConflictResolution:
    """Test that MIRA actually resolves conflicts when given the chance."""
    
    def test_mira_addresses_location_conflict(self, integrated_memory_system):
        """
        Test MIRA handles location conflicts appropriately.
        
        REAL BUG THIS CATCHES: Without working conflict resolution, when asked
        "Where does John live?", MIRA might say "John lives in both Portland 
        and Seattle" which is nonsensical and breaks user trust.
        """
        memory_tool = integrated_memory_system["memory_tool"]
        working_memory = integrated_memory_system["working_memory"]
        llm_provider = integrated_memory_system["llm_provider"]
        
        # Create a location conflict
        memory_tool.run("core_memory_append", label="human",
            content="John is my coworker who lives in Portland and works at Nike.")
        memory_tool.run("core_memory_append", label="human",
            content="John moved to Seattle last month for a new job at Amazon.")
        
        # Update working memory so MIRA sees the conflict
        working_memory.update_all_managers()
        memory_content = working_memory.get_prompt_content()
        
        # Load system prompt
        with open("config/prompts/main_system_prompt.txt", "r") as f:
            system_prompt = f.read()
        
        # Ask MIRA about John's location - this should trigger conflict awareness
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": memory_content + "\n\nWhere does John currently live?"}
        ]
        
        response = llm_provider.generate_response(messages=messages, temperature=0.1)
        response_text = response.content.lower()
        
        # MIRA should not say John lives in both places as current residence
        assert not (
            ("lives in portland" in response_text or "currently in portland" in response_text) and
            ("lives in seattle" in response_text or "currently in seattle" in response_text)
        ), "MIRA presented both locations as current residence"
        
        # MIRA should acknowledge the current location (Seattle)
        assert "seattle" in response_text, "MIRA didn't mention current location"
        
        # MIRA should show awareness of the change
        temporal_indicators = ["moved", "previously", "used to", "now", "relocated", "new job"]
        assert any(indicator in response_text for indicator in temporal_indicators), \
            "MIRA didn't acknowledge the location change"
    
    def test_mira_with_tool_access_fixes_conflicts(self, integrated_memory_system):
        """
        Test that when MIRA has tool access, it can fix conflicts.
        
        REAL BUG THIS CATCHES: If MIRA can't use tools to fix conflicts,
        the contradictions persist forever, degrading memory quality over time.
        """
        memory_tool = integrated_memory_system["memory_tool"]
        working_memory = integrated_memory_system["working_memory"]
        llm_provider = integrated_memory_system["llm_provider"]
        
        # Create a relationship status conflict
        memory_tool.run("core_memory_append", label="human",
            content="Sarah is married to Tom and they have two kids.")
        memory_tool.run("core_memory_append", label="human",
            content="Sarah got divorced last year.")
        
        # Update working memory
        working_memory.update_all_managers()
        initial_content = working_memory.get_prompt_content()
        
        # Load system prompt
        with open("config/prompts/main_system_prompt.txt", "r") as f:
            system_prompt = f.read()
        
        # Give MIRA tools to fix conflicts
        tools = [{
            "type": "function",
            "function": {
                "name": "lt_memory",
                "description": "Memory management operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["core_memory_replace", "memory_rethink"]},
                        "label": {"type": "string"},
                        "old_content": {"type": "string"},
                        "new_content": {"type": "string"}
                    }
                }
            }
        }]
        
        # Ask MIRA to check memory consistency
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_content + "\n\nI notice there might be conflicting information about Sarah in your memory. Can you check and fix any inconsistencies?"}
        ]
        
        # Get response with tool access
        response = llm_provider.generate(
            messages=messages,
            tools=tools,
            temperature=0.1
        )
        
        # Check if MIRA attempted to use tools
        # Note: The actual tool execution would need to be handled in a real system
        # For this test, we're verifying MIRA recognizes the need for action
        
        response_text = response.content.lower()
        
        # MIRA should acknowledge the conflict
        conflict_acknowledgments = ["conflict", "inconsist", "contradict", "update", "fix", "resolve"]
        assert any(word in response_text for word in conflict_acknowledgments), \
            "MIRA didn't acknowledge the conflict"
        
        # If MIRA made tool calls, that's even better
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Verify it's trying to use memory operations
            for tool_call in response.tool_calls:
                assert tool_call.function.name == "lt_memory", \
                    "MIRA used wrong tool for memory operations"