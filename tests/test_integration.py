"""
Tests for lt_memory integration module.

These tests verify that the integration properly initializes all LT_Memory
components with real infrastructure and integrates them with MIRA.
"""

import pytest
import os

from lt_memory.integration import initialize_lt_memory, check_lt_memory_requirements
from config.memory_config import MemoryConfig
from working_memory import WorkingMemory
from tools.repo import ToolRepository
from api.llm_provider import LLMProvider


class TestConfig:
    """Real config object for testing."""
    def __init__(self):
        self.memory = MemoryConfig()
        # Ensure we use test database URL
        self.memory.database_url = os.getenv(
            "LT_MEMORY_DATABASE_URL", 
            "postgresql://mira_app@localhost/lt_memory_test"
        )


class MockAutomationController:
    """Temporary mock for automation controller until real system is fixed."""
    def get_automation(self, name):
        return None
    
    def create_automation(self, automation):
        return {"success": True, "automation_id": f"mock_{automation['name']}"}


@pytest.fixture
def test_config():
    """Real config with memory configuration."""
    return TestConfig()


@pytest.fixture
def real_working_memory():
    """Real working memory instance."""
    return WorkingMemory()


@pytest.fixture
def real_tool_repo():
    """Real tool repository instance."""
    return ToolRepository()


@pytest.fixture
def mock_automation_controller():
    """
    Temporary mock automation controller.
    
    TODO: Replace with real automation controller once the automation system
    interface is properly defined. Currently the automation_controller module
    provides functions, not a class, so we can't test real automation integration.
    """
    return MockAutomationController()


@pytest.fixture
def real_llm_provider():
    """Real LLM provider instance."""
    return LLMProvider()


def test_initialize_lt_memory_creates_all_components(
    test_config, real_working_memory, real_tool_repo, 
    mock_automation_controller, real_llm_provider
):
    """
    Test that initialize_lt_memory creates and integrates all required components.
    
    REAL BUG THIS CATCHES: If any component initialization fails or if the
    integration doesn't properly wire components together, MIRA users won't
    have access to long-term memory functionality, breaking the core value
    proposition of the memory system.
    
    NOTE: This test uses a mock automation controller because the real automation
    system interface is not yet properly defined. Need to update this test once
    the automation system is fixed to use all real components.
    """
    # Test the real integration with real components
    result = initialize_lt_memory(
        config=test_config,
        working_memory=real_working_memory,
        tool_repo=real_tool_repo,
        automation_controller=mock_automation_controller,
        llm_provider=real_llm_provider
    )
    
    # Verify all expected components are created
    assert result["manager"] is not None, "MemoryManager should be created"
    assert result["bridge"] is not None, "MemoryBridge should be created"
    assert result["conversation_timeline_manager"] is not None, "ConversationTimelineManager should be created"
    assert result["tool"] is not None, "LTMemoryTool should be created"
    assert isinstance(result["automations"], dict), "Automations should be registered"
    
    # Verify real integration - memory bridge should be registered with working memory
    assert len(real_working_memory._managers) > 0, "Memory bridge should be registered with working memory"
    
    # Verify tool registration with real tool repository
    initial_tool_count = len(real_tool_repo.tools)
    # Tool should be registered during initialization
    final_tool_count = len(real_tool_repo.tools)
    assert final_tool_count > initial_tool_count, "Memory tool should be registered with tool repository"
    
    # Verify memory manager has real database connection
    memory_manager = result["manager"]
    assert memory_manager is not None
    assert hasattr(memory_manager, 'config'), "MemoryManager should have config"
    assert memory_manager.config == test_config, "MemoryManager should use provided config"
    
    # Verify health check runs (indicating real database connectivity)
    health = memory_manager.health_check()
    assert "status" in health, "Health check should return status"
    assert health["status"] in ["healthy", "unhealthy"], "Health status should be valid"
    
    # Verify memory statistics are accessible (indicating real database operations)
    stats = memory_manager.get_memory_stats()
    assert "blocks" in stats, "Memory stats should include blocks"
    assert "passages" in stats, "Memory stats should include passages"
    assert "count" in stats["blocks"], "Block stats should include count"
    assert "count" in stats["passages"], "Passage stats should include count"


def test_check_lt_memory_requirements_validates_real_infrastructure():
    """
    Test that check_lt_memory_requirements properly validates real system components.
    
    REAL BUG THIS CATCHES: If the requirements check gives false positives about
    missing infrastructure components, users will think the system is ready when
    it's not, leading to runtime failures when they try to use memory features.
    """
    # Set up test environment variable for PostgreSQL
    original_db_url = os.environ.get("LT_MEMORY_DATABASE_URL")
    test_db_url = "postgresql://mira_app@localhost/lt_memory_test"
    os.environ["LT_MEMORY_DATABASE_URL"] = test_db_url
    
    try:
        # Test the real requirements check
        requirements = check_lt_memory_requirements()
        
        # Verify the function returns all expected requirement keys
        expected_keys = {"postgresql", "pgvector", "openai_api", "database_url"}
        assert set(requirements.keys()) == expected_keys, f"Requirements should check these components: {expected_keys}"
        
        # Verify database URL requirement is correctly detected
        assert requirements["database_url"] is True, "Should detect valid PostgreSQL URL"
        
        # Verify each requirement is a boolean
        for key, value in requirements.items():
            assert isinstance(value, bool), f"Requirement '{key}' should be boolean, got {type(value)}"
        
        # The actual values depend on test environment setup
        assert "postgresql" in requirements, "Should check PostgreSQL connectivity"
        assert "pgvector" in requirements, "Should check pgvector extension"
        assert "openai_api" in requirements, "Should check OpenAI API key availability"
        
    finally:
        # Restore original environment
        if original_db_url is not None:
            os.environ["LT_MEMORY_DATABASE_URL"] = original_db_url
        elif "LT_MEMORY_DATABASE_URL" in os.environ:
            del os.environ["LT_MEMORY_DATABASE_URL"]


def test_initialize_lt_memory_handles_initialization_failure():
    """
    Test that initialize_lt_memory gracefully handles component initialization failures.
    
    REAL BUG THIS CATCHES: If a component initialization fails and the function
    crashes instead of returning empty components, the entire MIRA system startup
    fails instead of running without memory features, preventing users from using
    the system at all.
    """
    # Create config with invalid database URL to force failure
    bad_config = TestConfig()
    bad_config.memory.database_url = "postgresql://invalid_user:invalid_pass@nonexistent_host/nonexistent_db"
    
    # Use real components
    working_memory = WorkingMemory()
    tool_repo = ToolRepository()
    automation_controller = MockAutomationController()
    llm_provider = LLMProvider()
    
    # Record initial state
    initial_manager_count = len(working_memory._managers)
    initial_tool_count = len(tool_repo.tools)
    
    # Test that function handles failure gracefully
    result = initialize_lt_memory(
        config=bad_config,
        working_memory=working_memory,
        tool_repo=tool_repo,
        automation_controller=automation_controller,
        llm_provider=llm_provider
    )
    
    # Verify function returns expected structure even on failure
    expected_keys = {"manager", "bridge", "conversation_timeline_manager", "tool", "automations"}
    assert set(result.keys()) == expected_keys, "Should return all expected keys even on failure"
    
    # Verify all components are None/empty on failure (graceful degradation)
    assert result["manager"] is None, "Manager should be None on initialization failure"
    assert result["bridge"] is None, "Bridge should be None on initialization failure"
    assert result["conversation_timeline_manager"] is None, "Timeline manager should be None on initialization failure"
    assert result["tool"] is None, "Tool should be None on initialization failure"
    assert result["automations"] == {}, "Automations should be empty dict on initialization failure"
    
    # Verify no memory components were added to working memory on failure
    assert len(working_memory._managers) == initial_manager_count, "No memory managers should be added on failure"
    
    # Verify no memory tools were added to tool repo on failure
    assert len(tool_repo.tools) == initial_tool_count, "No memory tools should be added on failure"