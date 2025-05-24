"""
Integration tests for LT_Memory system.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lt_memory.integration import check_lt_memory_requirements, initialize_lt_memory
from working_memory import WorkingMemory
from tools.repo import ToolRepository
from config import config


class TestLTMemoryIntegration:
    """Test LT_Memory integration with MIRA."""
    
    def test_requirements_check(self):
        """Test that requirement checking works."""
        requirements = check_lt_memory_requirements()
        assert isinstance(requirements, dict)
        assert "postgresql" in requirements
        assert "pgvector" in requirements
        assert "onnx_model" in requirements
        assert "database_url" in requirements
    
    @pytest.mark.skipif(
        not os.getenv("LT_MEMORY_DATABASE_URL"),
        reason="LT_MEMORY_DATABASE_URL not set"
    )
    def test_initialization_with_requirements(self):
        """Test initialization when requirements are met."""
        # Check if all requirements are met
        requirements = check_lt_memory_requirements()
        if not all(requirements.values()):
            pytest.skip("LT_Memory requirements not met")
        
        # Create test components
        working_memory = WorkingMemory()
        tool_repo = ToolRepository()
        
        # Mock automation controller
        class MockAutomationController:
            def create_automation(self, automation):
                return {"success": True, "automation_id": "test-id"}
            
            def get_automation(self, name):
                return None
        
        automation_controller = MockAutomationController()
        
        # Initialize LT_Memory
        components = initialize_lt_memory(
            config,
            working_memory,
            tool_repo,
            automation_controller
        )
        
        # Verify components were created
        assert components["manager"] is not None
        assert components["bridge"] is not None
        assert components["tool"] is not None
        assert isinstance(components["automations"], dict)
        
        # Test memory tool
        memory_tool = components["tool"]
        assert memory_tool.name == "lt_memory"
        
        # Test a basic memory operation
        result = memory_tool.run(
            operation="get_core_memory"
        )
        assert result["success"] is True
        assert "blocks" in result
    
    def test_memory_tool_operations(self):
        """Test memory tool operations."""
        # Skip if requirements not met
        requirements = check_lt_memory_requirements()
        if not all(requirements.values()):
            pytest.skip("LT_Memory requirements not met")
        
        # Initialize components
        working_memory = WorkingMemory()
        tool_repo = ToolRepository()
        
        class MockAutomationController:
            def create_automation(self, automation):
                return {"success": True}
            def get_automation(self, name):
                return None
        
        components = initialize_lt_memory(
            config,
            working_memory,
            tool_repo,
            MockAutomationController()
        )
        
        memory_tool = components["tool"]
        
        # Test core memory operations
        result = memory_tool.run(
            operation="core_memory_append",
            label="persona",
            content="Test memory content"
        )
        assert result["success"] is True
        
        # Test search (should return empty results)
        result = memory_tool.run(
            operation="search_archival",
            query="test query",
            limit=5
        )
        assert result["success"] is True
        assert "results" in result
    
    def test_working_memory_integration(self):
        """Test working memory bridge integration."""
        requirements = check_lt_memory_requirements()
        if not all(requirements.values()):
            pytest.skip("LT_Memory requirements not met")
        
        working_memory = WorkingMemory()
        tool_repo = ToolRepository()
        
        class MockAutomationController:
            def create_automation(self, automation):
                return {"success": True}
            def get_automation(self, name):
                return None
        
        components = initialize_lt_memory(
            config,
            working_memory,
            tool_repo,
            MockAutomationController()
        )
        
        # Test that bridge was registered with working memory
        assert components["bridge"] in working_memory._managers
        
        # Test updating working memory
        working_memory.update_all_managers()
        
        # Check that memory content was added
        items = working_memory.get_items_by_category("lt_memory_core")
        assert len(items) > 0


if __name__ == "__main__":
    # Run basic integration test
    print("Testing LT_Memory integration...")
    
    # Check requirements
    requirements = check_lt_memory_requirements()
    print(f"Requirements status: {requirements}")
    
    if all(requirements.values()):
        print("✓ All requirements met")
        
        # Test basic initialization
        try:
            working_memory = WorkingMemory()
            tool_repo = ToolRepository()
            
            class MockAutomationController:
                def create_automation(self, automation):
                    return {"success": True}
                def get_automation(self, name):
                    return None
            
            components = initialize_lt_memory(
                config,
                working_memory,
                tool_repo,
                MockAutomationController()
            )
            
            print("✓ LT_Memory initialization successful")
            print(f"✓ Memory manager: {components['manager'].__class__.__name__}")
            print(f"✓ Memory bridge: {components['bridge'].__class__.__name__}")
            print(f"✓ Memory tool: {components['tool'].__class__.__name__}")
            print(f"✓ Registered automations: {len(components['automations'])}")
            
        except Exception as e:
            print(f"✗ LT_Memory initialization failed: {e}")
            sys.exit(1)
    else:
        missing = [k for k, v in requirements.values() if not v]
        print(f"✗ Missing requirements: {missing}")
        sys.exit(1)
    
    print("\n✅ Integration test completed successfully!")