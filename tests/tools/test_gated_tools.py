"""
Tests for gated tools infrastructure in ToolRepository.

Gated tools self-determine their availability via is_available() method,
enabling automatic tool appearance/disappearance based on state.

Following MIRA's real testing philosophy:
- No mocks, test actual behavior
- Test contracts, not implementation
- Verify exact return structures
"""
import pytest

from tools.repo import Tool, ToolRepository


class MockGatedTool(Tool):
    """Test tool with controllable is_available() behavior."""

    name = "mock_gated_tool"
    simple_description = "Test tool that can toggle availability"
    anthropic_schema = {
        "name": "mock_gated_tool",
        "description": "Test tool for gated tools infrastructure",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"}
            },
            "required": ["action"]
        }
    }

    # Class-level flag for testing (allows external control)
    _available = False

    @classmethod
    def set_available(cls, available: bool):
        """Set availability state for testing."""
        cls._available = available

    def is_available(self) -> bool:
        """Gated tool availability check."""
        return MockGatedTool._available

    def run(self, action: str):
        return {"success": True, "action": action}


class MockAlwaysAvailableTool(Tool):
    """Test tool that is always available when gated."""

    name = "mock_always_available"
    simple_description = "Always available gated tool"
    anthropic_schema = {
        "name": "mock_always_available",
        "description": "Test tool always available",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }

    def is_available(self) -> bool:
        return True

    def run(self):
        return {"success": True}


class TestGatedToolsInfrastructure:
    """Tests for gated tools registration and availability checking."""

    @pytest.fixture
    def tool_repo(self):
        """Create a fresh tool repository."""
        repo = ToolRepository(working_memory=None)
        # Register test tools
        repo.register_tool_class(MockGatedTool, "mock_gated_tool")
        repo.register_tool_class(MockAlwaysAvailableTool, "mock_always_available")
        return repo

    def test_gated_tools_set_initialized_empty(self, tool_repo):
        """Verify gated_tools set starts empty."""
        assert isinstance(tool_repo.gated_tools, set)
        assert len(tool_repo.gated_tools) == 0

    def test_register_gated_tool_adds_to_set(self, tool_repo):
        """Verify register_gated_tool() adds tool to gated_tools set."""
        tool_repo.register_gated_tool("mock_gated_tool")

        assert "mock_gated_tool" in tool_repo.gated_tools
        assert len(tool_repo.gated_tools) == 1

    def test_register_gated_tool_rejects_unregistered_tool(self, tool_repo):
        """Verify register_gated_tool() raises KeyError for unregistered tool."""
        with pytest.raises(KeyError, match="must be registered before marking as gated"):
            tool_repo.register_gated_tool("nonexistent_tool")

    def test_gated_tool_excluded_when_unavailable(self, tool_repo):
        """Verify gated tool excluded from definitions when is_available() returns False."""
        # Register as gated
        tool_repo.register_gated_tool("mock_gated_tool")

        # Set unavailable
        MockGatedTool.set_available(False)

        definitions = tool_repo.get_all_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        assert "mock_gated_tool" not in tool_names

    def test_gated_tool_included_when_available(self, tool_repo):
        """Verify gated tool included in definitions when is_available() returns True."""
        # Register as gated
        tool_repo.register_gated_tool("mock_gated_tool")

        # Set available
        MockGatedTool.set_available(True)

        definitions = tool_repo.get_all_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        assert "mock_gated_tool" in tool_names

    def test_gated_tool_always_available(self, tool_repo):
        """Verify tool with is_available() returning True always appears."""
        tool_repo.register_gated_tool("mock_always_available")

        definitions = tool_repo.get_all_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        assert "mock_always_available" in tool_names

    def test_multiple_gated_tools(self, tool_repo):
        """Verify multiple gated tools can be registered."""
        tool_repo.register_gated_tool("mock_gated_tool")
        tool_repo.register_gated_tool("mock_always_available")

        assert len(tool_repo.gated_tools) == 2
        assert "mock_gated_tool" in tool_repo.gated_tools
        assert "mock_always_available" in tool_repo.gated_tools

    def test_gated_and_enabled_tools_coexist(self, tool_repo):
        """Verify gated tools and enabled tools both appear in definitions."""
        # Register gated tool (available)
        tool_repo.register_gated_tool("mock_always_available")

        # Enable a standard tool
        tool_repo.enable_tool("mock_gated_tool")

        definitions = tool_repo.get_all_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        # Both should appear
        assert "mock_always_available" in tool_names
        assert "mock_gated_tool" in tool_names

    def test_gated_tool_availability_changes_dynamically(self, tool_repo):
        """Verify gated tool appears/disappears as availability changes."""
        tool_repo.register_gated_tool("mock_gated_tool")

        # Initially unavailable
        MockGatedTool.set_available(False)
        definitions = tool_repo.get_all_tool_definitions()
        assert "mock_gated_tool" not in [d["name"] for d in definitions]

        # Now available
        MockGatedTool.set_available(True)
        definitions = tool_repo.get_all_tool_definitions()
        assert "mock_gated_tool" in [d["name"] for d in definitions]

        # Back to unavailable
        MockGatedTool.set_available(False)
        definitions = tool_repo.get_all_tool_definitions()
        assert "mock_gated_tool" not in [d["name"] for d in definitions]

    def test_enable_tool_rejects_gated_tools(self, tool_repo):
        """Verify enable_tool() rejects gated tools to enforce mutual exclusivity.

        Gated tools use is_available() for dynamic availability control.
        They cannot be added to enabled_tools - the two patterns are mutually exclusive.
        """
        tool_repo.register_gated_tool("mock_gated_tool")

        # Attempting to enable a gated tool should raise ValueError
        with pytest.raises(ValueError, match="Cannot enable gated tool"):
            tool_repo.enable_tool("mock_gated_tool")

        # Tool should remain only in gated_tools, not enabled_tools
        assert "mock_gated_tool" in tool_repo.gated_tools
        assert "mock_gated_tool" not in tool_repo.enabled_tools

    def test_gated_tool_exception_handled_gracefully(self, tool_repo):
        """Verify exceptions in is_available() don't break get_all_tool_definitions()."""

        class BrokenGatedTool(Tool):
            name = "broken_gated_tool"
            anthropic_schema = {"name": "broken_gated_tool", "input_schema": {}}

            def is_available(self):
                raise RuntimeError("Broken is_available")

            def run(self):
                pass

        tool_repo.register_tool_class(BrokenGatedTool, "broken_gated_tool")
        tool_repo.register_gated_tool("broken_gated_tool")

        # Should not raise, just skip the broken tool
        definitions = tool_repo.get_all_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        assert "broken_gated_tool" not in tool_names


class TestGatedToolInvocation:
    """Tests for invoking gated tools via invoke_tool()."""

    @pytest.fixture
    def tool_repo(self):
        """Create a fresh tool repository."""
        repo = ToolRepository(working_memory=None)
        repo.register_tool_class(MockGatedTool, "mock_gated_tool")
        repo.register_tool_class(MockAlwaysAvailableTool, "mock_always_available")
        return repo

    def test_invoke_gated_tool_when_available(self, tool_repo):
        """Verify gated tool can be invoked when is_available() returns True."""
        tool_repo.register_gated_tool("mock_gated_tool")
        MockGatedTool.set_available(True)

        # Should succeed - gated tool is available
        result = tool_repo.invoke_tool("mock_gated_tool", {"action": "test"})

        assert result["success"] is True
        assert result["action"] == "test"

    def test_invoke_gated_tool_when_unavailable_raises(self, tool_repo):
        """Verify gated tool invocation raises when is_available() returns False."""
        tool_repo.register_gated_tool("mock_gated_tool")
        MockGatedTool.set_available(False)

        # Should raise - gated tool is not available
        with pytest.raises(RuntimeError, match="Tool is not available"):
            tool_repo.invoke_tool("mock_gated_tool", {"action": "test"})

    def test_invoke_always_available_gated_tool(self, tool_repo):
        """Verify always-available gated tool can be invoked."""
        tool_repo.register_gated_tool("mock_always_available")

        # Should succeed
        result = tool_repo.invoke_tool("mock_always_available", {})

        assert result["success"] is True

    def test_invoke_gated_tool_not_in_enabled_tools(self, tool_repo):
        """Verify gated tools don't need to be in enabled_tools to invoke."""
        tool_repo.register_gated_tool("mock_gated_tool")
        MockGatedTool.set_available(True)

        # Verify tool is NOT in enabled_tools
        assert "mock_gated_tool" not in tool_repo.enabled_tools

        # But invocation should still work
        result = tool_repo.invoke_tool("mock_gated_tool", {"action": "test"})
        assert result["success"] is True

    def test_invoke_non_gated_non_enabled_tool_raises(self, tool_repo):
        """Verify non-gated, non-enabled tool cannot be invoked."""
        # Tool is registered but not gated and not enabled
        with pytest.raises(RuntimeError, match="Tool is not enabled"):
            tool_repo.invoke_tool("mock_gated_tool", {"action": "test"})
