"""
Tests for the extraction tool.
"""
import pytest
from unittest.mock import MagicMock, patch

from tools.extraction_tool import ExtractionTool
from errors import ToolError


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response object."""
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "Extracted information"
    mock_response.content = [mock_content_block]
    return mock_response


@pytest.fixture
def extraction_tool():
    """Create an ExtractionTool instance for testing."""
    with patch('tools.extraction_tool.LLMBridge') as mock_llm_bridge_class:
        # Set up the mock for LLMBridge
        mock_llm_bridge = MagicMock()
        mock_llm_bridge_class.return_value = mock_llm_bridge
        
        # Create instance
        tool = ExtractionTool()
        
        # Attach the mock for use in tests
        tool._mock_llm_bridge = mock_llm_bridge
        
        yield tool


def test_run_with_general_template(extraction_tool, mock_llm_response):
    """Test running the tool with the general template."""
    # Set up the mock
    extraction_tool._mock_llm_bridge.generate_response.return_value = mock_llm_response
    extraction_tool._mock_llm_bridge.extract_text_content.return_value = "John Doe"
    
    # Run the tool
    result = extraction_tool.run(
        message="My name is John Doe",
        template="general",
        target="person's name"
    )
    
    # Verify results
    assert result["extracted"] == "John Doe"
    
    # Verify the LLM was called with correct parameters
    extraction_tool._mock_llm_bridge.generate_response.assert_called_once()
    # Get the args from the call
    call_args = extraction_tool._mock_llm_bridge.generate_response.call_args[1]
    assert "messages" in call_args
    assert call_args["temperature"] == 0.3
    assert "system_prompt" in call_args


def test_run_with_specific_template(extraction_tool, mock_llm_response):
    """Test running the tool with a specific template."""
    # Set up the mock
    extraction_tool._mock_llm_bridge.generate_response.return_value = mock_llm_response
    extraction_tool._mock_llm_bridge.extract_text_content.return_value = "positive"
    
    # Run the tool
    result = extraction_tool.run(
        message="I'm having a great day today!",
        template="sentiment"
    )
    
    # Verify results
    assert result["extracted"] == "positive"


def test_invalid_template(extraction_tool):
    """Test providing an invalid template."""
    with pytest.raises(ToolError) as e:
        extraction_tool.run(
            message="Test message",
            template="non_existent_template"
        )
    assert "Invalid template" in str(e.value)


def test_missing_target_with_general_template(extraction_tool):
    """Test missing target parameter with general template."""
    with pytest.raises(ToolError) as e:
        extraction_tool.run(
            message="Test message",
            template="general"
        )
    assert "Target parameter is required" in str(e.value)


def test_custom_template(extraction_tool, mock_llm_response):
    """Test using a custom template."""
    # Set up the mock
    extraction_tool._mock_llm_bridge.generate_response.return_value = mock_llm_response
    extraction_tool._mock_llm_bridge.extract_text_content.return_value = "Custom extraction result"
    
    # Run the tool with custom template
    result = extraction_tool.run(
        message="Test message with custom extraction needs",
        template="custom",
        target="Extract all the numbers mentioned in the message."
    )
    
    # Verify results
    assert result["extracted"] == "Custom extraction result"


def test_list_templates(extraction_tool):
    """Test listing available templates."""
    templates = extraction_tool.list_templates()
    
    # Verify standard templates are included
    assert "general" in templates
    assert "personal_info" in templates
    assert "keywords" in templates
    assert "custom" in templates