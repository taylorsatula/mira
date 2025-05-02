"""
Tests for the tag parser module.

This module contains tests for the tag parser utility to ensure that it correctly
parses all supported tags in assistant responses.
"""
import pytest
from utils.tag_parser import TagParser


class TestTagParser:
    """Test cases for the TagParser class."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.parser = TagParser()
    
    def test_topic_changed_true(self):
        """Test that topic_changed=true is correctly parsed."""
        text = "This is a response with <topic_changed=true/>"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is True
        
    def test_topic_changed_false(self):
        """Test that topic_changed=false is correctly parsed."""
        text = "This is a response with <topic_changed=false/>"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is False
        
    def test_topic_changed_case_insensitive(self):
        """Test that topic_changed tag is case insensitive."""
        text = "This is a response with <TOPIC_CHANGED=true/>"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is True
        
    def test_topic_changed_with_spaces(self):
        """Test that topic_changed tag works with spaces."""
        text = "This is a response with <topic_changed = true />"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is True
        
    def test_need_tool(self):
        """Test that need_tool tag is correctly parsed."""
        text = "I need a tool for this <need_tool/>"
        result = self.parser.parse_tags(text)
        assert result["need_tool"] is True
        
    def test_need_tool_with_spaces(self):
        """Test that need_tool tag works with spaces."""
        text = "I need a tool for this <need_tool />"
        result = self.parser.parse_tags(text)
        assert result["need_tool"] is True
        
    def test_need_tool_case_insensitive(self):
        """Test that need_tool tag is case insensitive."""
        text = "I need a tool for this <NEED_TOOL/>"
        result = self.parser.parse_tags(text)
        assert result["need_tool"] is True
        
    def test_workflow_start(self):
        """Test that workflow_start tag is correctly parsed."""
        text = "Let's start workflow <workflow_start:appointment_booking/>"
        result = self.parser.parse_tags(text)
        assert result["workflow"]["action"] == "start"
        assert result["workflow"]["id"] == "appointment_booking"
        
    def test_workflow_start_with_spaces(self):
        """Test that workflow_start tag works with spaces."""
        text = "Let's start workflow <workflow_start: appointment_booking />"
        result = self.parser.parse_tags(text)
        assert result["workflow"]["action"] == "start"
        assert result["workflow"]["id"] == "appointment_booking"
        
    def test_workflow_complete(self):
        """Test that workflow_complete tag is correctly parsed."""
        text = "The workflow step is done <workflow_complete/>"
        result = self.parser.parse_tags(text)
        assert result["workflow"]["action"] == "complete"
        
    def test_workflow_cancel(self):
        """Test that workflow_cancel tag is correctly parsed."""
        text = "Let's cancel this workflow <workflow_cancel/>"
        result = self.parser.parse_tags(text)
        assert result["workflow"]["action"] == "cancel"
        
    def test_multiple_tags(self):
        """Test that multiple tags are correctly parsed."""
        text = "This response has <topic_changed=true/> and <need_tool/>"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is True
        assert result["need_tool"] is True
        
    def test_no_tags(self):
        """Test behavior with no tags."""
        text = "This response has no tags"
        result = self.parser.parse_tags(text)
        assert result["topic_changed"] is False
        assert result["need_tool"] is False
        assert result["workflow"]["action"] is None
        
    def test_helper_methods(self):
        """Test the helper methods for extracting specific tag values."""
        text = "This has <topic_changed=true/> and <need_tool/> and <workflow_start:test_workflow/>"
        
        # Test extract_topic_changed
        assert self.parser.extract_topic_changed(text) is True
        
        # Test has_need_tool
        assert self.parser.has_need_tool(text) is True
        
        # Test get_workflow_action
        workflow_action = self.parser.get_workflow_action(text)
        assert workflow_action is not None
        assert workflow_action["action"] == "start"
        assert workflow_action["id"] == "test_workflow"