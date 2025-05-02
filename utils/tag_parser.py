"""
Parser for special markup tags in assistant responses.

This module provides utilities for detecting and extracting tag-based commands
and metadata that might be present in assistant responses.
"""
import re
import logging
from typing import Dict, Any, Optional, List


class TagParser:
    """
    Parser for extracting semantic tags from assistant responses.
    
    This class handles detection and extraction of all supported tag formats
    in one centralized location, making it easier to add new tags or modify
    existing ones without changes scattered throughout the codebase.
    """
    
    def __init__(self):
        """Initialize the tag parser with logger."""
        self.logger = logging.getLogger("tag_parser")
    
    def parse_tags(self, text: str) -> Dict[str, Any]:
        """
        Parse and extract all supported tags from text.
        
        Supported tags:
        - <topic_changed=true/> or <topic_changed=false/>
        - <need_tool />
        - <workflow_start:workflow_id />
        - <workflow_complete />
        - <workflow_cancel />
        
        Args:
            text: The text to parse for tags
            
        Returns:
            Dictionary with parsed tag values:
            {
                "topic_changed": True/False,
                "need_tool": True/False,
                "workflow": {
                    "action": "start"/"complete"/"cancel",
                    "id": "workflow_id" (only for "start" action)
                }
            }
        """
        # Initialize result with default values
        result = {
            "topic_changed": False,
            "need_tool": False,
            "workflow": {
                "action": None,
                "id": None
            }
        }
        
        # Check for topic_changed tag
        topic_changed_match = re.search(r'<topic_changed=(true|false)\s*/?>', text, re.IGNORECASE)
        if topic_changed_match:
            topic_changed_value = topic_changed_match.group(1).lower()
            result["topic_changed"] = (topic_changed_value == 'true')
            self.logger.info(f"Found topic_changed tag: {topic_changed_value}")
        
        # Check for need_tool tag
        need_tool_match = re.search(r'<need_tool\s*/?>', text, re.IGNORECASE)
        if need_tool_match:
            result["need_tool"] = True
            self.logger.info("Found need_tool tag")
        
        # Check for workflow tags
        workflow_start_match = re.search(r'<workflow_start:([a-zA-Z0-9_-]+)\s*/?>', text)
        if workflow_start_match:
            result["workflow"]["action"] = "start"
            result["workflow"]["id"] = workflow_start_match.group(1)
            self.logger.info(f"Found workflow_start tag: {result['workflow']['id']}")
        
        workflow_complete_match = re.search(r'<workflow_complete\s*/?>', text, re.IGNORECASE)
        if workflow_complete_match:
            result["workflow"]["action"] = "complete"
            self.logger.info("Found workflow_complete tag")
        
        workflow_cancel_match = re.search(r'<workflow_cancel\s*/?>', text, re.IGNORECASE)
        if workflow_cancel_match:
            result["workflow"]["action"] = "cancel"
            self.logger.info("Found workflow_cancel tag")
        
        return result
    
    def extract_topic_changed(self, text: str) -> bool:
        """
        Extract topic change status from text.
        
        Args:
            text: The text to check for the topic_changed tag
            
        Returns:
            Boolean indicating whether the topic has changed
        """
        tags = self.parse_tags(text)
        return tags["topic_changed"]
    
    def has_need_tool(self, text: str) -> bool:
        """
        Check if the text contains the need_tool tag.
        
        Args:
            text: The text to check for the need_tool tag
            
        Returns:
            Boolean indicating whether the need_tool tag is present
        """
        # Special case for when text is exactly the need_tool tag
        if text.strip() == "<need_tool />":
            return True
            
        tags = self.parse_tags(text)
        return tags["need_tool"]
    
    def get_workflow_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow action information from text.
        
        Args:
            text: The text to check for workflow tags
            
        Returns:
            Dictionary with action and id keys if a workflow tag is found,
            None otherwise
        """
        tags = self.parse_tags(text)
        workflow = tags["workflow"]
        
        if workflow["action"] is not None:
            return workflow
        return None


# Create a singleton instance for reuse
parser = TagParser()