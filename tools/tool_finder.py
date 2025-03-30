"""
Tool finder meta-tool for discovering and requesting additional tools.

This module provides a meta-tool that allows the model to discover and request
tools that weren't initially included in the tool set.
"""
import re
from typing import Dict, List, Any, Optional, Union

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context


class ToolFinderTool(Tool):
    """
    Meta-tool for discovering and requesting additional tools.
    
    Allows the model to request tools that weren't included in the
    initial selection, serving as a fallback mechanism.
    """
    
    name = "tool_finder"
    description = "Find or request tools by name or functionality description"
    usage_examples = [
        {
            "input": {"tool_name": "persistence"},
            "output": {
                "tool": {
                    "name": "persistence",
                    "description": "Store and retrieve data with dedicated getter and setter operations",
                    "input_schema": {"type": "object", "properties": {...}}
                }
            }
        },
        {
            "input": {"description": "I need to save data"},
            "output": {
                "matching_tools": [
                    {"name": "persistence", "description": "Store and retrieve data"},
                    {"name": "store_preferences", "description": "Save user preferences"}
                ]
            }
        }
    ]
    
    def __init__(self, tool_repo):
        """
        Initialize the tool finder meta-tool.
        
        Args:
            tool_repo: Repository of all available tools
        """
        super().__init__()
        self.tool_repo = tool_repo
        self.tool_summaries = self._generate_tool_summaries()
    
    def run(
        self, 
        tool_name: Optional[str] = None, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find or request tools by name or description.
        
        Args:
            tool_name: Specific tool name to request
            description: Description of the functionality needed
            
        Returns:
            Dictionary with tool definitions or matching tools
            
        Raises:
            ToolError: If both tool_name and description are missing
        """
        # Use centralized error context for standardized error handling
        with error_context(
            component_name=self.name,
            operation="finding tools",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_INVALID_INPUT,
            logger=self.logger
        ):
            # Validate input - require at least one parameter
            if not tool_name and not description:
                raise ToolError(
                    "Either tool_name or description must be provided",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Case 1: Specific tool requested by name
            if tool_name:
                return self._get_tool_by_name(tool_name)
            
            # Case 2: Tools requested by description
            return self._find_tools_by_description(description)
    
    def _get_tool_by_name(self, tool_name: str) -> Dict[str, Any]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Dictionary with the tool definition
            
        Raises:
            ToolError: If the tool is not found
        """
        # Check if the tool exists
        tool = self.tool_repo.get_tool(tool_name)
        if not tool:
            return {
                "error": f"Tool not found: {tool_name}",
                "available_tools": self._list_available_tools()
            }
        
        # Get the tool definition
        tool_definition = tool.get_tool_definition()
        
        return {"tool": tool_definition}
    
    def _find_tools_by_description(self, description: str) -> Dict[str, Any]:
        """
        Find tools matching a description.
        
        Args:
            description: Description of the functionality needed
            
        Returns:
            Dictionary with matching tools
        """
        matching_tools = []
        
        # Tokenize description for better matching
        description_tokens = set(re.findall(r'\b\w+\b', description.lower()))
        
        # Score each tool based on description match
        tool_scores = {}
        
        for tool_name, summary in self.tool_summaries.items():
            # Calculate match score based on keyword overlap
            tool_tokens = set(re.findall(r'\b\w+\b', summary.lower()))
            overlap = description_tokens.intersection(tool_tokens)
            
            if overlap:
                # Score based on percentage of overlapping words
                score = len(overlap) / len(description_tokens)
                tool_scores[tool_name] = score
        
        # Get all tools (we'll sort by score)
        all_tools = {
            name: self.tool_repo.get_tool(name)
            for name in tool_scores.keys()
        }
        
        # Sort by score and build result list
        for tool_name, score in sorted(tool_scores.items(), key=lambda x: x[1], reverse=True):
            if tool_name in all_tools and all_tools[tool_name]:
                tool = all_tools[tool_name]
                
                matching_tools.append({
                    "name": tool_name,
                    "description": tool.description,
                    "match_score": round(score, 2)
                })
        
        # If no matches found, return all available tools
        if not matching_tools:
            return {
                "message": "No matching tools found",
                "available_tools": self._list_available_tools()
            }
        
        return {"matching_tools": matching_tools[:5]}  # Return top 5 matches
    
    def _list_available_tools(self) -> List[Dict[str, str]]:
        """
        Get a list of all available tools.
        
        Returns:
            List of tool name and description pairs
        """
        tool_list = []
        
        for tool_name, summary in self.tool_summaries.items():
            tool_list.append({
                "name": tool_name,
                "description": summary
            })
        
        return sorted(tool_list, key=lambda x: x["name"])
    
    def _generate_tool_summaries(self) -> Dict[str, str]:
        """
        Generate concise summaries of all tools for matching.
        
        Returns:
            Dictionary mapping tool names to summaries
        """
        summaries = {}
        
        for tool_name, tool in self.tool_repo.tools.items():
            # Skip self to prevent recursion
            if tool_name == self.name:
                continue
                
            # Get description from tool class or instance
            if hasattr(tool, 'description'):
                summaries[tool_name] = tool.description
            
        return summaries
    
