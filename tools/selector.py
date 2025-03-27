"""
Tool selection system for dynamically selecting relevant tools based on user input.

This module provides a tool selection mechanism that analyzes user messages
to determine which tools are most likely to be needed, reducing token usage
and improving response times.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from errors import ToolError, ErrorCode, error_context


class ToolSelector:
    """
    Selector for choosing relevant tools based on message content.
    
    Analyzes user messages to select the most relevant tools,
    reducing token usage and improving response times.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the tool selector.
        
        Args:
            storage_path: Path to store selector data (optional)
        """
        self.logger = logging.getLogger("tool_selector")
        
        # Initialize storage location
        from config import config
        self.storage_path = storage_path or Path(config.paths.selector_data_path)
        
        # Initialize data structures
        self.essential_tools = ["tool_finder"]  # Always include these tools
        self.keyword_map = {}  # Maps words to relevant tools
        self.tool_usage = {}  # Tracks tool usage frequency
        self.recent_tools = []  # Tracks recently used tools
        self.tool_misses = {}  # Tracks tools that were requested but not initially selected
        
        # Load existing data or create defaults
        self._load_data()
        
        self.logger.info(f"Tool selector initialized with {len(self.keyword_map)} keyword mappings")
    
    def select_tools(
        self, 
        message: str, 
        all_tools: List[Dict[str, Any]], 
        min_tools: int = 3, 
        max_tools: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Select tools based on message content.
        
        Args:
            message: User message text
            all_tools: List of all available tool definitions
            min_tools: Minimum number of tools to include
            max_tools: Maximum number of tools to include
            
        Returns:
            List of selected tool definitions
        """
        self.logger.debug(f"Selecting tools for message: {message[:50]}...")
        
        # Get tool names for lookup
        tool_name_map = {tool["name"]: tool for tool in all_tools}
        
        # Track scores for each tool
        tool_scores = {tool["name"]: 0 for tool in all_tools}
        
        # 1. Add score for essential tools
        for tool_name in self.essential_tools:
            if tool_name in tool_scores:
                tool_scores[tool_name] += 10  # High score for essential tools
        
        # 2. Analyze message content for tool-relevant keywords
        self._score_tools_by_keywords(message, tool_scores)
        
        # 3. Add score for recent tools
        self._score_recent_tools(tool_scores)
        
        # 4. Add score for frequently used tools
        self._score_frequently_used_tools(tool_scores)
        
        # 5. Add score for previously missed tools
        self._score_previously_missed_tools(tool_scores)
        
        # Sort tools by score (descending)
        sorted_tools = sorted(
            tool_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Determine how many tools to include (between min and max)
        num_tools = min(max(min_tools, len(sorted_tools)), max_tools)
        
        # Filter out tools with zero score if we have enough tools
        selected_tools = [
            tool_name_map[name] 
            for name, score in sorted_tools[:num_tools] 
            if score > 0 or len([t for t, s in sorted_tools if s > 0]) < min_tools
        ]
        
        # Ensure we have at least min_tools
        if len(selected_tools) < min_tools and len(all_tools) >= min_tools:
            # Add remaining tools in order until we reach min_tools
            for name, score in sorted_tools[len(selected_tools):]:
                if name in tool_name_map and tool_name_map[name] not in selected_tools:
                    selected_tools.append(tool_name_map[name])
                    if len(selected_tools) >= min_tools:
                        break
        
        # Log selection results
        selected_names = [tool["name"] for tool in selected_tools]
        self.logger.info(f"Selected {len(selected_tools)} tools: {selected_names}")
        
        return selected_tools
    
    def record_usage(self, tool_name: str, was_selected: bool = True) -> None:
        """
        Record tool usage statistics.
        
        Args:
            tool_name: Name of the tool that was used
            was_selected: Whether the tool was in the initial selection
        """
        # Update usage count
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
        
        # Add to recent tools (at the beginning of the list)
        if tool_name in self.recent_tools:
            self.recent_tools.remove(tool_name)
        self.recent_tools.insert(0, tool_name)
        
        # Limit recent tools list
        self.recent_tools = self.recent_tools[:10]  # Keep only 10 most recent
        
        # Track misses (tools that were requested but not initially selected)
        if not was_selected:
            self.tool_misses[tool_name] = self.tool_misses.get(tool_name, 0) + 1
            self.logger.info(f"Recorded tool miss for: {tool_name}")
            
            # If a tool is frequently missed, add it to essential tools
            if self.tool_misses.get(tool_name, 0) >= 3 and tool_name not in self.essential_tools:
                self.essential_tools.append(tool_name)
                self.logger.info(f"Added frequently missed tool to essential list: {tool_name}")
        
        # Save updated data
        self._save_data()
    
    def add_keyword_mapping(self, keyword: str, tool_name: str) -> None:
        """
        Add a keyword to tool mapping.
        
        Args:
            keyword: Keyword to match in messages
            tool_name: Name of the tool to associate with the keyword
        """
        lower_keyword = keyword.lower()
        if lower_keyword not in self.keyword_map:
            self.keyword_map[lower_keyword] = []
        
        if tool_name not in self.keyword_map[lower_keyword]:
            self.keyword_map[lower_keyword].append(tool_name)
            self.logger.debug(f"Added keyword mapping: '{lower_keyword}' -> {tool_name}")
            self._save_data()
    
    def _score_tools_by_keywords(self, message: str, tool_scores: Dict[str, int]) -> None:
        """
        Score tools based on keyword matches in the message.
        
        Args:
            message: User message text
            tool_scores: Dictionary of tool scores to update
        """
        # Convert message to lowercase for case-insensitive matching
        lower_message = message.lower()
        
        # Extract words and phrases from the message
        words = re.findall(r'\b\w+\b', lower_message)
        
        # Check for keyword matches
        for keyword, tools in self.keyword_map.items():
            if keyword in lower_message:
                # Keyword found, increment scores for associated tools
                for tool in tools:
                    if tool in tool_scores:
                        tool_scores[tool] += 2  # Base score for keyword match
                        
                        # Higher score for exact word matches
                        if keyword in words:
                            tool_scores[tool] += 1
    
    def _score_recent_tools(self, tool_scores: Dict[str, int]) -> None:
        """
        Score tools based on recent usage.
        
        Args:
            tool_scores: Dictionary of tool scores to update
        """
        # Give higher scores to recently used tools
        for i, tool in enumerate(self.recent_tools):
            if tool in tool_scores:
                # Score decreases with recency (10 - position in recent list)
                tool_scores[tool] += max(0, 5 - i)
    
    def _score_frequently_used_tools(self, tool_scores: Dict[str, int]) -> None:
        """
        Score tools based on usage frequency.
        
        Args:
            tool_scores: Dictionary of tool scores to update
        """
        if not self.tool_usage:
            return
            
        # Find the most frequently used tool
        max_usage = max(self.tool_usage.values())
        
        # Scale other tool scores relative to max usage
        for tool, usage in self.tool_usage.items():
            if tool in tool_scores and max_usage > 0:
                # Score proportional to usage frequency (0-3 range)
                tool_scores[tool] += min(3, int((usage / max_usage) * 3))
    
    def _score_previously_missed_tools(self, tool_scores: Dict[str, int]) -> None:
        """
        Score tools based on previous misses.
        
        Args:
            tool_scores: Dictionary of tool scores to update
        """
        # Give higher scores to tools that were previously missed
        for tool, misses in self.tool_misses.items():
            if tool in tool_scores:
                # Score based on number of misses (up to 3)
                tool_scores[tool] += min(3, misses)
    
    def _load_data(self) -> None:
        """
        Load selector data from disk.
        """
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                self.essential_tools = data.get("essential_tools", ["tool_finder"])
                self.keyword_map = data.get("keyword_map", {})
                self.tool_usage = data.get("tool_usage", {})
                self.recent_tools = data.get("recent_tools", [])
                self.tool_misses = data.get("tool_misses", {})
                
                self.logger.info(f"Loaded selector data from {self.storage_path}")
            else:
                # Initialize with default mappings
                self._initialize_default_mappings()
                self.logger.info("Created default tool selector mappings")
        except Exception as e:
            self.logger.error(f"Error loading selector data: {e}")
            # Initialize with defaults if loading fails
            self._initialize_default_mappings()
    
    def _save_data(self) -> None:
        """
        Save selector data to disk.
        """
        try:
            data = {
                "essential_tools": self.essential_tools,
                "keyword_map": self.keyword_map,
                "tool_usage": self.tool_usage,
                "recent_tools": self.recent_tools,
                "tool_misses": self.tool_misses
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved selector data to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error saving selector data: {e}")
    
    def _initialize_default_mappings(self) -> None:
        """
        Initialize default keyword mappings.
        """
        # Common default keyword mappings
        default_mappings = {
            "save": ["persistence"],
            "store": ["persistence"],
            "remember": ["persistence"],
            "retrieve": ["persistence"],
            "get": ["persistence"],
            "extract": ["extract"],
            "find": ["extract"],
            "analyze": ["extract"],
            "weather": ["weather_tool"],
            "forecast": ["weather_tool"],
            "temperature": ["weather_tool"],
            "async": ["async_task"],
            "background": ["async_task"],
            "later": ["async_task"],
            "task": ["async_task", "check_task"],
            "find": ["tool_finder"],
            "tool": ["tool_finder"],
            "help": ["tool_finder"]
        }
        
        # Set up initial mappings
        self.keyword_map = default_mappings
        self.essential_tools = ["tool_finder"]  # Always include the tool finder