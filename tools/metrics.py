"""
Metrics collector for the tool selection system.

This module tracks metrics related to tool selection and usage,
enabling optimization and monitoring of the system performance.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class ToolMetrics:
    """
    Metrics collector for the tool selection system.
    
    Tracks and analyzes metrics related to tool selection accuracy,
    context savings, and response time impact.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            storage_path: Path to store metrics data (optional)
        """
        self.logger = logging.getLogger("tool_metrics")
        
        # Initialize storage location
        from config import config
        self.storage_path = storage_path or Path(config.paths.metrics_data_path)
        
        # Initialize metrics storage
        self.metrics = {
            "selection_counts": {
                "total_selections": 0,
                "selected_tools": 0,
                "available_tools": 0
            },
            "usage_counts": {
                "tools_used": 0,
                "tools_used_from_selection": 0,
                "tools_used_from_finder": 0
            },
            "response_times": {
                "with_selection": [],
                "all_tools": []
            },
            "estimated_savings": {
                "total_tokens_saved": 0,
                "average_tokens_per_request": 0
            },
            "daily_metrics": {},
            "tool_sequences": {
                # Dictionary recording which tools are called after others
                # Format: {"tool1": {"tool2": 5, "tool3": 10}, ...}
                # Meaning: tool2 was called 5 times after tool1, tool3 was called 10 times after tool1
            }
        }
        
        # Runtime tracking
        self.current_selection: Set[str] = set()
        self.current_day = self._get_current_day()
        self.last_used_tool = None  # Track last used tool for sequence tracking
        
        # Load existing data
        self._load_data()
        
        self.logger.info("Tool metrics collector initialized")
    
    def record_tool_selection(
        self, 
        message: str, 
        selected_tools: List[Dict[str, Any]], 
        all_tools: List[Dict[str, Any]]
    ) -> None:
        """
        Record metrics about tool selection.
        
        Args:
            message: User message that triggered the selection
            selected_tools: List of selected tool definitions
            all_tools: List of all available tool definitions
        """
        # Extract tool names for easier tracking
        selected_names = {tool["name"] for tool in selected_tools}
        
        # Store current selection for usage tracking
        self.current_selection = selected_names
        
        # Update selection counts
        self.metrics["selection_counts"]["total_selections"] += 1
        self.metrics["selection_counts"]["selected_tools"] += len(selected_tools)
        self.metrics["selection_counts"]["available_tools"] += len(all_tools)
        
        # Calculate estimated token savings
        # Rough estimate: each tool definition is ~100 tokens
        tokens_saved = (len(all_tools) - len(selected_tools)) * 100
        self.metrics["estimated_savings"]["total_tokens_saved"] += tokens_saved
        
        # Update average
        total_selections = self.metrics["selection_counts"]["total_selections"]
        total_saved = self.metrics["estimated_savings"]["total_tokens_saved"]
        self.metrics["estimated_savings"]["average_tokens_per_request"] = total_saved / total_selections
        
        # Update daily metrics
        self._update_daily_metrics("selections", 1)
        self._update_daily_metrics("tools_selected", len(selected_tools))
        self._update_daily_metrics("token_savings", tokens_saved)
        
        self.logger.debug(f"Recorded selection of {len(selected_tools)} tools from {len(all_tools)} available")
        
        # Periodically save data
        if total_selections % 10 == 0:
            self._save_data()
    
    def record_tool_usage(self, tool_name: str, was_in_selection: bool, conversation_id: str = None) -> None:
        """
        Record metrics about tool usage.
        
        Args:
            tool_name: Name of the tool that was used
            was_in_selection: Whether the tool was in the initial selection
            conversation_id: Optional ID to track tools within same conversation
        """
        # Update usage counts
        self.metrics["usage_counts"]["tools_used"] += 1
        
        if was_in_selection:
            self.metrics["usage_counts"]["tools_used_from_selection"] += 1
        else:
            self.metrics["usage_counts"]["tools_used_from_finder"] += 1
        
        # Update daily metrics
        self._update_daily_metrics("tools_used", 1)
        if not was_in_selection:
            self._update_daily_metrics("tools_missed", 1)
            
        # Track tool sequences
        if self.last_used_tool is not None:
            # We have a previous tool, so record this sequence
            tool_sequences = self.metrics["tool_sequences"]
            
            # Initialize if this is the first time seeing the previous tool
            if self.last_used_tool not in tool_sequences:
                tool_sequences[self.last_used_tool] = {}
                
            # Increment the count for this sequence
            next_tools = tool_sequences[self.last_used_tool]
            next_tools[tool_name] = next_tools.get(tool_name, 0) + 1
            
            self.logger.debug(f"Recorded sequence: {self.last_used_tool} -> {tool_name}")
            
        # Update last used tool
        self.last_used_tool = tool_name
        
        self.logger.debug(f"Recorded usage of tool: {tool_name}, was_in_selection: {was_in_selection}")
        
        # Save after significant changes
        if self.metrics["usage_counts"]["tools_used"] % 10 == 0:
            self._save_data()
    
    def record_response_time(self, seconds: float, used_selection: bool = True) -> None:
        """
        Record response time metrics.
        
        Args:
            seconds: Response time in seconds
            used_selection: Whether the response used the tool selection system
        """
        # Store times (limit to 100 data points to avoid growing too large)
        if used_selection:
            times = self.metrics["response_times"]["with_selection"]
            times.append(seconds)
            if len(times) > 100:
                times.pop(0)
        else:
            times = self.metrics["response_times"]["all_tools"]
            times.append(seconds)
            if len(times) > 100:
                times.pop(0)
        
        # Update daily metrics
        self._update_daily_metrics("total_response_time", seconds)
        self._update_daily_metrics("response_time_count", 1)
        
        self.logger.debug(f"Recorded response time: {seconds:.2f}s, used_selection: {used_selection}")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a summary metrics report.
        
        Returns:
            Dictionary with summary metrics
        """
        # Ensure data is up-to-date
        self._check_day_rollover()
        
        # Calculate derived metrics
        selection_counts = self.metrics["selection_counts"]
        usage_counts = self.metrics["usage_counts"]
        
        # Selection accuracy
        if usage_counts["tools_used"] > 0:
            selection_accuracy = usage_counts["tools_used_from_selection"] / usage_counts["tools_used"]
        else:
            selection_accuracy = 0
        
        # Miss rate
        if usage_counts["tools_used"] > 0:
            miss_rate = usage_counts["tools_used_from_finder"] / usage_counts["tools_used"]
        else:
            miss_rate = 0
        
        # Tool reduction ratio
        if selection_counts["available_tools"] > 0:
            reduction_ratio = selection_counts["selected_tools"] / selection_counts["available_tools"]
        else:
            reduction_ratio = 0
        
        # Average response times
        with_selection_times = self.metrics["response_times"]["with_selection"]
        all_tools_times = self.metrics["response_times"]["all_tools"]
        
        avg_time_with_selection = sum(with_selection_times) / len(with_selection_times) if with_selection_times else 0
        avg_time_all_tools = sum(all_tools_times) / len(all_tools_times) if all_tools_times else 0
        
        # Daily metrics summary (last 7 days)
        daily_metrics = self.metrics["daily_metrics"]
        daily_summary = {}
        
        for day, metrics in sorted(daily_metrics.items(), reverse=True)[:7]:
            if "tools_used" in metrics and metrics.get("tools_used", 0) > 0:
                daily_summary[day] = {
                    "selections": metrics.get("selections", 0),
                    "tools_used": metrics.get("tools_used", 0),
                    "miss_rate": metrics.get("tools_missed", 0) / metrics.get("tools_used", 1),
                    "token_savings": metrics.get("token_savings", 0),
                    "avg_response_time": metrics.get("total_response_time", 0) / metrics.get("response_time_count", 1)
                }
        
        # Process tool sequence data for the report
        tool_sequence_data = {}
        for first_tool, next_tools in self.metrics.get("tool_sequences", {}).items():
            if next_tools:
                # Get top 3 most common next tools
                sorted_next = sorted(next_tools.items(), key=lambda x: x[1], reverse=True)[:3]
                tool_sequence_data[first_tool] = {
                    "next_tools": {t: c for t, c in sorted_next},
                    "total_sequences": sum(next_tools.values())
                }
        
        # Build report
        report = {
            "summary": {
                "total_selections": selection_counts["total_selections"],
                "selection_accuracy": selection_accuracy,
                "miss_rate": miss_rate,
                "tool_reduction_ratio": reduction_ratio,
                "avg_tokens_saved_per_request": self.metrics["estimated_savings"]["average_tokens_per_request"],
                "total_tokens_saved": self.metrics["estimated_savings"]["total_tokens_saved"]
            },
            "response_times": {
                "with_selection": avg_time_with_selection,
                "all_tools": avg_time_all_tools,
                "improvement": avg_time_all_tools - avg_time_with_selection if all_tools_times and with_selection_times else 0
            },
            "daily_summary": daily_summary,
            "tool_sequences": tool_sequence_data
        }
        
        return report
    
    def _update_daily_metrics(self, metric: str, value: float) -> None:
        """
        Update metrics for the current day.
        
        Args:
            metric: Name of the metric to update
            value: Value to add to the metric
        """
        # Check for day rollover
        self._check_day_rollover()
        
        # Ensure the current day exists in metrics
        day = self.current_day
        if day not in self.metrics["daily_metrics"]:
            self.metrics["daily_metrics"][day] = {}
        
        # Update the metric
        daily_data = self.metrics["daily_metrics"][day]
        daily_data[metric] = daily_data.get(metric, 0) + value
    
    def _check_day_rollover(self) -> None:
        """
        Check if the day has changed and update current_day.
        """
        today = self._get_current_day()
        if today != self.current_day:
            self.logger.info(f"Day rollover detected: {self.current_day} -> {today}")
            self.current_day = today
            self._save_data()  # Save data on day change
    
    def _get_current_day(self) -> str:
        """
        Get the current day as a string (YYYY-MM-DD).
        
        Returns:
            Current day as a string
        """
        return time.strftime("%Y-%m-%d")
    
    def _load_data(self) -> None:
        """
        Load metrics data from disk.
        """
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    stored_metrics = json.load(f)
                    
                # Update metrics with stored data, preserving new metrics structure
                for category in self.metrics:
                    if category in stored_metrics:
                        if isinstance(self.metrics[category], dict) and isinstance(stored_metrics[category], dict):
                            self.metrics[category].update(stored_metrics[category])
                        else:
                            self.metrics[category] = stored_metrics[category]
                
                self.logger.info(f"Loaded metrics data from {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error loading metrics data: {e}")
    
    def _save_data(self) -> None:
        """
        Save metrics data to disk.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            self.logger.debug(f"Saved metrics data to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics data: {e}")
    
    def is_tool_in_current_selection(self, tool_name: str) -> bool:
        """
        Check if a tool is in the current selection.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is in the current selection, False otherwise
        """
        return tool_name in self.current_selection
        
    def reset_sequence_tracking(self) -> None:
        """
        Reset the tool sequence tracking for a new conversation turn.
        """
        self.last_used_tool = None
        self.logger.debug("Reset tool sequence tracking")
        
    def get_likely_next_tools(self, tool_name: str, limit: int = 3) -> List[str]:
        """
        Get the most likely tools to be used after a specific tool.
        
        Args:
            tool_name: Name of the tool to get likely next tools for
            limit: Maximum number of tools to return
            
        Returns:
            List of tool names sorted by likelihood
        """
        tool_sequences = self.metrics.get("tool_sequences", {})
        
        # If we don't have data for this tool, return empty list
        if tool_name not in tool_sequences:
            return []
            
        # Get the tools used after this one and their counts
        next_tools = tool_sequences[tool_name]
        
        # Sort by count (descending) and return the top N
        sorted_tools = sorted(next_tools.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, count in sorted_tools[:limit]]