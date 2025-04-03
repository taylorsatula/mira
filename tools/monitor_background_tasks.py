#!/usr/bin/env python3
"""
Real-time monitor for background tasks.

This script provides a live view of background task activities
by reading the event stream from the async task manager.
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"

# Event types and colors
EVENT_COLORS = {
    "task_created": Colors.GREEN,
    "task_started": Colors.GREEN,
    "task_completed": Colors.GREEN,
    "task_failed": Colors.RED,
    "tool_call": Colors.YELLOW,
    "tool_result": Colors.CYAN,
    "llm_thinking": Colors.BLUE,
    "llm_response": Colors.WHITE,
    "error": Colors.RED,
    "info": Colors.MAGENTA
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor background tasks in real-time")
    parser.add_argument("--task", "-t", type=str, help="Filter events by task ID")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow the event log (like tail -f)")
    parser.add_argument("--clear", "-c", action="store_true", help="Clear the screen before starting")
    parser.add_argument("--compact", action="store_true", help="Show events in compact format")
    return parser.parse_args()

def format_timestamp(timestamp):
    """Format a Unix timestamp as a readable date/time."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def format_event(event, compact=False):
    """Format an event for display."""
    event_type = event.get("event_type", "unknown").lower()
    timestamp = event.get("timestamp", time.time())
    task_id = event.get("task_id", "unknown")
    
    # Get color for event type
    color = EVENT_COLORS.get(event_type, Colors.WHITE)
    
    # Format timestamp
    time_str = format_timestamp(timestamp)
    
    # Basic event header
    header = f"{color}{Colors.BOLD}[{time_str}] [{event_type.upper()}] [{task_id[:8]}]{Colors.RESET}"
    
    # Format details based on event type
    if event_type == "task_created":
        description = event.get("description", "No description")
        return f"{header} Created task: {description}"
    
    elif event_type == "task_started":
        description = event.get("description", "No description")
        return f"{header} Started task: {description}"
    
    elif event_type == "task_completed":
        result = event.get("result", "No result")
        if compact and len(result) > 80:
            result = result[:77] + "..."
        return f"{header} Completed task: {result}"
    
    elif event_type == "task_failed":
        error = event.get("error", "Unknown error")
        return f"{header} Task failed: {error}"
    
    elif event_type == "tool_call":
        tool_name = event.get("tool_name", "unknown_tool")
        tool_input = event.get("tool_input", {})
        
        # Format tool input for display
        if compact:
            input_str = str(tool_input)
            if len(input_str) > 60:
                input_str = input_str[:57] + "..."
        else:
            input_str = json.dumps(tool_input, indent=2)
            # Indent each line
            input_str = "\n    ".join(input_str.split("\n"))
        
        return f"{header} Calling tool: {tool_name}({input_str})"
    
    elif event_type == "tool_result":
        tool_name = event.get("tool_name", "unknown_tool")
        result = event.get("result", "No result")
        is_error = event.get("is_error", False)
        
        # Format result for display
        if compact and len(result) > 60:
            result = result[:57] + "..."
        
        status = "ERROR" if is_error else "SUCCESS"
        return f"{header} Tool {tool_name} {status}: {result}"
    
    elif event_type == "llm_thinking":
        message = event.get("message", event.get("prompt", "No content"))
        
        if compact and len(message) > 80:
            message = message[:77] + "..."
        
        return f"{header} {message}"
    
    elif event_type == "llm_response":
        content = event.get("content", "No content")
        has_tool_calls = event.get("has_tool_calls", False)
        num_tool_calls = event.get("num_tool_calls", 0)
        
        if compact:
            if len(content) > 60:
                content = content[:57] + "..."
            tool_info = f" (with {num_tool_calls} tool calls)" if has_tool_calls else ""
            return f"{header} Response{tool_info}: {content}"
        else:
            tool_info = f"\n    [Has {num_tool_calls} tool calls]" if has_tool_calls else ""
            return f"{header} Response:{tool_info}\n    {content}"
    
    elif event_type == "error":
        tool_name = event.get("tool_name", "")
        error = event.get("error", "Unknown error")
        context = f"Tool {tool_name}: " if tool_name else ""
        return f"{header} {context}{error}"
    
    elif event_type == "info":
        message = event.get("message", "No message")
        return f"{header} {message}"
    
    else:
        # Generic fallback
        return f"{header} {json.dumps(event, indent=2 if not compact else None)}"

def follow_events(event_file, task_filter=None, compact=False):
    """Follow the event log file in real-time."""
    try:
        with open(event_file, "r") as f:
            # First, read any existing content
            f.seek(0, os.SEEK_END)
            
            print(f"{Colors.BOLD}Starting real-time monitoring of background tasks...{Colors.RESET}")
            print(f"{Colors.BOLD}Press Ctrl+C to exit{Colors.RESET}")
            print()
            
            while True:
                line = f.readline()
                if line:
                    try:
                        event = json.loads(line.strip())
                        # Apply task filter if provided
                        if task_filter and event.get("task_id") != task_filter:
                            continue
                        
                        print(format_event(event, compact))
                    except json.JSONDecodeError:
                        print(f"{Colors.RED}Error parsing event: {line.strip()}{Colors.RESET}")
                else:
                    # No new lines, wait a bit
                    time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Event file not found. No background tasks have been run yet.{Colors.RESET}")
        sys.exit(1)

def show_events(event_file, task_filter=None, compact=False):
    """Show all events from the file."""
    try:
        with open(event_file, "r") as f:
            events = []
            for line in f:
                try:
                    event = json.loads(line.strip())
                    # Apply task filter if provided
                    if task_filter and event.get("task_id") != task_filter:
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    print(f"{Colors.RED}Error parsing event: {line.strip()}{Colors.RESET}")
            
            if not events:
                task_msg = f" for task {task_filter}" if task_filter else ""
                print(f"{Colors.YELLOW}No events found{task_msg}.{Colors.RESET}")
                return
            
            for event in events:
                print(format_event(event, compact))
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Event file not found. No background tasks have been run yet.{Colors.RESET}")
        sys.exit(1)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Event file path
    event_file = os.path.join(project_root, "data", "async_events.jsonl")
    
    # Clear screen if requested
    if args.clear:
        os.system("cls" if os.name == "nt" else "clear")
    
    # Follow or show events based on arguments
    if args.follow:
        follow_events(event_file, args.task, args.compact)
    else:
        show_events(event_file, args.task, args.compact)

if __name__ == "__main__":
    main()