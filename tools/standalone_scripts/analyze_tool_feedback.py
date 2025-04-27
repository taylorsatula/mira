#!/usr/bin/env python
"""
Tool Feedback Analysis Script

This script allows viewing and analyzing collected tool feedback data.
It can be used to:
1. List all feedback entries
2. View detailed information for specific feedback
3. Extract common patterns in feedback
4. Generate reports on tool classification accuracy
"""
import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to sys.path to allow importing from parent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from tools.tool_feedback import get_feedback_summary


def list_feedback(show_analysis: bool = False) -> None:
    """
    List all feedback entries with optional analysis.
    
    Args:
        show_analysis: Whether to include LLM analysis in the output
    """
    summary = get_feedback_summary()
    
    print(f"Found {summary['count']} feedback entries")
    
    if not summary['count']:
        return
    
    print("\nTool Mentions:")
    for tool, count in summary['tools'].items():
        print(f"  {tool}: {count} mentions")
    
    print("\nRecent Feedback:")
    for i, entry in enumerate(summary['feedback'], 1):
        timestamp = entry.get('timestamp', 'Unknown time')
        feedback = entry.get('feedback', 'No feedback text')
        
        # Format timestamp for readability
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        print(f"{i}. [{timestamp}] {feedback}")
        
        if show_analysis and 'llm_analysis' in entry:
            analysis = entry['llm_analysis'].get('analysis', 'No analysis available')
            if len(analysis) > 500:
                analysis = analysis[:497] + "..."
            print(f"   Analysis: {analysis}")
            print()


def view_feedback(feedback_id: str) -> None:
    """
    View detailed information for a specific feedback entry.
    
    Args:
        feedback_id: ID of the feedback entry to view (filename without extension)
    """
    feedback_dir = Path(config.paths.persistent_dir) / "tool_feedback"
    
    if not feedback_dir.exists():
        print("No feedback directory found")
        return
    
    # Find the specific feedback file
    feedback_file = feedback_dir / f"{feedback_id}.json"
    if not feedback_file.exists():
        # Try with full filename
        feedback_file = feedback_dir / feedback_id
        if not feedback_file.exists():
            print(f"Feedback ID '{feedback_id}' not found")
            return
    
    # Read and display the feedback
    try:
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
        
        print("=" * 80)
        print(f"Feedback: {feedback_file.name}")
        print("=" * 80)
        
        # Display basic information
        timestamp = feedback.get('timestamp', 'Unknown time')
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        print(f"Time: {timestamp}")
        print(f"Conversation ID: {feedback.get('conversation_id', 'Not specified')}")
        print(f"Feedback: {feedback.get('feedback', 'No feedback text')}")
        
        # Display active tools and thresholds
        active_tools = feedback.get('active_tools', [])
        tool_thresholds = feedback.get('tool_thresholds', {})
        
        print(f"\nActive Tools: {', '.join(active_tools) if active_tools else 'None'}")
        
        # Display thresholds if available
        if tool_thresholds:
            print("\nTool Thresholds:")
            for tool, threshold in tool_thresholds.items():
                print(f"  {tool}: {threshold:.4f}")
        
        # Display recent messages
        print("\nRecent Messages:")
        for i, msg in enumerate(feedback.get('last_messages', []), 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', 'No content')
            print(f"  {i}. [{role}] {content}")
        
        # Display nearest examples
        print("\nNearest Training Examples:")
        nearest = feedback.get('nearest_examples', {})
        if nearest:
            for tool, examples in nearest.items():
                print(f"  Tool: {tool}")
                for j, ex in enumerate(examples, 1):
                    query = ex.get('query', 'No query')
                    similarity = ex.get('similarity', 0)
                    print(f"    {j}. [{similarity:.4f}] {query}")
        else:
            print("  No nearest examples found")
        
        # Display LLM analysis
        llm_analysis = feedback.get('llm_analysis', {})
        if llm_analysis:
            print("\nLLM Analysis:")
            analysis_text = llm_analysis.get('analysis', 'No analysis available')
            print(analysis_text)
        
    except Exception as e:
        print(f"Error reading feedback file: {e}")


def main():
    """Run the tool feedback analysis script."""
    parser = argparse.ArgumentParser(description="Analyze tool feedback data")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'list' command
    list_parser = subparsers.add_parser("list", help="List all feedback entries")
    list_parser.add_argument("--with-analysis", action="store_true", 
                           help="Include LLM analysis in the output")
    
    # 'view' command
    view_parser = subparsers.add_parser("view", help="View detailed information for a specific feedback")
    view_parser.add_argument("feedback_id", help="ID of the feedback entry to view (filename without extension)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "list":
        list_feedback(args.with_analysis)
    elif args.command == "view":
        view_feedback(args.feedback_id)
    else:
        # Default to listing feedback
        list_feedback()


if __name__ == "__main__":
    main()