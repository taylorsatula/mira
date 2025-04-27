#!/usr/bin/env python
"""
Dialogue Testing Tool

This script allows for rapid testing of multi-turn dialogues with the system.
It enables reusing saved dialogues or running predefined conversation sequences
to test tool activation, classification, and system responses.
"""
import os
import sys
import json
import time
import argparse
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to sys.path to allow importing from parent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from conversation import Conversation
from main import initialize_system
from errors import handle_error, AgentError, error_context, ErrorCode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s │ %(name)s │ %(message)s'
)
logger = logging.getLogger("dialogue_tester")


class DialogueTester:
    """
    Tool for testing dialogues with the system.
    
    Supports:
    - Loading predefined dialogue sequences from files
    - Creating and saving new test dialogues
    - Running conversation tests and measuring responses
    - Analyzing tool activation and classification
    """
    
    def __init__(self, system=None):
        """
        Initialize the dialogue tester.
        
        Args:
            system: Optional system dictionary with initialized components
        """
        # Use provided system or initialize a new one
        self.system = system or self._initialize_system()
        
        # Extract key components
        self.conversation = self.system['conversation']
        self.tool_repo = self.system.get('tool_repo')
        self.tool_relevance_engine = self.system.get('tool_relevance_engine')
        
        # Set up directories
        self.dialogues_dir = Path(config.paths.persistent_dir) / "test_dialogues"
        self.dialogues_dir.mkdir(exist_ok=True, parents=True)
        
        # Track test results
        self.current_dialogue = {
            "name": None,
            "description": None,
            "turns": [],
            "results": {
                "total_time": 0,
                "avg_response_time": 0,
                "tool_activations": {}
            }
        }
    
    def _initialize_system(self) -> Dict[str, Any]:
        """
        Initialize a new system instance for testing.
        
        Returns:
            System dictionary with components
        """
        logger.info("Initializing system components...")
        
        # Create minimal args for initialization
        class Args:
            def __init__(self):
                self.config = None
                self.conversation = None
                self.log_level = "INFO"
                self.stream_mode = False
                
        args = Args()
        
        # Initialize the system
        system = initialize_system(args)
        logger.info("System initialized successfully")
        
        return system
    
    def load_dialogue(self, dialogue_path: str) -> Dict[str, Any]:
        """
        Load a dialogue from a file.
        
        Args:
            dialogue_path: Path to the dialogue file
            
        Returns:
            The loaded dialogue
        """
        path = Path(dialogue_path)
        
        try:
            with open(path, 'r') as f:
                dialogue = json.load(f)
            
            logger.info(f"Loaded dialogue '{dialogue.get('name', 'unnamed')}' with {len(dialogue.get('turns', []))} turns")
            return dialogue
        except Exception as e:
            logger.error(f"Error loading dialogue: {e}")
            return None
    
    def save_dialogue(self, name: str = None, description: str = None) -> str:
        """
        Save the current dialogue to a file.
        
        Args:
            name: Optional name for the dialogue
            description: Optional description of the dialogue
            
        Returns:
            Path to the saved file
        """
        if name:
            self.current_dialogue["name"] = name
        
        if description:
            self.current_dialogue["description"] = description
        
        # Generate filename
        filename = self.current_dialogue.get("name", "dialogue")
        filename = filename.replace(" ", "_").lower()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.dialogues_dir / f"{filename}_{timestamp}.json"
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_dialogue, f, indent=2, default=lambda o: str(o))
            
            logger.info(f"Saved dialogue to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving dialogue: {e}")
            return None
    
    def run_dialogue(self, dialogue: Dict[str, Any], pause_between_turns: float = 0, save_results: bool = True) -> Dict[str, Any]:
        """
        Run a predefined dialogue and collect results.
        
        Args:
            dialogue: The dialogue to run
            pause_between_turns: Seconds to pause between turns (default: 0)
            save_results: Whether to save results to a file (default: True)
            
        Returns:
            Results of the dialogue execution
        """
        # Reset the conversation
        if hasattr(self.conversation, 'clear_history'):
            self.conversation.clear_history()
        
        # Initialize test run metadata
        dialogue_name = dialogue.get('name', 'unnamed')
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Reset the results
        results = {
            "dialogue_name": dialogue_name,
            "run_timestamp": timestamp.isoformat(),
            "total_time": 0,
            "response_times": [],
            "tool_activations": {},
            "turns": []
        }
        
        turns = dialogue.get("turns", [])
        if not turns:
            logger.warning("No turns found in dialogue")
            return results
        
        logger.info(f"Running dialogue '{dialogue_name}' with {len(turns)} turns")
        
        # Track overall execution time
        start_time = time.time()
        
        # Run each turn
        for i, turn in enumerate(turns):
            user_input = turn.get("user")
            expected_response = turn.get("expected_response")
            notes = turn.get("notes")
            
            if not user_input:
                logger.warning(f"No user input for turn {i+1}, skipping")
                continue
            
            logger.info(f"\nTurn {i+1}/{len(turns)}: {user_input}")
            if notes:
                logger.info(f"Notes: {notes}")
            
            # Generate response
            turn_start = time.time()
            with error_context(
                component_name="DialogueTester",
                operation="generating response",
                error_class=AgentError,
                logger=logger
            ):
                try:
                    response = self.conversation.generate_response(user_input)
                    turn_end = time.time()
                    turn_time = turn_end - turn_start
                    
                    # Create turn result
                    turn_result = {
                        "turn_number": i+1,
                        "user_input": user_input,
                        "response": response,
                        "response_time": turn_time,
                        "expected_response": expected_response,
                        "notes": notes,
                        "active_tools": []
                    }
                    
                    # Store the results
                    results["response_times"].append(turn_time)
                    
                    # Display the results
                    logger.info(f"Response ({turn_time:.2f}s): {response}")
                    
                    # Check for active tools
                    if self.tool_relevance_engine:
                        active_tools = list(self.tool_relevance_engine.tool_activation_history.keys())
                        turn_result["active_tools"] = active_tools
                        
                        for tool in active_tools:
                            results["tool_activations"][tool] = results["tool_activations"].get(tool, 0) + 1
                        
                        if active_tools:
                            logger.info(f"Active tools: {', '.join(active_tools)}")
                    
                    # Optional verification against expected response
                    if expected_response:
                        # Simple contains check - could be more sophisticated
                        contains_expected = expected_response.lower() in response.lower()
                        turn_result["contains_expected"] = contains_expected
                        
                        if contains_expected:
                            logger.info("✓ Response contains expected text")
                        else:
                            logger.warning("✗ Response does not contain expected text")
                            logger.warning(f"Expected to contain: {expected_response}")
                    
                    # Add the turn result to results
                    results["turns"].append(turn_result)
                    
                    # Pause between turns if requested
                    if pause_between_turns > 0 and i < len(turns) - 1:
                        time.sleep(pause_between_turns)
                
                except Exception as e:
                    error_message = handle_error(e)
                    logger.error(f"Error in turn {i+1}: {error_message}")
                    
                    # Add error turn to results
                    turn_result = {
                        "turn_number": i+1,
                        "user_input": user_input,
                        "error": str(error_message),
                        "expected_response": expected_response,
                        "notes": notes
                    }
                    results["turns"].append(turn_result)
        
        # Calculate overall results
        total_time = time.time() - start_time
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0
        
        results["total_time"] = total_time
        results["avg_response_time"] = avg_time
        
        # Log summary
        logger.info(f"\nDialogue completed in {total_time:.2f}s (avg {avg_time:.2f}s per turn)")
        
        if results["tool_activations"]:
            logger.info("Tool activations:")
            for tool, count in results["tool_activations"].items():
                logger.info(f"  {tool}: {count}")
        
        # Save results to file if requested
        if save_results:
            # Create results directory
            results_dir = Path(config.paths.persistent_dir) / "test_results"
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate filename based on dialogue name and timestamp
            safe_name = dialogue_name.replace(" ", "_").lower()
            results_file = results_dir / f"{safe_name}_results_{timestamp_str}.json"
            
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=lambda o: str(o))
                logger.info(f"Results saved to {results_file}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        
        return results
    
    def create_dialogue_interactive(self) -> Dict[str, Any]:
        """
        Create a new dialogue interactively.
        
        Returns:
            The created dialogue
        """
        print("\n=== Interactive Dialogue Creation ===")
        name = input("Dialogue name: ")
        description = input("Description: ")
        
        dialogue = {
            "name": name,
            "description": description,
            "turns": [],
            "created_at": datetime.datetime.now().isoformat()
        }
        
        print("\nEnter dialogue turns (leave user input empty to finish):")
        turn_num = 1
        
        while True:
            print(f"\nTurn {turn_num}:")
            user_input = input("User: ")
            
            if not user_input:
                break
            
            expected = input("Expected response contains (optional): ")
            
            turn = {
                "user": user_input,
                "expected_response": expected if expected else None,
                "notes": None
            }
            
            notes = input("Notes (optional): ")
            if notes:
                turn["notes"] = notes
            
            dialogue["turns"].append(turn)
            turn_num += 1
        
        print(f"\nCreated dialogue '{name}' with {len(dialogue['turns'])} turns")
        return dialogue
    
    def list_dialogues(self) -> List[Dict[str, Any]]:
        """
        List all available test dialogues.
        
        Returns:
            List of dialogue metadata
        """
        files = list(self.dialogues_dir.glob("*.json"))
        dialogues = []
        
        for file in files:
            try:
                with open(file, 'r') as f:
                    dialogue = json.load(f)
                
                dialogues.append({
                    "name": dialogue.get("name", "Unnamed"),
                    "description": dialogue.get("description", ""),
                    "turns": len(dialogue.get("turns", [])),
                    "file": str(file),
                    "created_at": dialogue.get("created_at", "Unknown")
                })
            except Exception as e:
                logger.warning(f"Error reading {file}: {e}")
        
        # Sort by creation time (newest first)
        dialogues.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return dialogues


def main():
    """Run the dialogue testing tool."""
    parser = argparse.ArgumentParser(description="Test dialogues with the system")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'list' command
    list_parser = subparsers.add_parser("list", help="List saved dialogues")
    
    # 'create' command
    create_parser = subparsers.add_parser("create", help="Create a new test dialogue")
    
    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run a test dialogue")
    run_parser.add_argument("dialogue", help="Path to dialogue file to run")
    run_parser.add_argument("--pause", type=float, default=0.0, 
                            help="Seconds to pause between turns (default: 0)")
    run_parser.add_argument("--no-save", action="store_true",
                           help="Don't save results to a file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create tester instance
    tester = DialogueTester()
    
    # Execute the appropriate command
    if args.command == "list":
        dialogues = tester.list_dialogues()
        
        if not dialogues:
            print("No dialogues found")
            return
        
        print(f"\nFound {len(dialogues)} dialogues:")
        for i, dialogue in enumerate(dialogues, 1):
            created = dialogue.get("created_at", "Unknown")
            try:
                # Format timestamp if it's in ISO format
                dt = datetime.datetime.fromisoformat(created)
                created = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
            
            print(f"{i}. {dialogue['name']} - {dialogue['turns']} turns - {created}")
            print(f"   {dialogue['description']}")
            print(f"   File: {dialogue['file']}")
            print()
    
    elif args.command == "create":
        dialogue = tester.create_dialogue_interactive()
        
        # Ask if the user wants to save
        save = input("\nSave this dialogue? (y/n): ")
        if save.lower() == 'y':
            path = tester.save_dialogue(dialogue["name"], dialogue["description"])
            if path:
                print(f"Dialogue saved to {path}")
                
                # Ask if the user wants to run it
                run = input("Run this dialogue now? (y/n): ")
                if run.lower() == 'y':
                    results = tester.run_dialogue(dialogue)
    
    elif args.command == "run":
        dialogue = tester.load_dialogue(args.dialogue)
        if dialogue:
            # Create test_results directory
            results_dir = Path(config.paths.persistent_dir) / "test_results"
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Run the dialogue
            save_results = not args.no_save
            results = tester.run_dialogue(dialogue, args.pause, save_results)
            
            # Print where results are saved if saving is enabled
            if save_results:
                print(f"\nResults saved to: {config.paths.persistent_dir}/test_results/")
        else:
            print(f"Error: Could not load dialogue from {args.dialogue}")
    
    else:
        # Default to listing dialogues
        parser.print_help()


if __name__ == "__main__":
    main()