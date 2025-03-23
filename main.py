"""
Main entry point for the AI agent system.

This module handles central control flow, system initialization,
and provides a clean, readable entry point with minimal complexity.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from config import config
from errors import handle_error, AgentError
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from tools.async_manager import AsyncTaskManager
from tools.async_tools import ScheduleAsyncTaskTool, CheckAsyncTaskTool
from conversation import Conversation
from crud import FileOperations


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='AI Agent System')
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--conversation', '-id',
        type=str,
        help='Conversation ID to load'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        '--stream',
        action='store_true',
        dest='stream_mode',
        help='Enable streaming mode for responses'
    )
    stream_group.add_argument(
        '--no-stream',
        action='store_false',
        dest='stream_mode',
        help='Disable streaming mode for responses'
    )
    # Default is None so we know if the user specified an option or not
    parser.set_defaults(stream_mode=None)
    return parser.parse_args()


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Optional override for the log level
    """
    level = log_level or config.get('log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_system(args) -> Dict[str, Any]:
    """
    Initialize the AI agent system.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of system components
    """
    # Initialize logger
    logger = logging.getLogger("main")
    logger.info("Initializing AI agent system")
    
    # Initialize components
    try:
        # Override config with command line arguments
        if args.config:
            from config import Config
            config_instance = Config(args.config)
        
        # Set up logging level
        if args.log_level:
            setup_logging(args.log_level)
        
        # Initialize data directory
        data_dir = Path(config.get('data_dir', 'data'))
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize file operations
        file_ops = FileOperations(data_dir)
        
        # Initialize LLM bridge
        llm_bridge = LLMBridge()
        
        # Initialize tool repository
        tool_repo = ToolRepository()
        
        # Task notification queue
        task_notifications = []
        
        # Initialize async task manager
        async_task_manager = AsyncTaskManager(tool_repo=tool_repo, llm_bridge=llm_bridge)
        
        # Register async tools with the task manager
        tool_repo.register_tool(ScheduleAsyncTaskTool(task_manager=async_task_manager))
        tool_repo.register_tool(CheckAsyncTaskTool(task_manager=async_task_manager))
        
        # Setup notification callback for async tasks
        def notify_task_completion(task):
            # This will be called when an async task completes if notify_on_completion is True
            logging.info(f"Task completed: {task.task_id} - {task.description}")
            # Add to notification queue if notify_on_completion is True
            if task.notify_on_completion:
                task_notifications.append(f"Task completed: {task.description}")
        
        async_task_manager.set_notification_callback(notify_task_completion)
        
        # Initialize or load conversation
        if args.conversation:
            try:
                conversation_data = file_ops.read(f"conversation_{args.conversation}")
                conversation = Conversation.from_dict(
                    conversation_data,
                    llm_bridge=llm_bridge,
                    tool_repo=tool_repo
                )
                logger.info(f"Loaded conversation: {conversation.conversation_id}")
            except Exception as e:
                logger.error(f"Failed to load conversation: {e}")
                conversation = Conversation(
                    conversation_id=args.conversation,
                    llm_bridge=llm_bridge,
                    tool_repo=tool_repo
                )
        else:
            conversation = Conversation(
                llm_bridge=llm_bridge,
                tool_repo=tool_repo
            )
        
        logger.info(f"System initialized with conversation ID: {conversation.conversation_id}")
        
        # Return system components
        return {
            'file_ops': file_ops,
            'llm_bridge': llm_bridge,
            'tool_repo': tool_repo,
            'async_task_manager': async_task_manager,
            'conversation': conversation,
            'task_notifications': task_notifications
        }
    
    except AgentError as e:
        logger.error(f"Initialization error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


def save_conversation(file_ops: FileOperations, conversation: Conversation) -> None:
    """
    Save the current conversation to a file.
    
    Args:
        file_ops: File operations handler
        conversation: Conversation to save
    """
    try:
        conversation_data = conversation.to_dict()
        file_ops.write(f"conversation_{conversation.conversation_id}", conversation_data)
        logging.info(f"Saved conversation: {conversation.conversation_id}")
    except Exception as e:
        logging.error(f"Failed to save conversation: {e}")
        print(f"Error saving conversation: {e}")


def interactive_mode(system: Dict[str, Any], stream_mode: bool = False) -> None:
    """
    Run the system in interactive mode.
    
    Args:
        system: Dictionary of system components
        stream_mode: Whether to enable streaming responses
    """
    conversation = system['conversation']
    file_ops = system['file_ops']
    async_task_manager = system['async_task_manager']
    task_notifications = system['task_notifications']
    
    print("\nAI Agent System - Interactive Mode")
    print(f"Conversation ID: {conversation.conversation_id}")
    print(f"Streaming mode: {'Enabled' if stream_mode else 'Disabled'}")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'save' to save the conversation")
    print("Type 'clear' to clear the conversation history")
    print("-" * 50)
    
    def print_token(token: str):
        """Print token by token for streaming effect."""
        print(token, end="", flush=True)
    
    while True:
        try:
            # Display any pending task notifications
            if task_notifications:
                print("\n" + "-" * 30)
                for notification in task_notifications:
                    print(notification)
                print("-" * 30)
                # Clear notifications after displaying them
                task_notifications.clear()
            
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for commands
            if user_input.lower() in ['exit', 'quit']:
                save_conversation(file_ops, conversation)
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'save':
                save_conversation(file_ops, conversation)
                print("Conversation saved.")
                continue
            
            elif user_input.lower() == 'clear':
                conversation.clear_history()
                print("Conversation history cleared.")
                continue
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            if stream_mode:
                # Streaming mode - tokens are printed via callback
                conversation.generate_response(
                    user_input,
                    stream=True, 
                    stream_callback=print_token
                )
                print()  # Add newline after streaming completes
            else:
                # Standard mode - print full response at once
                response = conversation.generate_response(user_input)
                print(response)
            
        except KeyboardInterrupt:
            print("\nInterrupted. Saving conversation...")
            save_conversation(file_ops, conversation)
            print("Goodbye!")
            break
        
        except Exception as e:
            error_message = handle_error(e)
            print(f"\nError: {error_message}")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Initialize the system
    system = initialize_system(args)
    
    # Determine streaming mode: use command-line flag if provided, otherwise use config
    stream_mode = args.stream_mode if args.stream_mode is not None else config.get("streaming", False)
    
    # Run in interactive mode with appropriate streaming setting
    try:
        interactive_mode(system, stream_mode=stream_mode)
    except Exception as e:
        logging.exception("Unexpected error in main loop")
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Display any pending notifications before exit
        task_notifications = system.get('task_notifications', [])
        if task_notifications:
            print("\n" + "-" * 30)
            print("Pending task notifications:")
            for notification in task_notifications:
                print(notification)
            print("-" * 30)
            
        # Ensure async task manager is properly shut down
        if 'async_task_manager' in system:
            system['async_task_manager'].shutdown()
            logging.info("Async task manager has been shut down")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())