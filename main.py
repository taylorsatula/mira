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
from errors import handle_error, AgentError, error_context, ErrorCode
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
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
    Set up logging configuration with colored output.

    Args:
        log_level: Optional override for the log level
    """
    level = log_level or config.system.log_level
    
    # High-viz theme with bold and italics
    COLORS = {
        'DEBUG': '\033[3;36m',    # Italic Cyan
        'INFO': '\033[1;32m',     # Bold Green
        'WARNING': '\033[1;33m',  # Bold Yellow
        'ERROR': '\033[1;31m',    # Bold Red
        'CRITICAL': '\033[1;37;41m', # Bold White on Red Background
        'RESET': '\033[0m'        # Reset
    }
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with color for macOS bash."""
        
        def format(self, record):
            # Add color directly to the entire log message for maximum compatibility
            levelname = record.levelname
            color = COLORS.get(levelname, '')
            reset = COLORS['RESET']
            
            # Format the record normally first
            message = super().format(record)
            
            # Then wrap the entire message with color codes
            return f"{color}{message}{reset}"
    
    # Configure root logger with a fresh handler
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    
    # Remove any existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Create simple formatter without timestamps
    formatter = logging.Formatter('%(levelname)s [%(name)s] %(message)s')
    
    # Create colored formatter with better visual separation
    colored_formatter = ColoredFormatter('%(levelname)s │ %(name)s │ %(message)s')
    
    # Use stderr for logs to prevent mixing with conversation output
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(colored_formatter)
    root.addHandler(handler)
    
    # Disable noisy HTTP library loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Add a print separator to cleanly show where log setup ends
    print("\nLogging initialized. Starting conversation...\n")


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

    # Use centralized error context for system initialization
    with error_context(
        component_name="System",
        operation="initialization",
        error_class=AgentError,
        error_code=ErrorCode.UNKNOWN_ERROR,
        logger=logger
    ):
        # Override config with command line arguments
        if args.config:
            pass  # Removed unused config_instance

        # Set up logging level
        if args.log_level:
            setup_logging(args.log_level)

        # Initialize data directory
        data_dir = Path(config.paths.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # Initialize file operations
        file_ops = FileOperations(data_dir)

        # Initialize LLM bridge
        llm_bridge = LLMBridge()

        # Initialize tool repository
        tool_repo = ToolRepository()
        
        # Discover and register tools
        tool_repo.discover_tools()
        
        # Enable tools from config
        tool_repo.enable_tools_from_config()
        
        # Initialize or load conversation
        if args.conversation:
            # Use error context specifically for loading the conversation
            with error_context(
                component_name="System", 
                operation=f"loading conversation {args.conversation}",
                error_class=AgentError,
                logger=logger
            ):
                try:
                    # Load from the conversation history directory
                    conversation_dir = Path(config.paths.conversation_history_dir)
                    os.makedirs(conversation_dir, exist_ok=True)
                    conversation_ops = FileOperations(conversation_dir)
                    conversation_data = conversation_ops.read(f"conversation_{args.conversation}")
                    conversation = Conversation.from_dict(
                        conversation_data,
                        llm_bridge=llm_bridge,
                        tool_repo=tool_repo
                    )
                    logger.info(f"Loaded conversation: {conversation.conversation_id}")
                except Exception as e:
                    logger.warning(f"Failed to load conversation, creating new one: {e}")
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
            'conversation': conversation
        }


def save_conversation(file_ops: FileOperations, conversation: Conversation) -> None:
    """
    Save the current conversation to a file.

    Args:
        file_ops: File operations handler
        conversation: Conversation to save
    """
    with error_context(
        component_name="System",
        operation="saving conversation",
        error_class=AgentError,
        logger=logging.getLogger("main")
    ):
        conversation_data = conversation.to_dict()
        # Create conversation history directory FileOperations
        conversation_dir = Path(config.paths.conversation_history_dir)
        os.makedirs(conversation_dir, exist_ok=True)
        conversation_ops = FileOperations(conversation_dir)
        conversation_ops.write(f"conversation_{conversation.conversation_id}", conversation_data)
        logging.info(f"Saved conversation: {conversation.conversation_id}")


def interactive_mode(system: Dict[str, Any], stream_mode: bool = False) -> None:
    """
    Run the system in interactive mode.

    Args:
        system: Dictionary of system components
        stream_mode: Whether to enable streaming responses
    """
    conversation = system['conversation']
    file_ops = system['file_ops']

    print("\nAI Agent System - Interactive Mode")
    print(f"Conversation ID: {conversation.conversation_id}")
    print(f"Streaming mode: {'Enabled' if stream_mode else 'Disabled'}")
    print("Type '/exit' to end the session")
    print("Type '/save' to save the conversation")
    print("Type '/clear' to clear the conversation history")
    print("-" * 50)

    def print_token(token: str):
        """Print token by token for streaming effect."""
        print(token, end="", flush=True)

    while True:
        try:

            # Get user input
            user_input = input("\nUser: ")

            # Check for commands
            if user_input.lower() in ['/exit']:
                save_conversation(file_ops, conversation)
                print("Goodbye!")
                break

            elif user_input.lower() == '/save':
                save_conversation(file_ops, conversation)
                print("Conversation saved.")
                continue

            elif user_input.lower() == '/clear':
                conversation.clear_history()
                print("Conversation history cleared.")
                continue

            # Generate response
            print("\nAssistant: ", end="", flush=True)

            # Use error context for response generation
            with error_context(
                component_name="Interactive",
                operation="generating response",
                error_class=AgentError,
                logger=logging.getLogger("interactive")
            ):
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
            # Handle any errors that weren't caught by the error context
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
    stream_mode = args.stream_mode if args.stream_mode is not None else config.system.streaming

    # Run in interactive mode with appropriate streaming setting
    try:
        with error_context(
            component_name="System",
            operation="main loop",
            error_class=AgentError,
            logger=logging.getLogger("main")
        ):
            interactive_mode(system, stream_mode=stream_mode)
    finally:
        # Final cleanup
        logging.info("Session ended")

    return 0


if __name__ == "__main__":
    sys.exit(main())
