"""
Main entry point for the AI agent system.

This module handles central control flow, system initialization,
and provides a clean, readable entry point with minimal complexity.
"""
# Set environment variable before importing any libraries that might use tokenizers
# This prevents deadlocks when the process is forked
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import uuid
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List

from config import config
from errors import handle_error, AgentError, error_context, ErrorCode, FileOperationError
from api.llm_provider import LLMProvider
from tools.repo import ToolRepository
from tools.tool_feedback import save_tool_feedback
from conversation import Conversation
from onload_checker import OnLoadChecker, add_stimuli_to_conversation
from working_memory import WorkingMemory, TimeManager, SystemStatusManager, ReminderManager
from lt_memory.integration import initialize_lt_memory, check_lt_memory_requirements


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
    
    # Reduced opacity (70%) theme
    COLORS = {
        'DEBUG': '\033[2;3;36m',    # Dim Italic Cyan
        'INFO': '\033[2;32m',       # Dim Green
        'WARNING': '\033[38;5;208m',# Bright Orange
        'ERROR': '\033[31m',      # Dim Red
        'CRITICAL': '\033[2;37;41m', # Dim White on Red Background
        'RESET': '\033[0m'          # Reset
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
    
    # Use stderr for logs to prevent mixing with conversation output #ANNOTATION In addition to stderr.log we should build functionality that allows us to see logging happening in realtime. I struggle to see what is happening on the server when testing the app.
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
    
    # Initialize terminal context if using basic auth
    import auth
    if hasattr(auth, 'init_terminal_context'):
        auth.init_terminal_context()

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

        # Initialize unified LLM provider
        llm_provider = LLMProvider()
        logger.info(
            f"Initialized LLM provider: type={config.api.provider}, "
            f"endpoint={config.api.api_endpoint}, model={config.api.model}"
        )

        # Initialize tool repository
        tool_repo = ToolRepository()
        
        # Discover and register tools
        tool_repo.discover_tools()
        
        # Enable tools from config
        tool_repo.enable_tools_from_config()
        
        # Import necessary modules
        from tool_relevance_engine import ToolRelevanceEngine
        from tools.workflows.workflow_manager import WorkflowManager
        from api.embeddings_provider import EmbeddingsProvider
        
        # Create shared embeddings provider for tool classification
        # Use local BGE for tool classification (fast, efficient)
        logger.info("Initializing shared embeddings provider for tool classification")
        shared_embeddings = EmbeddingsProvider(
            provider_type="local",
            enable_reranker=False  # No reranker needed for tool classification
        )
        logger.info("Shared embeddings provider initialized successfully")
        
        # Initialize the ToolRelevanceEngine with shared embeddings
        tool_relevance_engine = ToolRelevanceEngine(tool_repo, shared_embeddings)
        logger.info("Initialized ToolRelevanceEngine for dynamic tool management")
        
        # Initialize working memory system first
        working_memory = WorkingMemory()
        logger.info("Initialized WorkingMemory for centralized system prompt content")

        # Initialize the WorkflowManager with shared embeddings, LLM bridge, and working memory
        workflow_manager = WorkflowManager(
            tool_repo,
            model=shared_embeddings,
            llm_provider=llm_provider,
            working_memory=working_memory
        )
        logger.info("Initialized WorkflowManager with working memory integration")

        # Initialize TimeManager to handle datetime information (standalone trinket)
        time_manager = TimeManager(working_memory)
        logger.info("Initialized TimeManager for datetime information")


        # Initialize SystemStatusManager to handle system status (standalone trinket)
        system_status_manager = SystemStatusManager(working_memory)
        logger.info("Initialized SystemStatusManager for system status")

        # Initialize ReminderManager to handle reminders in system prompt (standalone trinket)
        reminder_manager = ReminderManager(working_memory)
        logger.info("Initialized ReminderManager for reminder information")

        # Register managers with working memory
        working_memory.register_manager(workflow_manager)
        logger.info("Registered WorkflowManager with working memory")

        working_memory.register_manager(tool_repo)
        logger.info("Registered ToolRepository with working memory")

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
                    file_path = conversation_dir / f"conversation_{args.conversation}.json"
                    
                    if not file_path.exists():
                        raise FileOperationError(
                            f"File not found: {file_path}",
                            ErrorCode.FILE_NOT_FOUND
                        )
                    
                    try:
                        with open(file_path, 'r') as f:
                            conversation_data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise FileOperationError(
                            f"Invalid JSON in file {file_path}: {e}",
                            ErrorCode.INVALID_JSON
                        )
                    except Exception as e:
                        raise FileOperationError(
                            f"Error reading file {file_path}: {e}",
                            ErrorCode.FILE_READ_ERROR
                        )
                    conversation = Conversation.from_dict(
                        conversation_data,
                        llm_provider=llm_provider,
                        tool_repo=tool_repo,
                        tool_relevance_engine=tool_relevance_engine,
                        workflow_manager=workflow_manager,
                        working_memory=working_memory
                    )
                    logger.info(f"Loaded conversation: {conversation.conversation_id}")
                except Exception as e:
                    logger.warning(f"Failed to load conversation, creating new one: {e}")
                    # Get default system prompt
                    system_prompt = config.get_system_prompt("main_system_prompt")
                    conversation = Conversation(
                        conversation_id=args.conversation,
                        system_prompt=system_prompt,
                        llm_provider=llm_provider,
                        tool_repo=tool_repo,
                        tool_relevance_engine=tool_relevance_engine,
                        workflow_manager=workflow_manager,
                        working_memory=working_memory
                    )
        else:
            # Generate a unique conversation ID and get default system prompt
            conversation_id = str(uuid.uuid4())
            system_prompt = config.get_system_prompt("main_system_prompt")
            conversation = Conversation(
                conversation_id=conversation_id,
                system_prompt=system_prompt,
                llm_provider=llm_provider,
                tool_repo=tool_repo,
                tool_relevance_engine=tool_relevance_engine,
                workflow_manager=workflow_manager,
                working_memory=working_memory
            )
            

        # Run on-load checks
        onload_checker = OnLoadChecker()
        logger.info("Running on-load checks...")
        onload_stimuli = onload_checker.run_all_checks(conversation)
        
        # Add any notifications from checks to the conversation
        if onload_stimuli:
            logger.info(f"Found {len(onload_stimuli)} notification(s) from on-load checks")
            add_stimuli_to_conversation(onload_stimuli, conversation)
        
        
        logger.info(f"System initialized with conversation ID: {conversation.conversation_id}")
        
        # Initialize LT_Memory - REQUIRED for operation
        logger.info("Checking LT_Memory requirements...")
        lt_memory_status = check_lt_memory_requirements()
        if not all(lt_memory_status.values()):
            missing = [k for k, v in lt_memory_status.items() if not v]
            raise AgentError(
                f"LT_Memory requirements not met: {missing}. "
                f"Please ensure PostgreSQL with pgvector is installed and "
                f"LT_MEMORY_DATABASE_URL environment variable is set.",
                ErrorCode.MISSING_ENV_VAR
            )
        
        logger.info("Initializing LT_Memory system...")
        lt_memory_components = initialize_lt_memory(
            config, 
            working_memory, 
            tool_repo,
            llm_provider
        )
        
        # Verify initialization was successful
        if not lt_memory_components.get("manager"):
            raise AgentError(
                "LT_Memory initialization failed - memory manager not created",
                ErrorCode.INITIALIZATION_FAILED
            )
        
        lt_memory = lt_memory_components
        logger.info("LT_Memory system initialized successfully")
        
        # Initialize memory consolidation service
        from memory_consolidation_service import MemoryConsolidationService
        memory_consolidation_service = MemoryConsolidationService(
            working_memory=working_memory,
            memory_manager=lt_memory_components["manager"],
            llm_provider=llm_provider
        )
        
        # Start the service (integration with auth system will be handled by auth lifecycle)
        memory_consolidation_service.start()
        logger.info("Memory consolidation service started")
        
        # Initialize ConversationArchiveManager trinket
        from working_memory import ConversationArchiveManager, ProactiveMemoryTrinket
        conversation_archive_manager = ConversationArchiveManager(
            working_memory=working_memory,
            conversation_timeline_manager=lt_memory_components["conversation_timeline_manager"]
        )
        logger.info("Initialized ConversationArchiveManager trinket")
        
        # Initialize ProactiveMemoryTrinket
        proactive_memory_trinket = ProactiveMemoryTrinket(
            working_memory=working_memory,
            memory_manager=lt_memory_components["manager"],
            conversation=conversation
        )
        logger.info("Initialized ProactiveMemoryTrinket")
        
        # Now that we have a conversation, add token tracking to the LLM bridge
        def token_tracking_decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                response = func(*args, **kwargs)
                
                # For non-streaming response with usage info
                if hasattr(response, 'usage') and isinstance(response.usage, Usage):
                    if hasattr(conversation, 'tokens_in') and hasattr(conversation, 'tokens_out'):
                        conversation.tokens_in += response.usage.input_tokens
                        conversation.tokens_out += response.usage.output_tokens
                        logger.debug(f"Tracked tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
                
                # For streaming response, we need to get usage from the final_message
                elif kwargs.get('stream') and hasattr(response, 'get_final_message'):
                    try:
                        final_message = response.get_final_message()
                        if hasattr(final_message, 'usage') and isinstance(final_message.usage, Usage):
                            conversation.tokens_in += final_message.usage.input_tokens
                            conversation.tokens_out += final_message.usage.output_tokens
                            logger.debug(f"Tracked tokens (stream): {final_message.usage.input_tokens} in, {final_message.usage.output_tokens} out")
                    except Exception as e:
                        logger.debug(f"Failed to get usage from stream: {e}")
                
                return response
            return wrapper
            
        # Patch the LLM bridge's generate_response method
        original_generate_response = llm_provider.generate_response #ANNOTATION what is 'original_generate_response? Is this a backwards compatibility thing?'
        llm_provider.generate_response = token_tracking_decorator(original_generate_response)

        # Return system components
        return {
            'llm_provider': llm_provider,
            'tool_repo': tool_repo,
            'conversation': conversation,
            'tool_relevance_engine': tool_relevance_engine,
            'workflow_manager': workflow_manager,
            'onload_checker': onload_checker,
            'automation_engine': automation_engine,
            'working_memory': working_memory,
            'time_manager': time_manager,
            'system_status_manager': system_status_manager,
            'reminder_manager': reminder_manager,
            'conversation_archive_manager': conversation_archive_manager,
            'lt_memory': lt_memory,
            'memory_consolidation_service': memory_consolidation_service
        }


def save_conversation(conversation: Conversation) -> None:
    """
    Save the current conversation to a file.

    Args:
        conversation: Conversation to save
    """
    with error_context(
        component_name="System",
        operation="saving conversation",
        error_class=AgentError,
        logger=logging.getLogger("main")
    ):
        conversation_data = conversation.to_dict()
        
        # Create conversation history directory
        conversation_dir = Path(config.paths.conversation_history_dir)
        os.makedirs(conversation_dir, exist_ok=True)
        
        # Define the file path for the conversation
        file_path = conversation_dir / f"conversation_{conversation.conversation_id}.json"
        
        # Write the conversation data to the file using serialization module
        try:
            from serialization import to_json
            with open(file_path, 'w') as f:
                # Use the custom to_json function that handles datetime objects
                json_data = to_json(conversation_data, indent=config.system.json_indent)
                f.write(json_data)
            logging.info(f"Saved conversation: {conversation.conversation_id}")
        except Exception as e:
            raise FileOperationError(
                f"Error writing to file {file_path}: {e}",
                ErrorCode.FILE_WRITE_ERROR
            )


def interactive_mode(system: Dict[str, Any], stream_mode: bool = False) -> None:
    """
    Run the system in interactive mode.

    Args:
        system: Dictionary of system components
        stream_mode: Whether to enable streaming responses
    """
    conversation = system['conversation']

    print("\nAI Agent System - Interactive Mode")
    print(f"Conversation ID: {conversation.conversation_id}")
    print(f"Streaming mode: {'Enabled' if stream_mode else 'Disabled'}")
    print("Type '/exit' to end the session")
    print("Type '/save' to save the conversation")
    print("Type '/clear' to clear the conversation history")
    print("Type '/tokens' to show token usage counts")
    print("Type '/toolfeedback [feedback]' to save feedback about tool activation")
    print("-" * 50)
    
    # Display any initial messages (like reminders) that were added during initialization
    for message in conversation.messages:
        if message.role == "assistant" and message.metadata.get("is_notification"):
            print(f"\nAssistant: {message.content}")
    
    # In the new unified automation system, notifications are handled differently
    # This section will be enhanced in future updates as needed

    def print_token(token: str):
        """Print token by token for streaming effect."""
        print(token, end="", flush=True)

    while True:
        try:

            # Get user input
            user_input = input("\nUser: ")

            # Check for commands
            if user_input.lower() in ['/exit']:
                save_conversation(conversation)
                print("Goodbye!")
                break

            elif user_input.lower() == '/save':
                save_conversation(conversation)
                print("Conversation saved.")
                continue

            elif user_input.lower() == '/clear':
                conversation.clear_history()
                print("Conversation history cleared.")
                continue
                
                
            elif user_input.lower() == '/tokens':
                tokens_in = getattr(conversation, 'tokens_in', 0)
                tokens_out = getattr(conversation, 'tokens_out', 0)
                print(f"Tokens in: {tokens_in}")
                print(f"Tokens out: {tokens_out}")
                print(f"Total tokens: {tokens_in + tokens_out}")
                continue
                
            elif user_input.lower().startswith('/toolfeedback'):
                feedback_text = user_input[len('/toolfeedback'):].strip()
                if not feedback_text:
                    print("Please provide feedback text after /toolfeedback")
                    continue
                
                # Save tool feedback and get analysis
                print("Saving feedback and generating analysis...")
                success, analysis = save_tool_feedback(system, feedback_text, conversation)
                
                if success:
                    print("Tool feedback saved successfully.")
                    
                    # Display analysis if available
                    if analysis:
                        print(f"Analysis: {analysis}")
                    else:
                        print("No analysis available.")
                else:
                    print("Error saving tool feedback.")
                
                continue
                
            elif user_input.lower() == '/memory':
                # Display current working memory contents
                working_memory = system['working_memory']
                memory_items = working_memory._memory_items
                
                if not memory_items:
                    print("Working memory is empty.")
                else:
                    print("=== Current Working Memory ===")
                    
                    # Group items by category for better organization
                    categories = {}
                    for item_id, item in memory_items.items():
                        category = item['category']
                        if category not in categories:
                            categories[category] = []
                        categories[category].append({
                            'id': item_id,
                            'content': item['content']
                        })
                    
                    # Display each category
                    for category, items in categories.items():
                        print(f"\n--- {category.upper()} ---")
                        for item in items:
                            print(f"[{item['id'][:8]}] {item['content']}")
                    
                    print(f"\nTotal items: {len(memory_items)}")
                
                continue

            # Generate response
            print("\nAssistant: ", end="", flush=True)

            # Update dynamic information in working memory before each response
            system['time_manager'].update_datetime_info()
            system['reminder_manager'].update_reminder_info()

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
                    logging.info(f"Main display path: response={response}")
                    print(response)

        except KeyboardInterrupt:
            print("\nInterrupted. Saving conversation...")
            save_conversation(conversation)
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
        # Stop all automation systems
        automation_controller.stop_systems()
        
        # Final cleanup
        logging.info("Session ended")

    return 0


if __name__ == "__main__":
    sys.exit(main())
