"""
Conversation management for the AI agent system.

This module handles conversation turns, context tracking,
tool result integration, and conversation history management.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""
import json
import logging
import os
import time
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from tool_relevance_engine import ToolRelevanceEngine
    from tools.workflows.workflow_manager import WorkflowManager
from zoneinfo import ZoneInfo

# Import timezone utilities for UTC-everywhere approach
from utils.timezone_utils import (
    utc_now, ensure_utc, convert_from_utc, format_datetime,
    parse_utc_time_string, get_default_timezone
)

from errors import ConversationError, ErrorCode, error_context
from config import config
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from utils.tag_parser import parser as tag_parser


@dataclass
class Message:
    """
    Representation of a message in a conversation.
    
    Attributes:
        role: The role of the message sender (must be 'user' or 'assistant')
        content: The content of the message (string or list of content blocks)
        id: Unique identifier for the message
        created_at: UTC datetime when the message was created
        metadata: Additional metadata for the message
    """
    role: str
    content: Union[str, List[Dict[str, Any]]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary representation.
        
        Returns:
            Dictionary representation of the message
        """
        # Handle complex content objects like TextBlock by converting to string
        content = self.content
        if not isinstance(content, (str, list, dict)) and hasattr(content, '__dict__'):
            # Convert non-serializable objects to string representation
            content = str(content)
        elif isinstance(content, list) and any(not isinstance(item, (str, int, float, bool, dict, list, type(None))) for item in content):
            # Handle list of complex objects
            content = [str(item) if not isinstance(item, (str, int, float, bool, dict, list, type(None))) else item for item in content]
        
        # Format created_at as ISO string if it's a datetime, or pass through if it's still a float
        created_at_formatted = self.created_at
        if isinstance(self.created_at, datetime.datetime):
            created_at_formatted = self.created_at.isoformat()
        
        return {
            "role": self.role,
            "content": content,
            "id": self.id,
            "created_at": created_at_formatted,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from a dictionary representation.
        
        Args:
            data: Dictionary representation of the message
            
        Returns:
            Message object
        """
        # Default to current UTC time if created_at is missing
        if "created_at" not in data:
            created_at = utc_now()
        else:
            # Parse ISO string into datetime - use timezone_utils for consistent handling
            created_at = parse_utc_time_string(data["created_at"])
            
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid.uuid4())),
            created_at=created_at,
            metadata=data.get("metadata", {})
        )


class Conversation:
    """
    Manager for conversation interactions.
    
    Handles conversation turns, context tracking, tool result
    integration, and conversation history management.
    """
    
    def __init__(
        self,
        conversation_id: str,
        system_prompt: str,
        llm_bridge: LLMBridge,
        tool_repo: ToolRepository,
        tool_relevance_engine: 'ToolRelevanceEngine',
        workflow_manager: 'WorkflowManager',
        working_memory: 'WorkingMemory'
    ):
        """
        Initialize a new conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            system_prompt: System prompt for the conversation
            llm_bridge: LLM bridge instance
            tool_repo: Tool repository instance
            tool_relevance_engine: Tool relevance engine instance
            workflow_manager: Workflow manager instance
            working_memory: Working memory instance for dynamic prompt content
        """
        # Set up logging
        self.logger = logging.getLogger("conversation")
        
        # Initialize conversation ID
        self.conversation_id = conversation_id
        self.logger.debug(f"Created conversation with ID: {self.conversation_id}")
        
        # Set up conversation history
        self.messages: List[Message] = []
        
        # Set up metadata storage (including location data)
        self.metadata: Dict[str, Any] = {}
        
        # Set up conversation components
        self.llm_bridge = llm_bridge
        self.tool_repo = tool_repo
        self.tool_relevance_engine = tool_relevance_engine
        self.workflow_manager = workflow_manager
        self.working_memory = working_memory
        
        # Set up conversation config
        self.max_history = config.conversation.max_history
        self.max_context_tokens = config.conversation.max_context_tokens
        
        # Store system prompt as a property (not as a message)
        self.system_prompt = system_prompt
            
        # Initialize error tracking
        self.last_error = None
        
        # Initialize workflow tracking
        self._detected_workflow_id = None
        
        # Initialize token tracking
        self.tokens_in = 0
        self.tokens_out = 0
        
        # Initialize tag handling state
        self._tried_loading_all_tools = False
        self._previously_enabled_tools = set()
    
    def add_message(self, role: str, content: Union[str, List[Dict[str, Any]]], metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender (must be 'user' or 'assistant')
            content: The content of the message (string or list of content blocks)
            metadata: Optional metadata for the message
            
        Returns:
            The newly created message
            
        Raises:
            ValueError: If role is not 'user' or 'assistant'
        """
        # Validate role is either 'user' or 'assistant'
        if role not in ["user", "assistant"]:
            self.logger.warning(f"Invalid role: {role}. Converting to 'user' role.")
            role = "user"  # For backwards compatibility, convert invalid roles to 'user'
            
        # Create a new message
        message = Message(role=role, content=content, metadata=metadata or {})
        
        # Add the message to the conversation
        self.messages.append(message)
        
        # Prune history if needed #ANNOTATION We should keep a complete conversation history stored somewhere outside the context window so that when the conversation is complete and saves we can go through it.
        if self.max_history and len(self.messages) > self.max_history * 2:
            # Keep only the most recent messages
            self.messages = self.messages[-self.max_history*2:]
            self.logger.debug(f"Pruned conversation history to {len(self.messages)} messages")
        
        return message
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """
        Get the formatted messages for Anthropic API consumption.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
            following Anthropic's API format requirements.
        """
        # Format messages for the API
        formatted_messages = []
        
        for message in self.messages:
            # This should never happen since add_message already converts invalid roles to 'user'
            if message.role not in ["user", "assistant"]:
                self.logger.error(f"Inconsistent state: found message with invalid role: {message.role} after role validation in add_message")
                # Convert to 'user' role for backwards compatibility
                message.role = "user"
                
            # Handle assistant messages with tool_use blocks
            if message.role == "assistant" and message.metadata.get("has_tool_calls", False):
                # For messages with tool_use blocks, preserve the original content structure
                formatted_messages.append({
                    "role": "assistant",
                    "content": message.content  # This preserves the tool_use blocks
                })
            # Handle user messages with tool results (content is already a list of content blocks)
            elif message.role == "user" and isinstance(message.content, list):
                formatted_messages.append({
                    "role": "user",
                    "content": message.content
                })
            else:
                # For standard text messages
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        return formatted_messages
    
    def extract_topic_changed_tag(self, response_text: str) -> bool:
        """
        Extract topic change status from response text.
        
        Looks for the <topic_changed=true/> or <topic_changed=false/> tag in the response.
        
        Args:
            response_text: The text response from the LLM
            
        Returns:
            Boolean indicating whether the topic has changed (True) or not (False)
        """
        # Use the tag parser module
        return tag_parser.extract_topic_changed(response_text)
        
    def _load_all_tools(self) -> None:
        """
        Enable all available tools in the tool repository.
        
        This method is called when the <need_tool /> marker is detected,
        to make all possible tools available for the next response.
        """
        self.logger.info("Loading all available tools due to <need_tool /> marker")
        if self.tool_repo:
            # Store currently enabled tools to restore later if needed
            self._previously_enabled_tools = set(self.tool_repo.get_enabled_tools())
            
            # Enable all tools
            self.tool_repo.enable_all_tools()
            
            # Log the newly enabled tools
            all_tools = self.tool_repo.get_enabled_tools()
            self.logger.info(f"Enabled all {len(all_tools)} tools: {', '.join(all_tools)}")
    
    def generate_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = config.system.streaming,
        stream_callback: Optional[callable] = None,
        max_tool_iterations: int = config.conversation.max_tool_iterations  # Limit iterations to prevent infinite loops
    ) -> Union[str, None]:
        """
        Generate a response to user input.
        
        Args:
            user_input: User input text
            temperature: Optional temperature parameter
            max_tokens: Optional maximum tokens for the response
            stream: Whether to stream the response (default: from config.system.streaming)
            stream_callback: Optional callback function for processing streamed tokens
            max_tool_iterations: Maximum number of tool call iterations
            
        Returns:
            If stream=False: Assistant's response text
            If stream=True with callback: None (results sent to callback)
            
        Raises:
            ConversationError: If response generation fails
        """
        # Check for empty input
        if not user_input or user_input.strip() == "":
            error_msg = "Empty message content is not allowed"
            self.logger.warning(error_msg)
            
            # Check if we're in the middle of a tool execution chain
            is_tool_execution_active = False
            if self.messages and len(self.messages) >= 2:
                last_msg = self.messages[-1]
                # Check if last message was a tool result or has tool calls
                if ((last_msg.role == "user" and last_msg.metadata.get("is_tool_result", False)) or
                    (last_msg.role == "assistant" and last_msg.metadata.get("has_tool_calls", False))):
                    is_tool_execution_active = True
                    self.logger.info("Blank message received during tool execution - suppressing error")
                    return "I'm still processing your previous request. Please wait a moment or send a new message."
            
            # If not in tool execution, raise the error as before
            raise ConversationError(error_msg, ErrorCode.INVALID_INPUT)
            
        # Add user message to conversation
        self.add_message("user", user_input)
        
        # Workflow detection and processing
        if self.workflow_manager and not self.workflow_manager.get_active_workflow():
            # Try to detect a workflow (only if no workflow is currently active)
            workflow_id, confidence = self.workflow_manager.detect_workflow(user_input)
            if workflow_id:
                self.logger.info(f"Detected potential workflow: {workflow_id} (confidence: {confidence:.2f})")
                # Store the detected workflow ID
                self._detected_workflow_id = workflow_id
                # Update the workflow hint in working memory
                self.workflow_manager.update_workflow_hint(workflow_id)
                # We'll continue with normal processing and let the LLM handle confirmation
        
        # If workflow is active, suspend tool relevance engine and enable workflow tools
        if self.workflow_manager and self.workflow_manager.get_active_workflow():
            if self.tool_relevance_engine:
                self.tool_relevance_engine.suspend()
                self.logger.info("Suspended tool relevance engine due to active workflow")
            # Tools are enabled by the workflow manager when the workflow advances
        # Otherwise use just-in-time tool enablement
        elif self.tool_relevance_engine:
            enabled_tools = self.tool_relevance_engine.manage_tool_relevance(user_input)
            if enabled_tools:
                self.logger.info(f"Enabled relevant tools for message: {', '.join(enabled_tools)}")
        
        # Use centralized error context manager for standardized error handling
        with error_context(
            component_name="Conversation",
            operation="generating response",
            error_class=ConversationError,
            error_code=ErrorCode.UNKNOWN_ERROR,  # Will be updated if context overflow is detected
            logger=self.logger
        ):
            # Initialize streaming handler if needed
            def _handle_streaming_response(text_chunk):
                # Call user-provided callback with each chunk
                if stream_callback:
                    stream_callback(text_chunk)
            
            # Keep track of the final response for return value
            final_response = None
            
            # Tool iteration counter
            tool_iterations = 0
            
            # Continue processing responses until no more tool calls are made #ANNOTATION Shouldn't we be using the official Anthropic stop_reason = tool_use? This edit may go past tool responses
            # or we reach the maximum number of iterations
            while tool_iterations < max_tool_iterations:
                # Get current messages for the API
                messages = self.get_formatted_messages()
                
                # Determine tools to load based on auto_discovery setting
                if self.tool_repo:
                    # Get the current available tools that are enabled
                    selected_tools = self.tool_repo.get_all_tool_definitions()
                    self.logger.info(f"Using {len(selected_tools)} tools for this interaction: {', '.join([t.get('name', 'unknown') for t in selected_tools])}")
                else:
                    selected_tools = None
                
                # Static content for caching (just the system prompt now)
                static_content = self.system_prompt

                # Get dynamic content from working memory
                dynamic_content = self.working_memory.get_prompt_content()
                
                # Working memory already contains all dynamic content
                
                # Working memory contains all dynamic guidance
                
                # Working memory contains all dynamic content including workflow hints
                
                # Define cache control for prompt caching
                cache_control = {"type": "ephemeral"}
                
                # Generate response (streaming or standard)
                if stream:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=static_content,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools,
                        stream=True,
                        callback=_handle_streaming_response,
                        cache_control=cache_control,
                        dynamic_content=dynamic_content
                    )
                    
                    # Update token counts if available in the response object
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        input_tokens = getattr(usage, 'input_tokens', 0)
                        output_tokens = getattr(usage, 'output_tokens', 0)
                        self.tokens_in += input_tokens
                        self.tokens_out += output_tokens

                        # Log token usage with timestamp for tracking
                        self.logger.info(f"Token usage at {format_datetime(utc_now(), 'iso')}: input={input_tokens}, output={output_tokens}")
                    
                    # For stream responses, get the final message
                    if hasattr(response, 'get_final_message'):
                        final_message = response.get_final_message()
                        response = final_message
                else:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=static_content,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools,
                        cache_control=cache_control,
                        dynamic_content=dynamic_content
                    )
                    
                    # Update token counts if available in the response object
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        input_tokens = getattr(usage, 'input_tokens', 0)
                        output_tokens = getattr(usage, 'output_tokens', 0)
                        self.tokens_in += input_tokens
                        self.tokens_out += output_tokens

                        # Log token usage with timestamp for tracking
                        self.logger.info(f"Token usage at {format_datetime(utc_now(), 'iso')}: input={input_tokens}, output={output_tokens}")
                
                
                # Extract text content for final return value
                assistant_response = self.llm_bridge.extract_text_content(response)
                final_response = assistant_response
                
                # Check for need_tool tag
                if tag_parser.has_need_tool(assistant_response):
                    self.logger.info("Detected <need_tool /> marker in assistant response")
                    
                    if not getattr(self, '_tried_loading_all_tools', False):
                        # Only try loading all tools once to prevent infinite loop
                        self._tried_loading_all_tools = True
                        
                        # Load all available tools
                        self._load_all_tools()
                        
                        # Don't add this message to the conversation
                        # Instead, retry with all tools enabled
                        continue
                    else:
                        # If we've already tried loading all tools, replace the <need_tool /> with a helpful message
                        self.logger.warning("Still received <need_tool /> after loading all tools")
                        modified_response = "I don't have the specific tool needed to handle this request. I can still try to help with the information and tools I do have available. Could you provide more details or ask your question in a different way?"
                        assistant_response = modified_response
                        final_response = modified_response
                
                # Check for workflow commands
                if self.workflow_manager:
                    # Check for workflow commands using the new format supporting more actions
                    command_found, command_type, command_params, command_data = self.workflow_manager.check_for_workflow_commands(assistant_response)
                    
                    if command_found:
                        if command_type == "start" and not self.workflow_manager.get_active_workflow():
                            workflow_id = command_params
                            try:
                                # Get the message that triggered the workflow to extract initial data
                                # We retrieve the most recent user message
                                triggering_message = None
                                for msg in reversed(self.messages):
                                    if msg.role == "user" and isinstance(msg.content, str):
                                        triggering_message = msg.content
                                        break
                                
                                # Start the workflow with potential data extraction from triggering message
                                self.workflow_manager.start_workflow(
                                    workflow_id,
                                    triggering_message=triggering_message, 
                                    llm_bridge=self.llm_bridge
                                )
                                self.logger.info(f"Started workflow: {workflow_id}")
                                # Suspend tool relevance engine
                                if self.tool_relevance_engine:
                                    self.tool_relevance_engine.suspend()
                            except Exception as e:
                                self.logger.error(f"Error starting workflow {workflow_id}: {e}")
                        
                        elif command_type == "complete_step" and self.workflow_manager.get_active_workflow():
                            step_id = command_params
                            try:
                                self.workflow_manager.complete_step(step_id, command_data)
                                self.logger.info(f"Completed workflow step: {step_id}")
                            except Exception as e:
                                self.logger.error(f"Error completing workflow step {step_id}: {e}")
                        
                        elif command_type == "skip_step" and self.workflow_manager.get_active_workflow():
                            step_id = command_params
                            try:
                                self.workflow_manager.skip_step(step_id)
                                self.logger.info(f"Skipped workflow step: {step_id}")
                            except Exception as e:
                                self.logger.error(f"Error skipping workflow step {step_id}: {e}")
                        
                        elif command_type == "revisit_step" and self.workflow_manager.get_active_workflow():
                            step_id = command_params
                            try:
                                self.workflow_manager.revisit_step(step_id)
                                self.logger.info(f"Revisiting workflow step: {step_id}")
                            except Exception as e:
                                self.logger.error(f"Error revisiting workflow step {step_id}: {e}")
                        
                        elif command_type == "complete" and self.workflow_manager.get_active_workflow():
                            try:
                                self.workflow_manager.complete_workflow()
                                self.logger.info("Completed workflow")
                                # Resume tool relevance engine
                                if self.tool_relevance_engine:
                                    self.tool_relevance_engine.resume()
                            except Exception as e:
                                self.logger.error(f"Error completing workflow: {e}")
                        
                        elif command_type == "cancel" and self.workflow_manager.get_active_workflow():
                            try:
                                self.workflow_manager.cancel_workflow()
                                self.logger.info("Cancelled workflow")
                                # Resume tool relevance engine
                                if self.tool_relevance_engine:
                                    self.tool_relevance_engine.resume()
                            except Exception as e:
                                self.logger.error(f"Error cancelling workflow: {e}")
                
                # Check for tool calls
                tool_calls = self.llm_bridge.extract_tool_calls(response)
                
                # Log tool calls for debugging
                if tool_calls:
                    self.logger.info(f"Found {len(tool_calls)} tool call(s):")
                    for i, tool_call in enumerate(tool_calls):
                        self.logger.info(f"  Tool call {i+1}: {tool_call['tool_name']}")
                else:
                    self.logger.info("No tool calls detected in response")
                
                # If no tool calls, add the response to conversation and break the loop
                if not tool_calls:
                    # Extract topic change tag from the response and update tool relevance engine
                    if self.tool_relevance_engine and not self.tool_relevance_engine.suspended:
                        topic_changed = self.extract_topic_changed_tag(assistant_response)
                        self.tool_relevance_engine.set_topic_changed(topic_changed)
                    
                    self.add_message("assistant", assistant_response)
                    break
                
                # Otherwise, process the tool calls and continue the loop
                tool_iterations += 1
                self.logger.debug(f"Processing tool iteration {tool_iterations}/{max_tool_iterations}")
                
                # Add the assistant's message with tool calls to the conversation
                if hasattr(response, 'content'):
                    content = response.content
                    if hasattr(content, '__dict__') and not isinstance(content, (str, list, dict)):
                        content = str(content)
                else:
                    content = str(response)
                
                self.add_message(
                    "assistant",
                    content,
                    {"has_tool_calls": True}
                )
                
                # Process tool calls
                tool_results = self._process_tool_calls(tool_calls)
                
                # Format tool results for the user message
                tool_result_blocks = []
                for tool_id, tool_result in tool_results.items():
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_result["content"],
                        "is_error": tool_result.get("is_error", False)
                    })
                
                # Add user message with tool results
                self.add_message(
                    "user", 
                    tool_result_blocks, 
                    {"is_tool_result": True}
                )
            
            # If we've exceeded the maximum iterations, log a warning
            if tool_iterations >= max_tool_iterations:
                self.logger.warning(f"Reached maximum tool iterations ({max_tool_iterations})")
                # Add a final response indicating the limit was reached
                if final_response:
                    self.add_message("assistant", 
                                    f"{final_response}\n\n[Note: Maximum tool iterations reached]")
            
            # In streaming mode with callback, return None as content was already sent
            if stream and stream_callback:
                return None
            else:
                return final_response
        
        # Error handling now provided by the error_context context manager
    
    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process tool calls from the LLM.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            Dictionary mapping tool use IDs to tool results
        """
        tool_results = {}
        
        for tool_call in tool_calls:
            tool_name = tool_call["tool_name"]
            tool_input = tool_call["input"]
            tool_id = tool_call["id"]
            
            with error_context(
                component_name="Conversation",
                operation=f"executing tool {tool_name}",
                error_class=ConversationError,
                logger=self.logger
            ):
                try:
                    # Log tool call information at INFO level for debugging
                    self.logger.info(f"Tool call: {tool_name} with input: {json.dumps(tool_input, indent=2)}")
                    
                    # Invoke the tool
                    result = self.tool_repo.invoke_tool(tool_name, tool_input)
                    
                    # Format the result for Claude's response
                    result_str = str(result)
                    tool_results[tool_id] = {
                        "content": result_str,
                        "is_error": False
                    }
                    
                    # Log successful result
                    self.logger.info(f"Tool call successful: {tool_name} returned: {result_str[:100]}..." 
                                    if len(result_str) > 100 else result_str)
                
                except Exception as e:
                    # We still need to catch exceptions here to continue processing
                    # other tools even if one fails, but we log properly
                    self.logger.error(f"Tool execution error: {tool_name}: {e}")
                    tool_results[tool_id] = {
                        "content": f"Error: {e}",
                        "is_error": True
                    }
        
        return tool_results
    
    def clear_history(self) -> None:
        """
        Clear the conversation history.
        
        Resets the conversation to its initial state while keeping
        the conversation ID and system prompt.
        """
        self.messages = []
        self.logger.debug(f"Cleared conversation history for {self.conversation_id}")
        
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary representation.
        
        Returns:
            Dictionary representation of the conversation
        """
        data = {
            "conversation_id": self.conversation_id,
            "system_prompt": self.system_prompt,
            "messages": [message.to_dict() for message in self.messages],
            "created_at": self.messages[0].created_at if self.messages else utc_now(),
            "updated_at": self.messages[-1].created_at if self.messages else utc_now(),
            "_detected_workflow_id": self._detected_workflow_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "_tried_loading_all_tools": getattr(self, '_tried_loading_all_tools', False),
            "_previously_enabled_tools": list(getattr(self, '_previously_enabled_tools', set())),
        }
        
        # Add workflow state if we have a workflow manager
        if self.workflow_manager:
            data["workflow_state"] = self.workflow_manager.to_dict()
            
        return data
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_bridge: LLMBridge,
        tool_repo: ToolRepository,
        tool_relevance_engine: 'ToolRelevanceEngine',
        workflow_manager: 'WorkflowManager',
        working_memory: 'WorkingMemory'
    ) -> 'Conversation':
        """
        Create a conversation from a dictionary representation.

        Args:
            data: Dictionary representation of the conversation
            llm_bridge: LLM bridge instance
            tool_repo: Tool repository instance
            tool_relevance_engine: Tool relevance engine instance
            workflow_manager: Workflow manager instance
            working_memory: Working memory instance for dynamic prompt content

        Returns:
            Conversation object
        """
        # Ensure required fields are present
        conversation_id = data["conversation_id"]
        system_prompt = data.get("system_prompt")
        
        # If system_prompt is missing, use the default
        if system_prompt is None:
            system_prompt = config.get_system_prompt("main_system_prompt")
            
        conversation = cls(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            llm_bridge=llm_bridge,
            tool_repo=tool_repo,
            tool_relevance_engine=tool_relevance_engine,
            workflow_manager=workflow_manager,
            working_memory=working_memory
        )
        
        # Load messages
        conversation.messages = [
            Message.from_dict(message_data)
            for message_data in data.get("messages", [])
        ]
        
        # Restore detected workflow ID
        conversation._detected_workflow_id = data.get("_detected_workflow_id")
        
        # Restore token counts
        conversation.tokens_in = data.get("tokens_in", 0)
        conversation.tokens_out = data.get("tokens_out", 0)
        
        # Restore tag handling state
        conversation._tried_loading_all_tools = data.get("_tried_loading_all_tools", False)
        conversation._previously_enabled_tools = set(data.get("_previously_enabled_tools", []))
        
        # Restore workflow state if we have a workflow manager and saved state
        if conversation.workflow_manager and "workflow_state" in data:
            conversation.workflow_manager.from_dict(data["workflow_state"])
            
            # If there's an active workflow, suspend the tool relevance engine
            if (conversation.workflow_manager.get_active_workflow() and 
                conversation.tool_relevance_engine):
                conversation.tool_relevance_engine.suspend()
        
        return conversation