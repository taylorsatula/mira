"""
Conversation management for the AI agent system.

This module handles conversation turns, context tracking,
tool result integration, and conversation history management.
"""
import json
import logging
import os
import re
import time
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from tool_relevance_engine import ToolRelevanceEngine
    from tools.workflows.workflow_manager import WorkflowManager
from zoneinfo import ZoneInfo

from errors import ConversationError, ErrorCode, error_context
from config import config
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository


@dataclass
class Message:
    """
    Representation of a message in a conversation.
    
    Attributes:
        role: The role of the message sender (must be 'user' or 'assistant')
        content: The content of the message (string or list of content blocks)
        id: Unique identifier for the message
        created_at: Timestamp when the message was created
        metadata: Additional metadata for the message
    """
    role: str
    content: Union[str, List[Dict[str, Any]]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
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
        
        return {
            "role": self.role,
            "content": content,
            "id": self.id,
            "created_at": self.created_at,
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
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
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
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_bridge: Optional[LLMBridge] = None,
        tool_repo: Optional[ToolRepository] = None,
        tool_relevance_engine: Optional['ToolRelevanceEngine'] = None,
        workflow_manager: Optional['WorkflowManager'] = None
    ):
        """
        Initialize a new conversation.
        
        Args:
            conversation_id: Optional unique identifier for the conversation
            system_prompt: Optional system prompt for the conversation
            llm_bridge: Optional LLM bridge instance
            tool_repo: Optional tool repository instance
            tool_relevance_engine: Optional tool relevance engine instance
            workflow_manager: Optional workflow manager instance
        """
        # Set up logging
        self.logger = logging.getLogger("conversation")
        
        # Initialize conversation ID
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.logger.debug(f"Created conversation with ID: {self.conversation_id}")
        
        # Set up conversation history
        self.messages: List[Message] = []
        
        # Set up metadata storage (including location data)
        self.metadata: Dict[str, Any] = {}
        
        # Set up conversation components
        self.llm_bridge = llm_bridge or LLMBridge()
        self.tool_repo = tool_repo or ToolRepository()
        self.tool_relevance_engine = tool_relevance_engine
        self.workflow_manager = workflow_manager
        
        # Set up conversation config
        self.max_history = config.conversation.max_history
        self.max_context_tokens = config.conversation.max_context_tokens
        
        # Store system prompt as a property (not as a message)
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            # Load system prompt from file
            self.system_prompt = config.get_system_prompt("main_system_prompt")
        
        # Load user information once and store in memory
        try:
            self.user_info = config.get_system_prompt("user_information")
            self.user_info = f"USER INFORMATION:\n{self.user_info}\n\n"
            self.logger.info("Successfully loaded user information")
        except Exception as e:
            self.logger.warning(f"Failed to load user information: {e}")
            self.user_info = ""
            
        # Initialize error tracking
        self.last_error = None
        
        # Initialize workflow tracking
        self._detected_workflow_id = None
        
        # Initialize token tracking
        self.tokens_in = 0
        self.tokens_out = 0
    
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
        
        # Prune history if needed
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
            # Validate that the message role is 'user' or 'assistant'
            if message.role not in ["user", "assistant"]:
                self.logger.warning(f"Skipping message with invalid role: {message.role}")
                continue
                
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
    
    def generate_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stream_callback: Optional[callable] = None,
        max_tool_iterations: int = config.conversation.max_tool_iterations  # Limit iterations to prevent infinite loops
    ) -> Union[str, None]:
        """
        Generate a response to user input.
        
        Args:
            user_input: User input text
            temperature: Optional temperature parameter
            max_tokens: Optional maximum tokens for the response
            stream: Whether to stream the response (default: False)
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
            raise ConversationError(error_msg, ErrorCode.INVALID_INPUT)
            
        # Add user message to conversation
        self.add_message("user", user_input)
        
        # Workflow detection and processing
        if self.workflow_manager and not self.workflow_manager.get_active_workflow():
            # Try to detect a workflow (only if no workflow is currently active)
            workflow_id, confidence = self.workflow_manager.detect_workflow(user_input)
            if workflow_id:
                self.logger.info(f"Detected potential workflow: {workflow_id} (confidence: {confidence:.2f})")
                # Store the detected workflow ID for the system prompt
                self._detected_workflow_id = workflow_id
                # We'll continue with normal processing and let the LLM handle confirmation
        
        # If workflow is active, suspend tool relevance engine and enable workflow tools
        if self.workflow_manager and self.workflow_manager.get_active_workflow():
            if self.tool_relevance_engine:
                self.tool_relevance_engine.suspend()
                self.logger.info("Suspended tool relevance engine due to active workflow")
            # Tools are enabled by the workflow manager when the workflow advances
        # Otherwise use just-in-time tool enablement
        elif self.tool_relevance_engine:
            enabled_tools = self.tool_relevance_engine.enable_relevant_tools(user_input)
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
            
            # Continue processing responses until no more tool calls are made
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
                
                # Add current time information to system prompt
                central_tz = ZoneInfo("America/Chicago")
                now = datetime.datetime.now(central_tz)
                time_info = f"Current datetime: {now.strftime('%Y-%m-%d %I:%M:%S %p')} Central Time (America/Chicago)\n\n"
                
                # Build enhanced system prompt with cached user information
                enhanced_system_prompt = time_info + self.user_info + self.system_prompt
                
                # Add workflow guidance if a workflow is active
                if self.workflow_manager and self.workflow_manager.get_active_workflow():
                    workflow_guidance = self.workflow_manager.get_system_prompt_extension()
                    enhanced_system_prompt += workflow_guidance
                
                # Add guidance for tool selection if multiple tools are enabled
                elif self.tool_repo and len(self.tool_repo.get_enabled_tools()) > 1:
                    enabled_tools = self.tool_repo.get_enabled_tools()
                    tool_list = ", ".join([t.replace("_tool", "") for t in enabled_tools])
                    tool_guidance = f"\nMultiple tools are currently available: {tool_list}. "
                    tool_guidance += "If the user's request is ambiguous about which tool to use, ask for clarification."
                    enhanced_system_prompt += tool_guidance
                
                # Add workflow detection guidance if we detected a potential workflow but none is active
                detected_workflow_id = getattr(self, '_detected_workflow_id', None)
                if self.workflow_manager and not self.workflow_manager.get_active_workflow() and detected_workflow_id:
                    workflow = self.workflow_manager.workflows.get(detected_workflow_id)
                    if workflow:
                        workflow_hint = f"\n\nI've detected that the user might want help with: {workflow['name']}. "
                        workflow_hint += "If this seems correct, you can confirm and start this workflow process by including this exact text in your response: "
                        workflow_hint += f"<!-- START_WORKFLOW:{detected_workflow_id} -->"
                        enhanced_system_prompt += workflow_hint
                
                # Generate response (streaming or standard)
                if stream:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=enhanced_system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools,
                        stream=True,
                        callback=_handle_streaming_response
                    )
                    
                    # Update token counts if available in the response object
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        self.tokens_in += getattr(usage, 'input_tokens', 0)
                        self.tokens_out += getattr(usage, 'output_tokens', 0)
                    
                    # For stream responses, get the final message
                    if hasattr(response, 'get_final_message'):
                        final_message = response.get_final_message()
                        response = final_message
                else:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=enhanced_system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools
                    )
                    
                    # Update token counts if available in the response object
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        self.tokens_in += getattr(usage, 'input_tokens', 0)
                        self.tokens_out += getattr(usage, 'output_tokens', 0)
                
                
                # Extract text content for final return value
                assistant_response = self.llm_bridge.extract_text_content(response)
                final_response = assistant_response
                
                # Check for workflow commands
                if self.workflow_manager:
                    # Check for workflow commands (start, complete, cancel)
                    command_found, command_type, command_params = self.workflow_manager.check_for_workflow_commands(assistant_response)
                    
                    if command_found:
                        if command_type == "start" and not self.workflow_manager.get_active_workflow():
                            workflow_id = command_params
                            try:
                                self.workflow_manager.start_workflow(workflow_id)
                                self.logger.info(f"Started workflow: {workflow_id}")
                                # Suspend tool relevance engine
                                if self.tool_relevance_engine:
                                    self.tool_relevance_engine.suspend()
                            except Exception as e:
                                self.logger.error(f"Error starting workflow {workflow_id}: {e}")
                        
                        elif command_type == "complete" and self.workflow_manager.get_active_workflow():
                            try:
                                self.workflow_manager.advance_workflow()
                                self.logger.info("Advanced workflow to next step")
                            except Exception as e:
                                self.logger.error(f"Error advancing workflow: {e}")
                        
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
        
    def reload_user_information(self) -> None:
        """
        Reload user information from file.
        
        Call this method when the user information file has been updated
        and you want to refresh the cached information without restarting
        the application.
        """
        try:
            self.user_info = config.get_system_prompt("user_information", reload=True)
            self.user_info = f"USER INFORMATION:\n{self.user_info}\n\n"
            self.logger.info("Successfully reloaded user information")
        except Exception as e:
            self.logger.warning(f"Failed to reload user information: {e}")
            # Keep existing user_info if refresh fails
    
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
            "created_at": self.messages[0].created_at if self.messages else time.time(),
            "updated_at": self.messages[-1].created_at if self.messages else time.time(),
            "_detected_workflow_id": self._detected_workflow_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }
        
        # Add workflow state if we have a workflow manager
        if self.workflow_manager:
            data["workflow_state"] = self.workflow_manager.to_dict()
            
        return data
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_bridge: Optional[LLMBridge] = None,
        tool_repo: Optional[ToolRepository] = None,
        tool_relevance_engine: Optional['ToolRelevanceEngine'] = None,
        workflow_manager: Optional['WorkflowManager'] = None
    ) -> 'Conversation':
        """
        Create a conversation from a dictionary representation.
        
        Args:
            data: Dictionary representation of the conversation
            llm_bridge: Optional LLM bridge instance
            tool_repo: Optional tool repository instance
            tool_relevance_engine: Optional tool relevance engine instance
            workflow_manager: Optional workflow manager instance
            
        Returns:
            Conversation object
        """
        conversation = cls(
            conversation_id=data["conversation_id"],
            system_prompt=data.get("system_prompt"),
            llm_bridge=llm_bridge,
            tool_repo=tool_repo,
            tool_relevance_engine=tool_relevance_engine,
            workflow_manager=workflow_manager
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
        
        # Restore workflow state if we have a workflow manager and saved state
        if conversation.workflow_manager and "workflow_state" in data:
            conversation.workflow_manager.from_dict(data["workflow_state"])
            
            # If there's an active workflow, suspend the tool relevance engine
            if (conversation.workflow_manager.get_active_workflow() and 
                conversation.tool_relevance_engine):
                conversation.tool_relevance_engine.suspend()
        
        return conversation