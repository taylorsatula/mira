"""
Conversation management for the AI agent system.

This module handles conversation turns, context tracking,
tool result integration, and conversation history management.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

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
        tool_repo: Optional[ToolRepository] = None
    ):
        """
        Initialize a new conversation.
        
        Args:
            conversation_id: Optional unique identifier for the conversation
            system_prompt: Optional system prompt for the conversation
            llm_bridge: Optional LLM bridge instance
            tool_repo: Optional tool repository instance
        """
        # Set up logging
        self.logger = logging.getLogger("conversation")
        
        # Initialize conversation ID
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.logger.debug(f"Created conversation with ID: {self.conversation_id}")
        
        # Set up conversation history
        self.messages: List[Message] = []
        
        # Set up conversation components
        self.llm_bridge = llm_bridge or LLMBridge()
        self.tool_repo = tool_repo or ToolRepository()
        
        # Set up conversation config
        self.max_history = config.conversation.max_history
        self.max_context_tokens = config.conversation.max_context_tokens
        
        # Store system prompt as a property (not as a message)
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            # Load system prompt from file
            self.system_prompt = config.get_system_prompt("main_system_prompt")
            
        # Initialize error tracking
        self.last_error = None
    
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
                
                # Start response time tracking
                start_time = time.time()
                
                # Reset sequence tracking at the start of a conversation turn
                if tool_iterations == 0 and self.tool_repo and hasattr(self.tool_repo, 'reset_sequence_tracking'):
                    self.tool_repo.reset_sequence_tracking()
                
                # Select appropriate tools based on iteration
                if self.tool_repo and hasattr(self.tool_repo, 'select_tools_for_message'):
                    if tool_iterations == 0:
                        # Initial message - use message-based selection
                        user_message = user_input
                        selected_tools = self.tool_repo.select_tools_for_message(user_message)
                        self.logger.debug(f"Using {len(selected_tools)} selected tools for initial response")
                    else:
                        # Subsequent iterations - use sequence-based prediction if available
                        last_tool_call = None
                        
                        # Try to find the last tool call from previous messages
                        for i in range(len(self.messages) - 1, -1, -1):
                            if self.messages[i].role == "assistant" and self.messages[i].metadata.get("has_tool_calls", False):
                                # Extract tool calls from content
                                tool_calls = self.llm_bridge.extract_tool_calls(self.messages[i].content) if hasattr(self.messages[i].content, 'content') else []
                                if tool_calls:
                                    last_tool_call = tool_calls[-1]["tool_name"]
                                    break
                        
                        if last_tool_call and hasattr(self.tool_repo, 'get_likely_next_tools'):
                            # Get likely next tools based on sequence data
                            self.logger.info(f"Getting likely next tools after: {last_tool_call}")
                            likely_tools = self.tool_repo.get_likely_next_tools(last_tool_call)
                            
                            if likely_tools:
                                # We have predictions for next tools
                                self.logger.info(f"Found likely next tools: {likely_tools}")
                                from config import config
                                # Get essential tools + likely next tools
                                essential_tools = config.tools.essential_tools
                                
                                # Combine essential and predicted tools, ensuring no duplicates
                                predict_tools = list(set(essential_tools + likely_tools))
                                self.logger.info(f"Combined essential and predicted tools: {predict_tools}")
                                
                                # Calculate how many additional tools we can include
                                max_tools = config.tools.max_tools
                                remaining_slots = max(0, max_tools - len(predict_tools))
                                
                                # Ensure all predicted tools are included
                                # First, get all tool definitions (we'll filter later)
                                all_tools = self.tool_repo.get_all_tool_definitions()
                                
                                # Create a priority dictionary for sorting
                                priority_dict = {name: idx for idx, name in enumerate(predict_tools)}
                                
                                # Filter and prioritize tools based on prediction
                                prioritized_tools = []
                                additional_tools = []
                                
                                for tool in all_tools:
                                    tool_name = tool.get("name", "")
                                    if tool_name in priority_dict:
                                        # Add to prioritized with its priority
                                        prioritized_tools.append((priority_dict[tool_name], tool))
                                    else:
                                        # Add to additional tools
                                        additional_tools.append(tool)
                                
                                # Sort prioritized tools by priority
                                prioritized_tools.sort()
                                # Extract just the tool definitions after sorting
                                selected_tools = [tool for _, tool in prioritized_tools]
                                
                                # If we have slots remaining, get more tools based on the message
                                if remaining_slots > 0 and additional_tools:
                                    # Use message content to select remaining tools
                                    selected_names = {tool.get("name", "") for tool in selected_tools}
                                    candidate_tools = [t for t in additional_tools if t.get("name", "") not in selected_names]
                                    
                                    if candidate_tools and self.tool_repo.selector:
                                        # Use selector for remaining tools
                                        more_tools = self.tool_repo.selector.select_tools(
                                            message=user_input,
                                            all_tools=candidate_tools,
                                            min_tools=0,  # We already have our essential tools
                                            max_tools=remaining_slots
                                        )
                                        selected_tools.extend(more_tools)
                                        
                                self.logger.info(f"Final tools after prediction: {[t.get('name', '') for t in selected_tools]}")
                            else:
                                # No predictions, use general selection
                                self.logger.info("No prediction data, using general selection")
                                selected_tools = self.tool_repo.select_tools_for_message(user_input)
                                self.logger.debug("Using general selection for follow-up response")
                        else:
                            # Fallback - use general selection based on the current conversation state
                            selected_tools = self.tool_repo.select_tools_for_message(user_input)
                            self.logger.debug("Using general selection for follow-up response")
                else:
                    # Fallback - use all tools if selection isn't available
                    selected_tools = self.tool_repo.get_all_tool_definitions() if self.tool_repo else None
                    self.logger.debug("Using all tools for follow-up response")
                
                # Generate response (streaming or standard)
                if stream:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=self.system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools,
                        stream=True,
                        callback=_handle_streaming_response
                    )
                    
                    # For stream responses, get the final message
                    if hasattr(response, 'get_final_message'):
                        final_message = response.get_final_message()
                        response = final_message
                else:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=self.system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=selected_tools
                    )
                
                # Record response time if metrics are available
                if self.tool_repo and hasattr(self.tool_repo, 'metrics') and self.tool_repo.metrics:
                    end_time = time.time()
                    self.tool_repo.metrics.record_response_time(
                        end_time - start_time,
                        used_selection=(tool_iterations == 0)  # Only first iteration uses selection
                    )
                
                # Extract text content for final return value
                assistant_response = self.llm_bridge.extract_text_content(response)
                final_response = assistant_response
                
                # Check for tool calls
                tool_calls = self.llm_bridge.extract_tool_calls(response)
                
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
                    # Invoke the tool
                    result = self.tool_repo.invoke_tool(tool_name, tool_input)
                    tool_results[tool_id] = {
                        "content": str(result),
                        "is_error": False
                    }
                    
                    self.logger.debug(f"Tool call successful: {tool_name}")
                
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
        return {
            "conversation_id": self.conversation_id,
            "system_prompt": self.system_prompt,
            "messages": [message.to_dict() for message in self.messages],
            "created_at": self.messages[0].created_at if self.messages else time.time(),
            "updated_at": self.messages[-1].created_at if self.messages else time.time(),
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_bridge: Optional[LLMBridge] = None,
        tool_repo: Optional[ToolRepository] = None
    ) -> 'Conversation':
        """
        Create a conversation from a dictionary representation.
        
        Args:
            data: Dictionary representation of the conversation
            llm_bridge: Optional LLM bridge instance
            tool_repo: Optional tool repository instance
            
        Returns:
            Conversation object
        """
        conversation = cls(
            conversation_id=data["conversation_id"],
            system_prompt=data.get("system_prompt"),
            llm_bridge=llm_bridge,
            tool_repo=tool_repo
        )
        
        # Load messages
        conversation.messages = [
            Message.from_dict(message_data)
            for message_data in data.get("messages", [])
        ]
        
        return conversation
