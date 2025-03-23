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

from errors import ConversationError, ErrorCode
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
        self.max_history = config.get("conversation.max_history", 10)
        self.max_context_tokens = config.get("conversation.max_context_tokens", 100000)
        
        # Store system prompt as a property (not as a message)
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = """You are a helpful AI assistant with access to tools.

## Available Tool: extract
Use the extraction tool to analyze and extract information from user messages.
- Available templates: "general", "personal_info", "keywords", "question", "sentiment", "entities", "food_preferences", "custom"
- For general template, provide the target information as a parameter
- For custom template, provide the full extraction prompt as the target parameter

Examples:
- extract(message="I'm John from New York", template="personal_info")
- extract(message="Is the weather nice today?", template="question")
- extract(message="I'm feeling great today!", template="sentiment")
- extract(message="When was the Declaration of Independence signed?", template="general", target="year")

## Available Tool: persistence
Use the persistence tool to store and retrieve data:
- Operations: "get", "set", "delete", "list"
- Files are stored as JSON in the persistent/ directory
- Must include .json extension in filename

Examples:
- persistence(filename="preferences.json", operation="get", key="preferences")
- persistence(filename="preferences.json", operation="set", key="preferences", value=extracted_data)
- persistence(filename="user_info.json", operation="list")

## Asynchronous Task Tools
For operations that might take a long time, you can use async tools to run tasks in the background:

### schedule_async_task
Schedule a task to run in the background without blocking the conversation:
- description: A short description of the task
- task_prompt: Detailed instructions for the background assistant
- notify_on_completion: Whether to notify the user when the task completes

Example:
- schedule_async_task(description="Generate food preference report", task_prompt="Analyze user's food preferences and create a detailed report", notify_on_completion=True)

### check_async_task
Check the status and results of a background task:
- task_id: The ID of the task to check

Example:
- check_async_task(task_id="123e4567-e89b-12d3-a456-426614174000")

Background tasks can save their results to the persistent/ directory which you can retrieve later with the persistence tool.

## Food Preference Workflow
When users express food preferences (e.g., "I love rosemary" or "I hate cilantro"), use these specific steps:
1. Extract: extract(message="user message", template="food_preferences")
2. Get current preferences: persistence(filename="preferences.json", operation="get", key="preferences")
3. Update the preferences with new data
4. Save: persistence(filename="preferences.json", operation="set", key="preferences", value=updated_preferences)
5. Confirm to user that their preference was saved"""
            
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
        max_tool_iterations: int = 5  # Limit iterations to prevent infinite loops
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
        # Add user message to conversation
        self.add_message("user", user_input)
        
        try:
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
                
                # Generate response (streaming or standard)
                if stream:
                    response = self.llm_bridge.generate_response(
                        messages=messages,
                        system_prompt=self.system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=self.tool_repo.get_all_tool_definitions() if self.tool_repo else None,
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
                        tools=self.tool_repo.get_all_tool_definitions() if self.tool_repo else None
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
        
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            self.logger.error(error_msg)
            
            # Log the error but don't add to conversation since we can't use system role
            self.logger.error(f"Error message not added to conversation: {error_msg}")
            # Store error in instance variable for potential later use
            self.last_error = error_msg
            
            raise ConversationError(
                error_msg,
                ErrorCode.CONTEXT_OVERFLOW if "context" in str(e).lower() else ErrorCode.UNKNOWN_ERROR
            )
    
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
            
            try:
                # Invoke the tool
                result = self.tool_repo.invoke_tool(tool_name, tool_input)
                tool_results[tool_id] = {
                    "content": str(result),
                    "is_error": False
                }
                
                self.logger.debug(f"Tool call successful: {tool_name}")
            
            except Exception as e:
                # Handle tool execution errors
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