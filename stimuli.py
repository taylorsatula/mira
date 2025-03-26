"""
External stimulus handling for the AI agent system.

This module provides interfaces and data structures for receiving,
processing, and routing external stimuli to the conversation system.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable

from errors import ErrorCode, StimulusError


class StimulusType(Enum):
    """
    Enumeration of supported stimulus types.

    Each stimulus type represents a different category of external trigger
    that the system can process.
    """
    MESSAGE = "message"  # Direct message stimulus
    NOTIFICATION = "notification"  # Notification-based stimulus
    EVENT = "event"  # Event-based stimulus
    SCHEDULE = "schedule"  # Time-based scheduled stimulus
    SENSOR = "sensor"  # Sensor data stimulus
    API = "api"  # External API webhook stimulus
    CUSTOM = "custom"  # Custom/user-defined stimulus


@dataclass
class Stimulus:
    """
    Representation of an external stimulus.

    Attributes:
        type: The type of stimulus
        content: The content/payload of the stimulus
        source: The source of the stimulus
        id: Unique identifier for the stimulus
        created_at: Timestamp when the stimulus was created
        metadata: Additional metadata for the stimulus
    """
    type: StimulusType
    content: Any
    source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the stimulus to a dictionary representation.

        Returns:
            Dictionary representation of the stimulus
        """
        return {
            "type": self.type.value,
            "content": self.content,
            "source": self.source,
            "id": self.id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stimulus':
        """
        Create a stimulus from a dictionary representation.

        Args:
            data: Dictionary representation of the stimulus

        Returns:
            Stimulus object
        """
        return cls(
            type=StimulusType(data["type"]),
            content=data["content"],
            source=data["source"],
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {})
        )

    def format_for_prompt(self) -> str:
        """
        Format the stimulus for inclusion in an LLM prompt.

        Returns:
            Formatted string representing the stimulus
        """
        formatted = f"[{self.type.value.upper()} from {self.source}]: {self.content}"

        # Add metadata if present
        if self.metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            formatted += f" ({metadata_str})"

        return formatted


class StimulusHandler:
    """
    Handler for routing and processing external stimuli.

    Provides methods for registering callbacks for different stimulus types
    and processing incoming stimuli.
    """

    def __init__(self):
        """Initialize the stimulus handler."""
        self.logger = logging.getLogger("stimulus_handler")
        self.handlers: Dict[StimulusType, List[Callable[[Stimulus], None]]] = {
            stim_type: [] for stim_type in StimulusType
        }
        self.conversations = []  # Track attached conversations

    def register_handler(
        self,
        stimulus_type: StimulusType,
        handler_func: Callable[[Stimulus], None]
    ) -> None:
        """
        Register a handler function for a specific stimulus type.

        Args:
            stimulus_type: The type of stimulus to handle
            handler_func: The function to call when the stimulus is received
        """
        self.handlers[stimulus_type].append(handler_func)
        self.logger.debug(
            f"Registered handler for stimulus type: {stimulus_type.value}"
        )

    def process_stimulus(self, stimulus: Stimulus) -> None:
        """
        Process an incoming stimulus by routing it to the appropriate handlers.

        Args:
            stimulus: The stimulus to process

        Raises:
            StimulusError: If there are no handlers for the stimulus type
        """
        handlers = self.handlers.get(stimulus.type, [])

        if not handlers:
            self.logger.warning(
                f"No handlers registered for stimulus type: {stimulus.type.value}"
            )
            return

        self.logger.debug(
            f"Processing {stimulus.type.value} stimulus from {stimulus.source}"
        )

        # Call all registered handlers for the stimulus type
        for handler in handlers:
            try:
                handler(stimulus)
            except Exception as e:
                self.logger.error(f"Error in stimulus handler: {e}")

    def create_stimulus(
        self,
        stimulus_type: StimulusType,
        content: Any,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Stimulus:
        """
        Create a new stimulus object.

        Args:
            stimulus_type: The type of stimulus
            content: The content/payload of the stimulus
            source: The source of the stimulus
            metadata: Optional metadata for the stimulus

        Returns:
            The newly created stimulus
        """
        return Stimulus(
            type=stimulus_type,
            content=content,
            source=source,
            metadata=metadata or {}
        )

    def create_and_process(
        self,
        stimulus_type: StimulusType,
        content: Any,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Stimulus:
        """
        Create and immediately process a new stimulus.

        Args:
            stimulus_type: The type of stimulus
            content: The content/payload of the stimulus
            source: The source of the stimulus
            metadata: Optional metadata for the stimulus

        Returns:
            The processed stimulus
        """
        stimulus = self.create_stimulus(
            stimulus_type=stimulus_type,
            content=content,
            source=source,
            metadata=metadata
        )

        self.process_stimulus(stimulus)
        return stimulus

    def attach_conversation(
        self,
        conversation: Any,
        stimulus_types: Optional[List[StimulusType]] = None,
        response_callback: Optional[Callable[[Stimulus, str], None]] = None
    ) -> None:
        """
        Attach a conversation to this stimulus handler.

        This creates handlers that add stimuli to the conversation and
        optionally process responses.

        Args:
            conversation: The conversation to attach
            stimulus_types: Optional list of stimulus types to handle
                           (defaults to all types)
            response_callback: Optional callback for processing responses
        """
        # Track the conversation
        self.conversations.append(conversation)

        # Create a handler function for the conversation
        handler = create_conversation_handler(conversation, response_callback)

        # Register the handler for specified stimulus types or all types
        if stimulus_types:
            for stim_type in stimulus_types:
                self.register_handler(stim_type, handler)
        else:
            # Register for all stimulus types
            for stim_type in StimulusType:
                self.register_handler(stim_type, handler)

        self.logger.debug(
            f"Attached conversation {conversation.conversation_id} to handler"
        )

    def process_stimulus_with_conversation(
        self,
        stimulus: Stimulus,
        conversation_id: Optional[str] = None,
        response_callback: Optional[Callable[[Stimulus, str], None]] = None
    ) -> Optional[str]:
        """
        Process a stimulus with a specific conversation.

        Args:
            stimulus: The stimulus to process
            conversation_id: Optional ID of the conversation to use
                           (uses the first attached conversation if None)
            response_callback: Optional callback for processing the response

        Returns:
            The assistant's response text, or None if no response was generated

        Raises:
            StimulusError: If no suitable conversation is found
        """
        # Find the target conversation
        target_conversation = None

        if conversation_id:
            # Find conversation by ID
            for conversation in self.conversations:
                if conversation.conversation_id == conversation_id:
                    target_conversation = conversation
                    break
        elif self.conversations:
            # Use the first attached conversation
            target_conversation = self.conversations[0]

        if not target_conversation:
            raise StimulusError(
                "No suitable conversation found for processing stimulus",
                ErrorCode.STIMULUS_PROCESSING_ERROR
            )

        # Process the stimulus with the conversation
        return process_stimulus(
            stimulus, target_conversation, response_callback
        )


def format_stimulus_context(stimuli: List[Stimulus], max_count: int = 5) -> str:
    """
    Format a list of stimuli for inclusion in an LLM prompt.

    Args:
        stimuli: List of stimuli to format
        max_count: Maximum number of stimuli to include

    Returns:
        Formatted string representing the stimuli
    """
    if not stimuli:
        return ""

    # Sort by timestamp (most recent first) and limit
    sorted_stimuli = sorted(
        stimuli, key=lambda s: s.created_at, reverse=True
    )[:max_count]

    # Format each stimulus
    formatted_items = [s.format_for_prompt() for s in sorted_stimuli]

    # Combine into a single string
    return "Recent stimuli:\n" + "\n".join(formatted_items)


def add_stimulus_to_conversation(stimulus: Stimulus, conversation: Any) -> None:
    """
    Add a stimulus to a conversation as a user message.

    Args:
        stimulus: The stimulus to add
        conversation: The conversation to add the stimulus to
    """
    # Format the stimulus content for a user message
    content = f"[{stimulus.type.value.upper()} from {stimulus.source}]: {stimulus.content}"

    # Add metadata for tracking and identification
    metadata = {
        "is_stimulus": True,
        "stimulus_id": stimulus.id,
        "stimulus_type": stimulus.type.value,
        "stimulus_source": stimulus.source,
        **stimulus.metadata  # Include original stimulus metadata
    }

    # Add as a user message (conversation.py converts roles to "user")
    conversation.add_message("user", content, metadata)


def create_conversation_handler(
    conversation: Any,
    response_callback: Optional[Callable[[Stimulus, str], None]] = None
) -> Callable[[Stimulus], None]:
    """
    Create a stimulus handler function that adds stimuli to a conversation.

    Args:
        conversation: The conversation to add stimuli to
        response_callback: Optional callback to process the response

    Returns:
        A function that can be registered as a stimulus handler
    """
    def handle_stimulus(stimulus: Stimulus) -> None:
        # Add the stimulus to the conversation
        add_stimulus_to_conversation(stimulus, conversation)

        # Generate a response if requested
        if response_callback:
            # Generate response from empty string since we already added
            # the stimulus as a user message to the conversation
            response = conversation.generate_response("")

            # Call the callback with the stimulus and response
            response_callback(stimulus, response)

    return handle_stimulus


def is_stimulus_message(message: Any) -> bool:
    """
    Check if a message originated from a stimulus.

    Args:
        message: The message to check

    Returns:
        True if the message originated from a stimulus, False otherwise
    """
    return message.metadata.get("is_stimulus", False)


def get_stimulus_metadata(message: Any) -> Dict[str, Any]:
    """
    Extract stimulus metadata from a message.

    Args:
        message: The message containing stimulus metadata

    Returns:
        A dictionary of stimulus metadata or empty dict if not a stimulus
    """
    if not is_stimulus_message(message):
        return {}

    # Extract stimulus-specific metadata
    return {
        "stimulus_id": message.metadata.get("stimulus_id"),
        "stimulus_type": message.metadata.get("stimulus_type"),
        "stimulus_source": message.metadata.get("stimulus_source")
    }


def process_stimulus(
    stimulus: Stimulus,
    conversation: Any,
    response_callback: Optional[Callable[[Stimulus, str], None]] = None
) -> Optional[str]:
    """
    Process a stimulus through a conversation and handle the response.

    This is a convenience method that:
    1. Adds the stimulus to the conversation
    2. Generates a response
    3. Optionally processes the response via callback

    Args:
        stimulus: The stimulus to process
        conversation: The conversation to use
        response_callback: Optional callback function to process the response

    Returns:
        The assistant's response text, or None if no response was generated
    """
    # Add stimulus to conversation
    add_stimulus_to_conversation(stimulus, conversation)

    try:
        # Generate response (empty input since stimulus is already added)
        response = conversation.generate_response("")

        # Process response if callback provided
        if response_callback:
            response_callback(stimulus, response)

        return response
    except Exception as e:
        logging.error(f"Error processing stimulus: {e}")
        return None
