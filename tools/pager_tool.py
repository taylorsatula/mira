"""
Pager tool for managing virtual pager device messaging.

This tool simulates a pager device system allowing users to send and receive messages
through virtual pagers. Each pager has a unique ID and can send/receive messages with
location tracking and priority levels.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

import logging
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from db import Database, Base
from config.registry import registry
from utils.timezone_utils import (
    validate_timezone, get_default_timezone, convert_to_timezone,
    format_datetime, parse_time_string, utc_now, ensure_utc
)
from api.llm_provider import LLMProvider

# Define configuration class for PagerTool
class PagerToolConfig(BaseModel):
    """Configuration for the pager_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    default_expiry_hours: int = Field(default=24, description="Default message expiry time in hours")
    max_message_length: int = Field(default=300, description="Maximum message length")
    ai_distillation_enabled: bool = Field(default=True, description="Whether to use AI for message distillation")

# Register with registry
registry.register("pager_tool", PagerToolConfig)


class PagerDevice(Base):
    """
    Pager device model for storing pager device registry.
    
    Maps to the 'pager_devices' table with columns for device details
    including ID, name, and registration information.
    """
    __tablename__ = 'pager_devices'

    # Primary key
    id = Column(String, primary_key=True)  # Format: PAGER-XXXX
    user_id = Column(String, nullable=False)  # User who owns this pager
    
    # Device details
    name = Column(String, nullable=False)  # Friendly name for the pager
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: utc_now())
    last_active = Column(DateTime, default=lambda: utc_now())
    
    # Authentication
    device_secret = Column(String, nullable=False)  # Secret key for device (never shared)
    device_fingerprint = Column(String, nullable=False)  # Public fingerprint for verification
    
    # Status
    active = Column(Boolean, default=True)
    
    # Relationships
    sent_messages = relationship("PagerMessage", foreign_keys="PagerMessage.sender_id", back_populates="sender_device")
    received_messages = relationship("PagerMessage", foreign_keys="PagerMessage.recipient_id", back_populates="recipient_device")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the pager device with timestamps in user timezone
        """
        # Get user's timezone
        user_tz = get_default_timezone()
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": format_datetime(self.created_at, "date_time", user_tz) if self.created_at else None,
            "last_active": format_datetime(self.last_active, "date_time", user_tz) if self.last_active else None,
            "active": self.active,
            "timezone": user_tz,
            "device_fingerprint": self.device_fingerprint  # Public fingerprint for verification
        }


class PagerTrust(Base):
    """
    Trust relationship model for storing known device fingerprints.
    Similar to SSH known_hosts - tracks which devices we trust.
    """
    __tablename__ = 'pager_trust'
    
    # Primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    
    # Trust relationship
    trusting_device_id = Column(String, ForeignKey('pager_devices.id'), nullable=False)
    trusted_device_id = Column(String, nullable=False)  # Device ID we trust
    trusted_fingerprint = Column(String, nullable=False)  # Their fingerprint
    trusted_name = Column(String)  # Their name at time of trust
    
    # Metadata
    first_seen = Column(DateTime, default=lambda: utc_now())
    last_verified = Column(DateTime, default=lambda: utc_now())
    trust_status = Column(String, default="trusted")  # trusted, revoked, conflicted
    
    # Ensure unique trust relationships
    __table_args__ = (
        UniqueConstraint('trusting_device_id', 'trusted_device_id', name='_trust_uc'),
    )
    
    # Relationships
    trusting_device = relationship("PagerDevice", foreign_keys=[trusting_device_id])


class PagerMessage(Base):
    """
    Pager message model for storing messages between pager devices.
    
    Maps to the 'pager_messages' table with columns for message content,
    metadata, and delivery status.
    """
    __tablename__ = 'pager_messages'

    # Primary key
    id = Column(String, primary_key=True)  # Format: MSG-XXXXXXXX
    user_id = Column(String, nullable=False)  # User context for this message
    
    # Message details
    sender_id = Column(String, ForeignKey('pager_devices.id'), nullable=False)
    recipient_id = Column(String, ForeignKey('pager_devices.id'), nullable=False)
    content = Column(Text, nullable=False)
    
    # AI distillation
    original_content = Column(Text)  # Original content before distillation
    ai_distilled = Column(Boolean, default=False)
    
    # Metadata
    priority = Column(Integer, default=0)  # 0=normal, 1=high, 2=urgent
    location = Column(String)  # Optional location information
    
    # Timestamps
    sent_at = Column(DateTime, default=lambda: utc_now())
    expires_at = Column(DateTime)
    read_at = Column(DateTime)
    
    # Status
    delivered = Column(Boolean, default=True)  # In our simulation, always delivered
    read = Column(Boolean, default=False)
    
    # Authentication
    message_signature = Column(String)  # Signature to verify message authenticity
    sender_fingerprint = Column(String)  # Sender's device fingerprint
    
    # Relationships
    sender_device = relationship("PagerDevice", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient_device = relationship("PagerDevice", foreign_keys=[recipient_id], back_populates="received_messages")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the message with timestamps in user timezone
        """
        # Get user's timezone
        user_tz = get_default_timezone()
        
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_device.name if self.sender_device else None,
            "recipient_id": self.recipient_id,
            "recipient_name": self.recipient_device.name if self.recipient_device else None,
            "content": self.content,
            "original_content": self.original_content,
            "ai_distilled": self.ai_distilled,
            "priority": self.priority,
            "priority_label": ["normal", "high", "urgent"][self.priority] if 0 <= self.priority <= 2 else "unknown",
            "location": self.location,
            "sent_at": format_datetime(self.sent_at, "date_time", user_tz) if self.sent_at else None,
            "expires_at": format_datetime(self.expires_at, "date_time", user_tz) if self.expires_at else None,
            "read_at": format_datetime(self.read_at, "date_time", user_tz) if self.read_at else None,
            "delivered": self.delivered,
            "read": self.read,
            "timezone": user_tz,
            "sender_fingerprint": self.sender_fingerprint,
            "message_signature": self.message_signature
        }


class PagerTool(Tool):
    """
    Tool for managing virtual pager devices and messaging.
    
    This tool simulates a pager system where users can create virtual pagers,
    send messages between them, and manage message delivery with priority
    and location tracking.
    """

    name = "pager_tool"
    
    openai_schema = {
        "type": "function",
        "function": {
            "name": "pager_tool",
            "description": "Virtual pager messaging system. Create pager devices and send/receive short messages with priority levels and location tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["register_device", "send_message", "get_received_messages", "get_sent_messages", "mark_message_read", "get_devices", "deactivate_device", "cleanup_expired", "list_trusted_devices", "revoke_trust", "send_location"],
                        "description": "The pager operation to perform"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the pager device (for register_device)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the pager device (optional)"
                    },
                    "sender_id": {
                        "type": "string",
                        "description": "ID of the sending pager (format: PAGER-XXXX)"
                    },
                    "recipient_id": {
                        "type": "string",
                        "description": "ID of the receiving pager (format: PAGER-XXXX)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content (max 300 chars or will be AI-distilled)"
                    },
                    "priority": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "Message priority: 0=normal, 1=high, 2=urgent"
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location information to attach to message"
                    },
                    "expiry_hours": {
                        "type": "integer",
                        "description": "Hours until message expires (default: 24)"
                    },
                    "device_secret": {
                        "type": "string",
                        "description": "Device secret for authentication (proves sender identity)"
                    },
                    "untrusted_device_id": {
                        "type": "string",
                        "description": "ID of device to revoke trust for"
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional message to include with location pin (max 50 chars)"
                    },
                    "pager_id": {
                        "type": "string",
                        "description": "ID of a specific pager device"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "ID of a specific message (format: MSG-XXXXXXXX)"
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only return unread messages (default: false)"
                    },
                    "include_expired": {
                        "type": "boolean",
                        "description": "Include expired messages (default: false)"
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Only return active devices (default: true)"
                    },
                    "kwargs": {
                        "type": "string",
                        "description": "JSON string containing operation parameters"
                    }
                },
                "required": ["operation"]
            }
        }
    }
    
    simple_description = """
    Manages virtual pager devices for sending and receiving short messages. Use this tool when the user
    wants to simulate pager messaging, send urgent notifications, or manage a virtual pager system."""
    
    implementation_details = """
    
    IMPORTANT: This tool requires parameters to be passed as a JSON string in the "kwargs" field.
    The tool supports these operations:
    
    1. register_device: Create a new virtual pager device.
       - Required: name (friendly name for the pager)
       - Optional: description (details about the pager)
       - Returns the created pager device with unique ID (format: PAGER-XXXX)
    
    2. send_message: Send a message from one pager to another.
       - Required: sender_id, recipient_id, content
       - Optional: priority (0=normal, 1=high, 2=urgent), location, expiry_hours
       - Content over 300 chars will be AI-distilled to fit pager constraints
       - Returns the sent message details
    
    3. get_received_messages: Get messages received by a specific pager.
       - Required: pager_id (the pager device ID)
       - Optional: unread_only (boolean), include_expired (boolean)
       - Returns list of received messages
    
    4. get_sent_messages: Get messages sent from a specific pager.
       - Required: pager_id (the pager device ID)
       - Optional: include_expired (boolean)
       - Returns list of sent messages
       
    5. mark_message_read: Mark a message as read.
       - Required: message_id
       - Returns the updated message
       
    6. get_devices: List all registered pager devices.
       - Optional: active_only (boolean, default true)
       - Returns list of pager devices
       
    7. deactivate_device: Deactivate a pager device.
       - Required: pager_id
       - Returns confirmation of deactivation
       
    8. cleanup_expired: Remove expired messages from the system.
       - Returns count of messages cleaned up
    
    9. list_trusted_devices: List devices trusted by a specific pager.
       - Required: pager_id
       - Returns list of trusted devices with fingerprints and status
       
    10. revoke_trust: Revoke trust for a specific device (allows re-establishing trust).
        - Required: pager_id, untrusted_device_id
        - Returns confirmation of trust revocation
        
    11. send_location: Send a location pin message from one pager to another.
        - Required: sender_id, recipient_id
        - Optional: priority (0=normal, 1=high, 2=urgent), message (brief note), device_secret
        - Automatically includes current location as a pin
        - Returns the sent location message
       
    The tool uses AI to automatically distill long messages to fit pager constraints while
    preserving the essential information. Location information can be attached to messages
    for context, and priority levels help indicate message urgency.
    
    Location pins are a special feature that allow sending your current coordinates as a 
    message, perfect for emergencies or meetups. The location is formatted with both a 
    human-readable address and technical coordinates.
    """
    
    description = simple_description + implementation_details
    
    usage_examples = [
        {
            "input": {
                "operation": "register_device",
                "kwargs": "{\"name\": \"Field Unit Alpha\", \"description\": \"Emergency response team leader\"}"
            },
            "output": {
                "device": {
                    "id": "PAGER-A1B2",
                    "name": "Field Unit Alpha",
                    "description": "Emergency response team leader",
                    "active": True
                }
            }
        },
        {
            "input": {
                "operation": "send_message",
                "kwargs": "{\"sender_id\": \"PAGER-A1B2\", \"recipient_id\": \"PAGER-C3D4\", \"content\": \"Code 3 response needed at Main St\", \"priority\": 2, \"location\": \"Main St & 5th Ave\"}"
            },
            "output": {
                "message": {
                    "id": "MSG-12345678",
                    "sender_id": "PAGER-A1B2",
                    "recipient_id": "PAGER-C3D4",
                    "content": "Code 3 response needed at Main St",
                    "priority": 2,
                    "priority_label": "urgent",
                    "location": "Main St & 5th Ave"
                }
            }
        },
        {
            "input": {
                "operation": "send_location",
                "kwargs": "{\"sender_id\": \"PAGER-A1B2\", \"recipient_id\": \"PAGER-C3D4\", \"message\": \"Stuck in traffic\", \"priority\": 2}"
            },
            "output": {
                "message": {
                    "id": "MSG-87654321",
                    "content": "📍 Location Pin: 1-1 Kitahama, Chuo-ku, Osaka (Near Osaka City Hall)\\nNote: Stuck in traffic\\n[34.6937, 135.5023]",
                    "priority": 2,
                    "location": "{\"lat\": 34.6937, \"lng\": 135.5023, \"accuracy_meters\": 15}"
                }
            }
        }
    ]

    def __init__(self):
        """Initialize the pager tool with database access and LLM provider."""
        super().__init__()
        self.db = Database()
        self.llm = LLMProvider()
        
        # Ensure data directory exists
        self.data_dir = os.path.join("data", "tools", "pager_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("PagerTool initialized")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a pager operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        1. register_device: Create a new pager device
           - Required: name
           - Optional: description
           - Returns: Dict with created device

        2. send_message: Send a message between pagers
           - Required: sender_id, recipient_id, content
           - Optional: priority, location, expiry_hours
           - Returns: Dict with sent message

        3. get_received_messages: Get messages for a pager
           - Required: pager_id
           - Optional: unread_only, include_expired
           - Returns: Dict with list of messages

        4. get_sent_messages: Get messages sent by a pager
           - Required: pager_id
           - Optional: include_expired
           - Returns: Dict with list of messages

        5. mark_message_read: Mark a message as read
           - Required: message_id
           - Returns: Dict with updated message

        6. get_devices: List pager devices
           - Optional: active_only
           - Returns: Dict with list of devices

        7. deactivate_device: Deactivate a pager
           - Required: pager_id
           - Returns: Dict with confirmation

        8. cleanup_expired: Clean up expired messages
           - Returns: Dict with cleanup stats
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Route to the appropriate operation
            if operation == "register_device":
                return self._register_device(**kwargs)
            elif operation == "send_message":
                return self._send_message(**kwargs)
            elif operation == "get_received_messages":
                return self._get_received_messages(**kwargs)
            elif operation == "get_sent_messages":
                return self._get_sent_messages(**kwargs)
            elif operation == "mark_message_read":
                return self._mark_message_read(**kwargs)
            elif operation == "get_devices":
                return self._get_devices(**kwargs)
            elif operation == "deactivate_device":
                return self._deactivate_device(**kwargs)
            elif operation == "cleanup_expired":
                return self._cleanup_expired(**kwargs)
            elif operation == "list_trusted_devices":
                return self._list_trusted_devices(**kwargs)
            elif operation == "revoke_trust":
                return self._revoke_trust(**kwargs)
            elif operation == "send_location":
                return self._send_location(**kwargs)
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "register_device, send_message, get_received_messages, "
                    "get_sent_messages, mark_message_read, get_devices, "
                    "deactivate_device, cleanup_expired, list_trusted_devices, "
                    "revoke_trust, send_location",
                    ErrorCode.TOOL_INVALID_INPUT,
                )

    def _register_device(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new pager device.
        
        Args:
            name: Friendly name for the pager device
            description: Optional description of the device
            
        Returns:
            Dict containing the created device
            
        Raises:
            ToolError: If required fields are missing
        """
        self.logger.info(f"Registering pager device: {name}")
        
        # Validate required parameters
        if not name:
            raise ToolError(
                "Name is required for registering a pager device",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Generate a unique pager ID
        pager_id = f"PAGER-{uuid.uuid4().hex[:4].upper()}"
        
        # Generate device secret and fingerprint
        import hashlib
        device_secret = f"SECRET-{uuid.uuid4().hex[:32].upper()}"
        # Fingerprint is hash of device ID + secret (like SSH host key fingerprint)
        device_fingerprint = hashlib.sha256(
            f"{pager_id}{device_secret}".encode()
        ).hexdigest()[:16].upper()
        
        # Create the device object
        device = PagerDevice(
            id=pager_id,
            user_id=self.user_id,
            name=name,
            description=description,
            created_at=utc_now(),
            last_active=utc_now(),
            active=True,
            device_secret=device_secret,
            device_fingerprint=device_fingerprint
        )
        
        # Save device to database
        try:
            self.db.add(device)
            self.logger.info(f"Registered pager device with ID: {pager_id}")
        except Exception as e:
            self.logger.error(f"Error saving pager device: {e}")
            raise ToolError(
                f"Failed to save pager device: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        return {
            "device": device.to_dict(),
            "message": f"Pager device '{name}' registered successfully with ID {pager_id}"
        }

    def _send_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        priority: Optional[int] = 0,
        location: Optional[str] = None,
        expiry_hours: Optional[int] = 24,
        device_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message from one pager to another.
        
        Args:
            sender_id: ID of the sending pager
            recipient_id: ID of the receiving pager
            content: Message content (will be distilled if too long)
            priority: Message priority (0=normal, 1=high, 2=urgent)
            location: Optional location information
            expiry_hours: Hours until message expires (default 24)
            device_secret: Device secret for authentication (proves sender identity)
            
        Returns:
            Dict containing the sent message
            
        Raises:
            ToolError: If devices not found or parameters invalid
        """
        self.logger.info(f"Sending message from {sender_id} to {recipient_id}")
        
        # Validate required parameters
        if not all([sender_id, recipient_id, content]):
            raise ToolError(
                "sender_id, recipient_id, and content are required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate priority
        if priority not in [0, 1, 2]:
            raise ToolError(
                "Priority must be 0 (normal), 1 (high), or 2 (urgent)",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Get sender and recipient devices
        sender = self.db.get(PagerDevice, sender_id)
        if not sender or not sender.active:
            raise ToolError(
                f"Sender device '{sender_id}' not found or inactive",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Verify device secret (optional but recommended)
        if device_secret and device_secret != sender.device_secret:
            raise ToolError(
                f"Invalid device secret for device '{sender_id}'",
                ErrorCode.TOOL_PERMISSION_DENIED
            )
            
        recipient = self.db.get(PagerDevice, recipient_id)
        if not recipient or not recipient.active:
            raise ToolError(
                f"Recipient device '{recipient_id}' not found or inactive",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Update sender's last active time
        sender.last_active = utc_now()
        self.db.update(sender)
        
        # Check if content needs distillation
        original_content = None
        ai_distilled = False
        
        from config import config
        max_length = config.tools.pager_tool.max_message_length
        
        if len(content) > max_length:
            if config.tools.pager_tool.ai_distillation_enabled:
                original_content = content
                content = self._distill_message(content, max_length)
                ai_distilled = True
            else:
                # Enforce character limit for human messages without AI
                raise ToolError(
                    f"Message too long: {len(content)} characters (max {max_length})",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
        # Calculate expiry time
        expires_at = utc_now() + timedelta(hours=expiry_hours)
        
        # Generate message ID
        message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        
        # Create message signature using device secret
        import hashlib
        message_signature = hashlib.sha256(
            f"{message_id}{sender.device_secret}{content}{recipient_id}".encode()
        ).hexdigest()[:16].upper()
        
        # Create the message
        message = PagerMessage(
            id=message_id,
            user_id=self.user_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            original_content=original_content,
            ai_distilled=ai_distilled,
            priority=priority,
            location=location,
            sent_at=utc_now(),
            expires_at=expires_at,
            delivered=True,
            read=False,
            message_signature=message_signature,
            sender_fingerprint=sender.device_fingerprint
        )
        
        # Save message to database
        try:
            self.db.add(message)
            self.logger.info(f"Sent message with ID: {message_id}")
        except Exception as e:
            self.logger.error(f"Error saving message: {e}")
            raise ToolError(
                f"Failed to send message: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        result = {
            "message": message.to_dict(),
            "status": "delivered"
        }
        
        if ai_distilled:
            result["distillation_note"] = f"Message was distilled from {len(original_content)} to {len(content)} characters"
            
        return result

    def _get_received_messages(
        self,
        pager_id: str,
        unread_only: bool = False
    ) -> Dict[str, Any]:
        """
        Get messages received by a pager device.
        
        Args:
            pager_id: ID of the pager device
            unread_only: Only return unread messages
            
        Returns:
            Dict containing list of received messages
            
        Raises:
            ToolError: If device not found
        """
        self.logger.info(f"Getting received messages for pager {pager_id}")
        
        # Validate device exists
        device = self.db.get(PagerDevice, pager_id)
        if not device:
            raise ToolError(
                f"Pager device '{pager_id}' not found",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Update last active time
        device.last_active = utc_now()
        self.db.update(device)
        
        # Query messages
        with self.db.get_session() as session:
            query = session.query(PagerMessage).filter(
                PagerMessage.recipient_id == pager_id
            )
            
            if unread_only:
                query = query.filter(PagerMessage.read.is_(False))
                
            # Always filter out expired messages
            query = query.filter(PagerMessage.expires_at > utc_now())
                
            # Sort by sent time descending (newest first)
            query = query.order_by(PagerMessage.sent_at.desc())
            
            messages = query.all()
            
            # Check trust status for each message
            message_list = []
            for msg in messages:
                msg_dict = msg.to_dict()
                
                # Check if we trust this sender
                trust_status = self._check_trust_status(
                    pager_id, 
                    msg.sender_id, 
                    msg.sender_fingerprint
                )
                msg_dict['trust_status'] = trust_status
                
                # Note: Conflicted messages will never reach here as they're rejected during send
                message_list.append(msg_dict)
            
        return {
            "messages": message_list,
            "count": len(message_list),
            "pager_id": pager_id,
            "pager_name": device.name,
            "filters": {
                "unread_only": unread_only
            }
        }

    def _get_sent_messages(
        self,
        pager_id: str,
        include_expired: bool = False
    ) -> Dict[str, Any]:
        """
        Get messages sent by a pager device.
        
        Args:
            pager_id: ID of the pager device
            include_expired: Include expired messages
            
        Returns:
            Dict containing list of sent messages
            
        Raises:
            ToolError: If device not found
        """
        self.logger.info(f"Getting sent messages for pager {pager_id}")
        
        # Validate device exists
        device = self.db.get(PagerDevice, pager_id)
        if not device:
            raise ToolError(
                f"Pager device '{pager_id}' not found",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Query messages
        with self.db.get_session() as session:
            query = session.query(PagerMessage).filter(
                PagerMessage.sender_id == pager_id
            )
            
            if not include_expired:
                query = query.filter(PagerMessage.expires_at > utc_now())
                
            # Sort by sent time descending (newest first)
            query = query.order_by(PagerMessage.sent_at.desc())
            
            messages = query.all()
            message_list = [msg.to_dict() for msg in messages]
            
        return {
            "messages": message_list,
            "count": len(message_list),
            "pager_id": pager_id,
            "pager_name": device.name,
            "filters": {
                "include_expired": include_expired
            }
        }

    def _mark_message_read(self, message_id: str) -> Dict[str, Any]:
        """
        Mark a message as read.
        
        Args:
            message_id: ID of the message to mark as read
            
        Returns:
            Dict containing the updated message
            
        Raises:
            ToolError: If message not found
        """
        self.logger.info(f"Marking message {message_id} as read")
        
        # Get the message
        message = self.db.get(PagerMessage, message_id)
        if not message:
            raise ToolError(
                f"Message with ID '{message_id}' not found",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Update message
        message.read = True
        message.read_at = utc_now()
        
        # Update recipient device's last active time
        recipient = self.db.get(PagerDevice, message.recipient_id)
        if recipient:
            recipient.last_active = utc_now()
            self.db.update(recipient)
        
        # Save changes
        try:
            self.db.update(message)
            self.logger.info(f"Marked message {message_id} as read")
        except Exception as e:
            self.logger.error(f"Error updating message: {e}")
            raise ToolError(
                f"Failed to update message: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        return {
            "message": message.to_dict(),
            "status": "Message marked as read"
        }

    def _get_devices(self, active_only: bool = True) -> Dict[str, Any]:
        """
        Get list of pager devices.
        
        Args:
            active_only: Only return active devices
            
        Returns:
            Dict containing list of devices
        """
        self.logger.info(f"Getting pager devices (active_only={active_only})")
        
        # Query devices
        filters = []
        if active_only:
            filters.append(PagerDevice.active.is_(True))
            
        devices = self.db.query(PagerDevice, *filters)
        device_list = [device.to_dict() for device in devices]
        
        # Sort by last active descending
        device_list.sort(key=lambda x: x.get('last_active', ''), reverse=True)
        
        return {
            "devices": device_list,
            "count": len(device_list),
            "filters": {
                "active_only": active_only
            }
        }

    def _deactivate_device(self, pager_id: str) -> Dict[str, Any]:
        """
        Deactivate a pager device.
        
        Args:
            pager_id: ID of the device to deactivate
            
        Returns:
            Dict containing confirmation
            
        Raises:
            ToolError: If device not found
        """
        self.logger.info(f"Deactivating pager device {pager_id}")
        
        # Get the device
        device = self.db.get(PagerDevice, pager_id)
        if not device:
            raise ToolError(
                f"Pager device '{pager_id}' not found",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
            
        # Update device
        device.active = False
        
        # Save changes
        try:
            self.db.update(device)
            self.logger.info(f"Deactivated pager device {pager_id}")
        except Exception as e:
            self.logger.error(f"Error deactivating device: {e}")
            raise ToolError(
                f"Failed to deactivate device: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        return {
            "device": device.to_dict(),
            "message": f"Pager device '{device.name}' deactivated successfully"
        }

    def _cleanup_expired(self) -> Dict[str, Any]:
        """
        Clean up expired messages from the system.
        
        Returns:
            Dict containing cleanup statistics
        """
        self.logger.info("Cleaning up expired messages")
        
        # Count expired messages first
        with self.db.get_session() as session:
            expired_count = session.query(PagerMessage).filter(
                PagerMessage.expires_at <= utc_now()
            ).count()
            
            # Delete expired messages
            if expired_count > 0:
                session.query(PagerMessage).filter(
                    PagerMessage.expires_at <= utc_now()
                ).delete()
                session.commit()
                
        self.logger.info(f"Cleaned up {expired_count} expired messages")
        
        return {
            "expired_messages_removed": expired_count,
            "message": f"Cleaned up {expired_count} expired message(s)"
        }

    def _distill_message(self, content: str, max_length: int) -> str:
        """
        Use AI to distill a long message to fit pager constraints.
        
        Args:
            content: Original message content
            max_length: Maximum allowed length
            
        Returns:
            Distilled message content
        """
        prompt = f"""Distill the following message to fit within {max_length} characters while preserving all critical information. Focus on actionable content and key details. Remove unnecessary words but keep all important facts, numbers, names, and instructions.

Original message:
{content}

Provide ONLY the distilled message, no explanations or meta-text."""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            distilled = response.content.strip()
            
            # Ensure it fits within max_length
            if len(distilled) > max_length:
                distilled = distilled[:max_length-3] + "..."
                
            return distilled
            
        except Exception as e:
            self.logger.warning(f"AI distillation failed: {e}, truncating instead")
            # Fallback to simple truncation
            return content[:max_length-3] + "..."
    
    def _check_trust_status(self, trusting_device_id: str, sender_id: str, sender_fingerprint: str) -> str:
        """
        Check the trust status of a sender for a given device.
        
        Args:
            trusting_device_id: The device receiving the message
            sender_id: The device that sent the message
            sender_fingerprint: The fingerprint claimed by the sender
            
        Returns:
            Trust status: "trusted", "untrusted", "conflicted", or "first_contact"
        """
        # Query for existing trust relationship
        trusts = self.db.query(
            PagerTrust,
            PagerTrust.trusting_device_id == trusting_device_id,
            PagerTrust.trusted_device_id == sender_id
        )
        
        if not trusts or len(trusts) == 0:
            # First contact - add to trust store
            self._add_trust_relationship(trusting_device_id, sender_id, sender_fingerprint)
            return "first_contact"
        
        trust = trusts[0]
        
        if trust.trust_status == "revoked":
            return "revoked"
        
        if trust.trusted_fingerprint != sender_fingerprint:
            # Fingerprint mismatch! This is a security threat - reject the message completely
            trust.trust_status = "conflicted"
            self.db.update(trust)
            self.logger.error(
                f"SECURITY BREACH: Fingerprint mismatch for {sender_id} on device {trusting_device_id}. "
                f"Expected: {trust.trusted_fingerprint}, Got: {sender_fingerprint}. MESSAGE REJECTED."
            )
            raise ToolError(
                f"MESSAGE DELIVERY FAILED: Device {sender_id} fingerprint mismatch detected! "
                f"This could indicate an impersonation attempt. The message has been rejected for security. "
                f"If this device legitimately changed, the recipient must use 'revoke_trust' for device {sender_id} "
                f"and then you can send a new message to re-establish trust.",
                ErrorCode.TOOL_PERMISSION_DENIED
            )
        
        # Update last verified time
        trust.last_verified = utc_now()
        self.db.update(trust)
        
        return "trusted"
    
    def _add_trust_relationship(self, trusting_device_id: str, trusted_device_id: str, trusted_fingerprint: str) -> None:
        """
        Add a new trust relationship (TOFU - Trust on First Use).
        
        Args:
            trusting_device_id: The device that trusts
            trusted_device_id: The device being trusted
            trusted_fingerprint: The fingerprint to trust
        """
        # Get sender device info if available
        sender = self.db.get(PagerDevice, trusted_device_id)
        
        trust = PagerTrust(
            id=f"TRUST-{uuid.uuid4().hex[:8].upper()}",
            user_id=self.user_id,
            trusting_device_id=trusting_device_id,
            trusted_device_id=trusted_device_id,
            trusted_fingerprint=trusted_fingerprint,
            trusted_name=sender.name if sender else "Unknown",
            first_seen=utc_now(),
            last_verified=utc_now(),
            trust_status="trusted"
        )
        
        try:
            self.db.add(trust)
            self.logger.info(f"Added trust relationship: {trusting_device_id} trusts {trusted_device_id}")
        except Exception as e:
            self.logger.warning(f"Failed to add trust relationship: {e}")
    
    def _list_trusted_devices(self, pager_id: str) -> Dict[str, Any]:
        """
        List all devices trusted by a specific pager.
        
        Args:
            pager_id: ID of the pager device
            
        Returns:
            Dict containing list of trusted devices
        """
        device = self.db.get(PagerDevice, pager_id)
        if not device:
            raise ToolError(
                f"Pager device '{pager_id}' not found",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
        
        trusts = self.db.query(
            PagerTrust,
            PagerTrust.trusting_device_id == pager_id
        )
        
        trust_list = []
        for trust in trusts:
            trust_list.append({
                "trusted_device_id": trust.trusted_device_id,
                "trusted_name": trust.trusted_name,
                "trusted_fingerprint": trust.trusted_fingerprint,
                "first_seen": format_datetime(trust.first_seen, "date_time", get_default_timezone()),
                "last_verified": format_datetime(trust.last_verified, "date_time", get_default_timezone()),
                "trust_status": trust.trust_status
            })
        
        return {
            "pager_id": pager_id,
            "pager_name": device.name,
            "trusted_devices": trust_list,
            "count": len(trust_list)
        }
    
    def _revoke_trust(self, pager_id: str, untrusted_device_id: str) -> Dict[str, Any]:
        """
        Revoke trust for a specific device.
        
        Args:
            pager_id: ID of the pager revoking trust
            untrusted_device_id: ID of the device to untrust
            
        Returns:
            Dict with revocation confirmation
        """
        trusts = self.db.query(
            PagerTrust,
            PagerTrust.trusting_device_id == pager_id,
            PagerTrust.trusted_device_id == untrusted_device_id
        )
        
        if not trusts or len(trusts) == 0:
            raise ToolError(
                f"No trust relationship found between {pager_id} and {untrusted_device_id}",
                ErrorCode.TOOL_RESOURCE_NOT_FOUND
            )
        
        trust = trusts[0]
        
        # Delete the trust relationship entirely to allow fresh start
        self.db.delete(trust)
        self.logger.info(f"Revoked trust: {pager_id} no longer trusts {untrusted_device_id}")
        
        return {
            "message": f"Trust revoked for device {untrusted_device_id}. They can message again to establish new trust.",
            "pager_id": pager_id,
            "untrusted_device_id": untrusted_device_id
        }
    
    def _get_device_location(self) -> Dict[str, Any]:
        """
        Get the current device location.
        
        Returns:
            Dict with location information including coordinates and address
        """
        # In a real implementation, this would use device GPS or IP geolocation
        # For this simulation, we'll generate realistic location data
        import random
        
        # Simulate some common locations
        locations = [
            {
                "lat": 34.6937,
                "lng": 135.5023,
                "address": "1-1 Kitahama, Chuo-ku, Osaka",
                "description": "Near Osaka City Hall"
            },
            {
                "lat": 35.6762,
                "lng": 139.6503,
                "address": "2-8-1 Nishi-Shinjuku, Tokyo",
                "description": "Tokyo Metropolitan Building"
            },
            {
                "lat": 40.7128,
                "lng": -74.0060,
                "address": "City Hall Park, New York, NY",
                "description": "Near City Hall"
            },
            {
                "lat": 37.7749,
                "lng": -122.4194,
                "address": "1 Dr Carlton B Goodlett Pl, San Francisco, CA",
                "description": "San Francisco City Hall"
            }
        ]
        
        # Pick a random location for simulation
        location = random.choice(locations)
        
        # Add timestamp
        location["timestamp"] = utc_now().isoformat()
        location["accuracy_meters"] = random.randint(5, 50)
        
        return location
    
    def _send_location(
        self,
        sender_id: str,
        recipient_id: str,
        priority: Optional[int] = 1,  # Default to high priority for location pins
        message: Optional[str] = None,
        device_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a location pin message from one pager to another.
        
        Args:
            sender_id: ID of the sending pager
            recipient_id: ID of the receiving pager
            priority: Message priority (default 1=high for location pins)
            message: Optional brief message (max 50 chars)
            device_secret: Device secret for authentication
            
        Returns:
            Dict containing the sent location message
        """
        self.logger.info(f"Sending location pin from {sender_id} to {recipient_id}")
        
        # Get current location
        location_data = self._get_device_location()
        
        # Format location as JSON string for storage
        location_json = json.dumps({
            "lat": location_data["lat"],
            "lng": location_data["lng"],
            "accuracy_meters": location_data["accuracy_meters"]
        })
        
        # Create location message content
        content = f"📍 Location Pin: {location_data['address']}"
        if location_data.get('description'):
            content += f" ({location_data['description']})"
        
        # Add optional message if provided
        if message:
            # Enforce brief message limit for location pins
            if len(message) > 50:
                raise ToolError(
                    f"Location pin message too long: {len(message)} characters (max 50)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            content += f"\nNote: {message}"
        
        # Add coordinates for technical reference
        content += f"\n[{location_data['lat']:.4f}, {location_data['lng']:.4f}]"
        
        # Use the existing send_message method with location data
        return self._send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            priority=priority,
            location=location_json,
            expiry_hours=6,  # Location pins expire faster (6 hours)
            device_secret=device_secret
        )