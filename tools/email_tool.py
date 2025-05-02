"""
Email Tool - Email account access and management via IMAP/SMTP.

This tool provides a clean interface to email functionality, with a focus on
allowing the LLM to intelligently categorize and handle emails based on content
rather than rigid pattern matching.
"""
import email
import email.header
import email.message
import email.parser
import email.utils
import imaplib
import logging
import os
import re
import smtplib
import ssl
import uuid
from datetime import datetime
from email.message import EmailMessage, Message
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field
from errors import ErrorCode, ToolError, error_context
from tools.repo import Tool
from config.registry import registry

# Define configuration class for EmailTool
class EmailToolConfig(BaseModel):
    """Configuration for the email_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    imap_server: str = Field(
        default="mi3-ts111.a2hosting.com",
        description="IMAP server hostname"
    )
    imap_port: int = Field(
        default=993,
        description="IMAP server port (typically 993 for SSL/TLS)"
    )
    smtp_server: str = Field(
        default="mi3-ts111.a2hosting.com",
        description="SMTP server hostname (often same as IMAP server)"
    )
    smtp_port: int = Field(
        default=465,
        description="SMTP server port (typically 465 for SSL/TLS)"
    )
    email_address: str = Field(
        default="taylor@rocketcitywindowcleaning.com",
        description="Email address to use for IMAP/SMTP connections"
    )
    use_ssl: bool = Field(
        default=True,
        description="Whether to use SSL/TLS for connections"
    )
    max_emails_to_fetch: int = Field(
        default=50,
        description="Maximum number of emails to fetch in a single request"
    )
    max_preview_length: int = Field(
        default=10,
        description="Maximum length of email body preview text"
    )
    default_folders: Dict[str, str] = Field(
        default={
            "inbox": "INBOX", 
            "sent": "Sent", 
            "drafts": "Drafts", 
            "trash": "Trash"
        },
        description="Default folder names mapping"
    )

# Register with registry
registry.register("email_tool", EmailToolConfig)


class EmailTool(Tool):
    """
    Tool for accessing and managing email through IMAP/SMTP protocols.
    
    This implementation focuses on:
    1. Loading email content into context for LLM categorization
    2. Session state management for referencing emails
    3. Progressive loading for efficiency
    4. Clean, focused API for common email operations
    """
    
    name = "email_tool"
    description = """
    Email management tool that provides access to email accounts via IMAP/SMTP protocols. 
    Use this tool to read, search, send, and manage emails.
    
    OPERATIONS:
    - get_emails: Retrieve emails from specified folder with options for filtering and content loading
      Parameters:
        folder (optional, default="INBOX"): Email folder to access
        unread_only (optional, default=False): Set to True to only return unread emails
        load_content (optional, default=True): Set to True to load full email content
        sender (optional): Filter by sender email or name
        max_emails (optional, default=20): Maximum number of emails to return
    
    - get_email_content: Get full content of a specific email
      Parameters:
        email_id (required): UUID of the email to retrieve
        folder (optional, default="INBOX"): Email folder containing the email
    
    - mark_as_read: Mark an email as read
      Parameters:
        email_id (required): UUID of the email to mark
        folder (optional, default="INBOX"): Email folder containing the email
    
    - mark_as_unread: Mark an email as unread
      Parameters:
        email_id (required): UUID of the email to mark
        folder (optional, default="INBOX"): Email folder containing the email
    
    - delete_email: Delete an email
      Parameters:
        email_id (required): UUID of the email to delete
        folder (optional, default="INBOX"): Email folder containing the email
    
    - move_email: Move an email to another folder
      Parameters:
        email_id (required): UUID of the email to move
        destination_folder (required): Folder to move the email to
        folder (optional, default="INBOX"): Source folder containing the email
    
    - send_email: Send a new email
      Parameters:
        to (required): Recipient email address(es)
        subject (required): Email subject
        body (required): Email body content
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
    
    - reply_to_email: Reply to an existing email
      Parameters:
        email_id (required): UUID of the email to reply to
        body (required): Reply content
        folder (optional, default="INBOX"): Email folder containing the email
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
        
    - create_draft: Create a draft email without sending
      Parameters:
        to (required): Recipient email address(es)
        subject (required): Email subject
        body (required): Email body content
        cc (optional): CC recipient(s)
        bcc (optional): BCC recipient(s)
    
    - search_emails: Search emails with various criteria
      Parameters:
        folder (optional, default="INBOX"): Email folder to search in
        sender (optional): Sender email or name to search for
        subject (optional): Subject text to search for
        start_date (optional): Start date for range search (DD-Mon-YYYY format)
        end_date (optional): End date for range search (DD-Mon-YYYY format)
        unread_only (optional, default=False): Set to True to only return unread emails
        load_content (optional, default=True): Set to True to load full email content
        max_emails (optional, default=20): Maximum number of emails to return
        
    - list_folders: List available email folders
      Parameters: None
      
    - mark_for_later_reply: Mark an email to be replied to later in the conversation
      Parameters:
        email_id (required): UUID of the email to mark
        
    - get_emails_for_later_reply: Get list of emails marked for later reply
      Parameters: None
    
    USAGE NOTES:
    - Emails are loaded with full content by default to enable intelligent categorization
    - The LLM should categorize emails into groups like: from humans, priority, notifications, newsletters
    - Use the email_id to reference specific emails throughout the conversation
    - For handling multiple emails efficiently, process them by category
    - Mark emails for later reply to keep track of emails the user wants to address during the conversation
    """
    
    def __init__(self):
        """Initialize the email tool with configuration."""
        super().__init__()
        
        # Import config when needed (avoids circular imports)
        from config import config
        
        self.imap_server = config.email_tool.imap_server
        self.imap_port = config.email_tool.imap_port
        self.smtp_server = config.email_tool.smtp_server
        self.smtp_port = config.email_tool.smtp_port
        self.email_address = config.email_tool.email_address
        self.password = config.email_password
        self.use_ssl = config.email_tool.use_ssl
        
        # Session state
        self.connection = None
        self.selected_folder = None
        
        # In-memory email reference mapping - provides session stability
        self.uuid_to_msgid = {}
        self.msgid_to_uuid = {}
        
        # Session tracking - emails flagged for later reply
        self.emails_for_later_reply = set()
        
        # Set fetch limits
        self.default_max_emails = 20
        
        # Create data directory
        self.data_dir = os.path.join("data", "tools", "email_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("EmailTool initialized")
    
    def _is_connection_alive(self) -> bool:
        """
        Check if the IMAP connection is still alive and responsive.
        
        Returns:
            True if connection is alive, False otherwise
        """
        if not self.connection:
            return False
            
        try:
            # Try a NOOP command to check if connection is still responsive
            status, response = self.connection.noop()
            return status == 'OK'
        except Exception as e:
            self.logger.warning(f"IMAP connection check failed: {e}")
            self.connection = None
            return False
    
    def _connect(self) -> bool:
        """
        Connect to the IMAP server if not already connected or if connection is dead.
        
        Returns:
            True if connection succeeded, False otherwise
        """
        # Check if already connected and connection is alive
        if self.connection and self._is_connection_alive():
            # Already connected and alive
            return True
            
        # Reset connection state
        self.connection = None
        self.selected_folder = None
            
        try:
            if self.use_ssl:
                self.connection = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            else:
                self.connection = imaplib.IMAP4(self.imap_server, self.imap_port)
            
            # Login
            self.connection.login(self.email_address, self.password)
            self.logger.info(f"Connected to IMAP server {self.imap_server}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IMAP server: {e}")
            self.connection = None
            return False
    
    def _disconnect(self) -> None:
        """Close the IMAP connection if open."""
        if self.connection:
            try:
                self.connection.logout()
                self.logger.info("Disconnected from IMAP server")
            except Exception as e:
                self.logger.error(f"Error disconnecting from IMAP server: {e}")
            finally:
                self.connection = None
                self.selected_folder = None
    
    def _ensure_connected(self) -> bool:
        """
        Ensure the IMAP connection is alive, reconnecting if necessary.
        
        Returns:
            True if connection is established, False if connection failed
        """
        # First check if the connection is alive
        if self._is_connection_alive():
            return True
            
        # If we get here, we need to connect/reconnect
        return self._connect()
    
    def _select_folder(self, folder_name: str) -> bool:
        """
        Select a folder/mailbox.
        
        Args:
            folder_name: Name of the folder to select
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure connection is established
        if not self._ensure_connected():
            return False
        
        # No need to reselect if already on this folder
        if folder_name == self.selected_folder:
            return True
        
        try:
            status, response = self.connection.select(folder_name)
            if status != 'OK':
                self.logger.warning(f"Failed to select folder '{folder_name}': {response}")
                return False
                
            self.selected_folder = folder_name
            self.logger.info(f"Selected mailbox '{folder_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to select mailbox '{folder_name}': {e}")
            # Connection might have been lost, try to reconnect and retry once
            if self._connect():
                try:
                    status, response = self.connection.select(folder_name)
                    if status == 'OK':
                        self.selected_folder = folder_name
                        self.logger.info(f"Successfully selected mailbox '{folder_name}' after reconnection")
                        return True
                except Exception:
                    pass
            return False
    
    def _register_email(self, message_id: int) -> str:
        """
        Register an email message ID and get a session-stable UUID for it.
        
        Args:
            message_id: IMAP message ID
            
        Returns:
            UUID string for reference
        """
        # Check if we already have a UUID for this message ID
        if message_id in self.msgid_to_uuid:
            return self.msgid_to_uuid[message_id]
        
        # Create a new UUID
        email_uuid = str(uuid.uuid4())
        
        # Store the mapping both ways
        self.uuid_to_msgid[email_uuid] = message_id
        self.msgid_to_uuid[message_id] = email_uuid
        
        return email_uuid
    
    def _get_message_id(self, email_uuid: str) -> Optional[int]:
        """
        Get the message ID for a UUID.
        
        Args:
            email_uuid: UUID string
            
        Returns:
            IMAP message ID or None if not found
        """
        return self.uuid_to_msgid.get(email_uuid)
    
    def _decode_header(self, header: str) -> str:
        """
        Decode an email header with proper handling of character encodings.
        
        Args:
            header: Raw header string
            
        Returns:
            Decoded header string
        """
        if not header:
            return ""
        
        decoded_parts = []
        
        for decoded_header, charset in email.header.decode_header(header):
            if isinstance(decoded_header, bytes):
                if charset:
                    try:
                        decoded_parts.append(decoded_header.decode(charset))
                    except (LookupError, UnicodeDecodeError):
                        try:
                            decoded_parts.append(decoded_header.decode("utf-8"))
                        except UnicodeDecodeError:
                            decoded_parts.append(decoded_header.decode("latin1", errors="replace"))
                else:
                    try:
                        decoded_parts.append(decoded_header.decode("utf-8"))
                    except UnicodeDecodeError:
                        decoded_parts.append(decoded_header.decode("latin1", errors="replace"))
            else:
                decoded_parts.append(str(decoded_header))
        
        return " ".join(decoded_parts)
    
    def _get_email_body(self, msg: Message) -> Dict[str, Any]:
        """
        Extract the body content and attachment info from a message.
        
        Args:
            msg: Email message object
            
        Returns:
            Dictionary with body text and attachment info
        """
        result = {
            "text": "",
            "has_attachments": False,
            "attachments": []
        }
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # Handle attachments - detect but don't download
                if "attachment" in content_disposition:
                    result["has_attachments"] = True
                    filename = part.get_filename()
                    if filename:
                        attachment_info = {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(part.get_payload(decode=False)) if part.get_payload() else 0
                        }
                        result["attachments"].append(attachment_info)
                    continue
                
                # Handle text parts
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            text = payload.decode(charset)
                        except UnicodeDecodeError:
                            try:
                                text = payload.decode("utf-8")
                            except UnicodeDecodeError:
                                text = payload.decode("latin1", errors="replace")
                        
                        result["text"] += text
        else:
            # Non-multipart - get the payload directly
            content_type = msg.get_content_type()
            
            if content_type == "text/plain":
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    try:
                        result["text"] = payload.decode(charset)
                    except UnicodeDecodeError:
                        try:
                            result["text"] = payload.decode("utf-8")
                        except UnicodeDecodeError:
                            result["text"] = payload.decode("latin1", errors="replace")
        
        return result
    
    def _get_message_flags(self, message_id: int) -> List[str]:
        """
        Get the flags for a message.
        
        Args:
            message_id: IMAP message ID
            
        Returns:
            List of flag strings
        """
        if not self._ensure_connected():
            return []
            
        try:
            # Fetch the flags
            typ, data = self.connection.fetch(str(message_id), "(FLAGS)")
            if typ != "OK" or not data or not data[0]:
                return []
                
            # Parse the flags
            flags_str = data[0].decode("utf-8")
            flags = []
            
            if "\\Seen" in flags_str:
                flags.append("read")
            else:
                flags.append("unread")
                
            if "\\Flagged" in flags_str:
                flags.append("flagged")
                
            if "\\Answered" in flags_str:
                flags.append("answered")
                
            if "\\Draft" in flags_str:
                flags.append("draft")
                
            return flags
        except Exception as e:
            self.logger.error(f"Error getting flags for message {message_id}: {e}")
            return []
    
    def _search_messages(self, criteria: str) -> List[int]:
        """
        Search for messages in the selected folder.
        
        Args:
            criteria: IMAP search criteria string
            
        Returns:
            List of message IDs matching the criteria
        """
        if not self._ensure_connected():
            return []
            
        try:
            # Execute the search
            typ, data = self.connection.search(None, criteria)
            if typ != "OK" or not data or not data[0]:
                return []
                
            # Parse message IDs
            message_ids = data[0].decode("utf-8").split()
            return list(map(int, message_ids))
        except Exception as e:
            self.logger.error(f"Error searching messages with criteria '{criteria}': {e}")
            return []
    
    def _fetch_message_headers(self, message_ids: List[int], limit: int = None, load_content: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch headers for a list of message IDs.
        
        Args:
            message_ids: List of IMAP message IDs
            limit: Maximum number of messages to fetch
            load_content: Whether to load the full message content
            
        Returns:
            List of email header dictionaries (with content if requested)
        """
        if not message_ids or not self._ensure_connected():
            return []
            
        if limit and len(message_ids) > limit:
            message_ids = message_ids[-limit:]  # Take the most recent messages
            
        email_items = []
        
        for msg_id in message_ids:
            try:
                if load_content:
                    # Fetch the full message
                    typ, data = self.connection.fetch(str(msg_id), "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        continue
                        
                    # Parse the full message
                    email_data = data[0][1]
                    msg = email.message_from_bytes(email_data)
                    
                    # Extract body content
                    body = self._get_email_body(msg)
                    
                    # Register the message with a UUID
                    email_uuid = self._register_email(msg_id)
                    
                    # Get flags
                    flags = self._get_message_flags(msg_id)
                    
                    # Create the result with content
                    email_info = {
                        "id": email_uuid,
                        "from": self._decode_header(msg.get("From", "")),
                        "to": self._decode_header(msg.get("To", "")),
                        "cc": self._decode_header(msg.get("Cc", "")),
                        "subject": self._decode_header(msg.get("Subject", "")),
                        "date": self._decode_header(msg.get("Date", "")),
                        "body_text": body["text"],
                        "has_attachments": body["has_attachments"],
                        "flags": flags,
                    }
                    
                    # Add attachment information if present
                    if body["has_attachments"]:
                        email_info["attachments"] = body["attachments"]
                else:
                    # Fetch just the headers
                    typ, data = self.connection.fetch(str(msg_id), "(BODY.PEEK[HEADER])")
                    if typ != "OK" or not data or not data[0]:
                        continue
                        
                    header_data = data[0][1]
                    if not header_data:
                        continue
                        
                    # Parse headers
                    parser = email.parser.BytesParser()
                    headers = parser.parsebytes(header_data, headersonly=True)
                    
                    # Register the message with a UUID
                    email_uuid = self._register_email(msg_id)
                    
                    # Get flags
                    flags = self._get_message_flags(msg_id)
                    
                    # Create header dictionary
                    email_info = {
                        "id": email_uuid,
                        "from": self._decode_header(headers.get("From", "")),
                        "subject": self._decode_header(headers.get("Subject", "")),
                        "date": self._decode_header(headers.get("Date", "")),
                        "flags": flags
                    }
                
                email_items.append(email_info)
            except Exception as e:
                self.logger.error(f"Error fetching {'full message' if load_content else 'headers'} for message {msg_id}: {e}")
        
        return email_items
    
    def _set_flag(self, email_id: str, flag: str, value: bool) -> bool:
        """
        Set or unset a flag on an email.
        
        Args:
            email_id: Email UUID
            flag: Flag name ('\\Seen', '\\Flagged', etc.)
            value: True to set, False to unset
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
            
        message_id = self._get_message_id(email_id)
        if not message_id:
            return False
            
        try:
            # Set or unset the flag
            command = "+FLAGS" if value else "-FLAGS"
            self.connection.store(str(message_id), command, flag)
            return True
        except Exception as e:
            self.logger.error(f"Error setting flag {flag} for message {message_id}: {e}")
            return False
    
    def _parse_email_addresses(self, email_param: Optional[str]) -> Optional[str]:
        """
        Parse email address string that might be in JSON array format.
        
        Args:
            email_param: Email address string that might be a JSON array
            
        Returns:
            Properly formatted email string or None if input was None
        """
        if not email_param:
            return None
            
        # Check if the input might be a JSON array
        if email_param.startswith('[') and email_param.endswith(']'):
            try:
                import json
                emails = json.loads(email_param)
                # Ensure we have a list of strings
                if isinstance(emails, list):
                    # Filter empty values and join with commas
                    valid_emails = [e.strip() for e in emails if e and isinstance(e, str) and e.strip()]
                    if valid_emails:
                        return ", ".join(valid_emails)
                    return None
            except json.JSONDecodeError:
                # If not valid JSON, continue with original value
                pass
                
        # Process as comma-separated string, filtering empty addresses
        valid_emails = [addr.strip() for addr in email_param.split(",") if addr and addr.strip()]
        if valid_emails:
            return ", ".join(valid_emails)
        return None
    
    def run(
        self,
        operation: str,
        folder: str = "INBOX",
        email_id: Optional[str] = None,
        unread_only: bool = False,
        load_content: bool = True,
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_emails: Optional[int] = None,
        to: Optional[str] = None,
        body: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        destination_folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an email operation.
        
        Args:
            operation: The operation to perform (get_emails, get_email_content, etc.)
            folder: Email folder to access (default: "INBOX")
            email_id: UUID of a specific email
            unread_only: Whether to only return unread emails
            load_content: Whether to load full email content
            sender: Sender email address or name to search for
            subject: Subject text to search for
            start_date: Start date for range search (DD-Mon-YYYY format)
            end_date: End date for range search (DD-Mon-YYYY format)
            max_emails: Maximum number of emails to return
            to: Recipient for sending emails
            body: Body text for sending emails
            cc: CC recipients for sending emails
            bcc: BCC recipients for sending emails
            destination_folder: Destination folder for move_email
            
        Returns:
            Dictionary with operation results
            
        Raises:
            ToolError: If the operation is invalid or fails
        """
        with error_context(
            component_name=self.name,
            operation=operation,
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Ensure we're connected to the server
            if not self._ensure_connected():
                raise ToolError(
                    "Failed to connect to email server",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            # Make sure we're looking at the right folder for message operations
            if folder and folder != self.selected_folder and operation not in ["list_folders", "send_email", "create_draft", "mark_for_later_reply", "get_emails_for_later_reply"]:
                if not self._select_folder(folder):
                    raise ToolError(
                        f"Failed to select folder '{folder}'",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            # Set default max_emails if not provided and convert to int if it's a string
            if max_emails is None:
                max_emails = self.default_max_emails
            elif isinstance(max_emails, str):
                max_emails = int(max_emails)
            
            # Handle each operation type
            if operation == "get_emails":
                # Build search criteria
                search_parts = []
                
                if unread_only:
                    search_parts.append("UNSEEN")
                
                if sender:
                    search_parts.append(f'FROM "{sender}"')
                
                if subject:
                    search_parts.append(f'SUBJECT "{subject}"')
                
                if start_date:
                    search_parts.append(f'SINCE "{start_date}"')
                
                if end_date:
                    search_parts.append(f'BEFORE "{end_date}"')
                
                # Default to ALL if no criteria specified
                search_criteria = " ".join(search_parts) if search_parts else "ALL"
                
                # Execute search and fetch emails
                message_ids = self._search_messages(search_criteria)
                emails = self._fetch_message_headers(message_ids, max_emails, load_content)
                
                # LLM handling notes for categorization and summarization
                categorization_note = """
                CATEGORIZATION INSTRUCTIONS:
                
                Group these emails into the following categories:
                1. "humans" - Emails from real people requiring personal attention
                2. "priority" - Important emails needing immediate action
                3. "notifications" - Automated notifications from services
                4. "newsletters" - Marketing and newsletter emails
                
                For each email, consider:
                - Sender address and name patterns
                - Subject line keywords
                - Content patterns and formality
                - Importance to the recipient
                
                Provide a summary of how many emails are in each category before showing details.
                When the user asks to see emails from a specific category, show a numbered list with
                brief summaries of each email in that category.
                
                Example output:
                "You have 3 emails from humans, 2 priority emails, 4 notifications, and 7 newsletters."
                
                When showing emails in a category:
                "Here are your emails from humans:
                1. John Smith - Meeting tomorrow at 2pm
                2. Sarah Lee - Question about the project timeline
                3. Mike Johnson - Kids soccer practice cancelled"
                """
                
                return {
                    "emails": emails,
                    "total": len(message_ids),
                    "showing": len(emails),
                    "content_loaded": load_content,
                    "categorization_note": categorization_note if load_content else "Load content to enable intelligent categorization"
                }
            
            elif operation == "get_email_content":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for get_email_content operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Get the message ID from the UUID
                message_id = self._get_message_id(email_id)
                if not message_id:
                    raise ToolError(
                        f"Email with ID {email_id} not found or no longer available",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Fetch the full message
                    typ, data = self.connection.fetch(str(message_id), "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        raise ToolError(
                            f"Failed to fetch email content for ID {email_id}",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                    
                    # Parse the message
                    email_data = data[0][1]
                    msg = email.message_from_bytes(email_data)
                    
                    # Extract body content
                    body = self._get_email_body(msg)
                    
                    # Get the flags
                    flags = self._get_message_flags(message_id)
                    
                    # Mark the email as read if it wasn't already
                    if "unread" in flags:
                        self._set_flag(email_id, "\\Seen", True)
                        # Update flags to reflect the change
                        flags = [flag for flag in flags if flag != "unread"]
                        flags.append("read")
                    
                    # Create the result
                    result = {
                        "id": email_id,
                        "from": self._decode_header(msg.get("From", "")),
                        "to": self._decode_header(msg.get("To", "")),
                        "cc": self._decode_header(msg.get("Cc", "")),
                        "subject": self._decode_header(msg.get("Subject", "")),
                        "date": self._decode_header(msg.get("Date", "")),
                        "body_text": body["text"],
                        "has_attachments": body["has_attachments"],
                        "flags": flags
                    }
                    
                    # Add attachment information if present
                    if body["has_attachments"]:
                        result["attachments"] = body["attachments"]
                    
                    return result
                except Exception as e:
                    raise ToolError(
                        f"Error fetching email content: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "mark_as_read":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for mark_as_read operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Set the \Seen flag
                success = self._set_flag(email_id, "\\Seen", True)
                
                if not success:
                    raise ToolError(
                        f"Failed to mark email {email_id} as read",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
                
                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_as_read"
                }
            
            elif operation == "mark_as_unread":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for mark_as_unread operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Remove the \Seen flag
                success = self._set_flag(email_id, "\\Seen", False)
                
                if not success:
                    raise ToolError(
                        f"Failed to mark email {email_id} as unread",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
                
                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_as_unread"
                }
            
            elif operation == "delete_email":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for delete_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Get the message ID from the UUID
                message_id = self._get_message_id(email_id)
                if not message_id:
                    raise ToolError(
                        f"Email with ID {email_id} not found or no longer available",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Mark the message as deleted
                    self.connection.store(str(message_id), "+FLAGS", "\\Deleted")
                    
                    # Expunge the message
                    self.connection.expunge()
                    
                    # Remove from our mappings
                    if email_id in self.uuid_to_msgid:
                        del self.uuid_to_msgid[email_id]
                    
                    if message_id in self.msgid_to_uuid:
                        del self.msgid_to_uuid[message_id]
                    
                    # Remove from later reply set if present
                    if email_id in self.emails_for_later_reply:
                        self.emails_for_later_reply.remove(email_id)
                    
                    return {
                        "success": True,
                        "email_id": email_id,
                        "operation": "delete_email"
                    }
                except Exception as e:
                    raise ToolError(
                        f"Failed to delete email {email_id}: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "move_email":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for move_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not destination_folder:
                    raise ToolError(
                        "destination_folder is required for move_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Get the message ID from the UUID
                message_id = self._get_message_id(email_id)
                if not message_id:
                    raise ToolError(
                        f"Email with ID {email_id} not found or no longer available",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Try to use MOVE command if supported
                    move_supported = hasattr(self.connection, "move")
                    
                    if move_supported:
                        # Use the MOVE command
                        self.connection.move(str(message_id), destination_folder)
                    else:
                        # Fall back to copy and delete
                        # Copy to destination
                        self.connection.copy(str(message_id), destination_folder)
                        
                        # Mark as deleted
                        self.connection.store(str(message_id), "+FLAGS", "\\Deleted")
                        
                        # Expunge
                        self.connection.expunge()
                    
                    # Update our mappings
                    # We'll remove mappings since the message ID may change in the new folder
                    if email_id in self.uuid_to_msgid:
                        del self.uuid_to_msgid[email_id]
                    
                    if message_id in self.msgid_to_uuid:
                        del self.msgid_to_uuid[message_id]
                    
                    # Remove from later reply set if present
                    if email_id in self.emails_for_later_reply:
                        self.emails_for_later_reply.remove(email_id)
                    
                    return {
                        "success": True,
                        "email_id": email_id,
                        "destination": destination_folder,
                        "operation": "move_email"
                    }
                except Exception as e:
                    raise ToolError(
                        f"Failed to move email {email_id} to {destination_folder}: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "send_email":
                # Validate required parameters
                if not to:
                    raise ToolError(
                        "to is required for send_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not subject:
                    raise ToolError(
                        "subject is required for send_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not body:
                    raise ToolError(
                        "body is required for send_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Parse the email addresses before creating the message
                    parsed_to = self._parse_email_addresses(to)
                    if not parsed_to:
                        raise ToolError(
                            "No valid email addresses in 'to' field",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                    
                    parsed_cc = self._parse_email_addresses(cc) if cc else None
                    parsed_bcc = self._parse_email_addresses(bcc) if bcc else None
                    
                    # Create the message
                    msg = EmailMessage()
                    msg["Subject"] = subject
                    msg["From"] = self.email_address
                    msg["To"] = parsed_to
                    
                    if parsed_cc:
                        msg["Cc"] = parsed_cc
                    
                    if parsed_bcc:
                        msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Add Message-ID
                    domain = self.smtp_server.split(".", 1)[1] if "." in self.smtp_server else self.smtp_server
                    msg["Message-ID"] = email.utils.make_msgid(domain=domain)
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Build list of recipients
                    recipients = []
                    
                    # Use our helper method to parse the addresses
                    parsed_to = self._parse_email_addresses(to)
                    if parsed_to:
                        recipients.extend([addr.strip() for addr in parsed_to.split(",") if addr.strip()])
                    
                    parsed_cc = self._parse_email_addresses(cc)
                    if parsed_cc:
                        recipients.extend([addr.strip() for addr in parsed_cc.split(",") if addr.strip()])
                    
                    parsed_bcc = self._parse_email_addresses(bcc)
                    if parsed_bcc:
                        recipients.extend([addr.strip() for addr in parsed_bcc.split(",") if addr.strip()])
                    
                    # Connect to SMTP server
                    context = ssl.create_default_context()
                    
                    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                        # Login
                        server.login(self.email_address, self.password)
                        
                        # Send the email
                        server.send_message(msg)
                    
                    # Save to Sent folder
                    try:
                        self.connection.append("Sent", None, None, msg.as_bytes())
                    except Exception as e:
                        self.logger.warning(f"Failed to save to sent folder: {e}")
                    
                    return {
                        "success": True,
                        "to": to,
                        "subject": subject,
                        "operation": "send_email"
                    }
                except Exception as e:
                    raise ToolError(
                        f"Failed to send email: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "reply_to_email":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for reply_to_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not body:
                    raise ToolError(
                        "body is required for reply_to_email operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Get the message ID from the UUID
                message_id = self._get_message_id(email_id)
                if not message_id:
                    raise ToolError(
                        f"Email with ID {email_id} not found or no longer available",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Fetch the original message
                    typ, data = self.connection.fetch(str(message_id), "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        raise ToolError(
                            f"Failed to fetch original email for reply",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                    
                    # Parse the message
                    email_data = data[0][1]
                    original_msg = email.message_from_bytes(email_data)
                    
                    # Create reply message
                    msg = EmailMessage()
                    
                    # Set subject with Re: prefix if needed
                    original_subject = self._decode_header(original_msg.get("Subject", ""))
                    if original_subject.lower().startswith("re:"):
                        msg["Subject"] = original_subject
                    else:
                        msg["Subject"] = f"Re: {original_subject}"
                    
                    # Set From
                    msg["From"] = self.email_address
                    
                    # Set To (reply to sender by default)
                    reply_to = original_msg.get("Reply-To")
                    if reply_to:
                        msg["To"] = reply_to
                    else:
                        msg["To"] = original_msg.get("From", "")
                    
                    # Override To if specified
                    if to:
                        # Parse email addresses 
                        parsed_to = self._parse_email_addresses(to)
                        if parsed_to:
                            msg["To"] = parsed_to
                        else:
                            msg["To"] = original_msg.get("From", "")
                    
                    # Add CC/BCC if specified
                    if cc:
                        parsed_cc = self._parse_email_addresses(cc)
                        if parsed_cc:
                            msg["Cc"] = parsed_cc
                    
                    if bcc:
                        parsed_bcc = self._parse_email_addresses(bcc)
                        if parsed_bcc:
                            msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Set In-Reply-To and References headers for threading
                    message_id_header = original_msg.get("Message-ID")
                    if message_id_header:
                        msg["In-Reply-To"] = message_id_header
                        
                        # Set References
                        references = original_msg.get("References", "")
                        if references:
                            msg["References"] = f"{references} {message_id_header}"
                        else:
                            msg["References"] = message_id_header
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Connect to SMTP server
                    context = ssl.create_default_context()
                    
                    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                        # Login
                        server.login(self.email_address, self.password)
                        
                        # Send the email
                        server.send_message(msg)
                    
                    # Save to Sent folder
                    try:
                        self.connection.append("Sent", None, None, msg.as_bytes())
                    except Exception as e:
                        self.logger.warning(f"Failed to save to sent folder: {e}")
                    
                    # Mark as answered
                    self._set_flag(email_id, "\\Answered", True)
                    
                    # Remove from later reply set if present
                    if email_id in self.emails_for_later_reply:
                        self.emails_for_later_reply.remove(email_id)
                    
                    return {
                        "success": True,
                        "replied_to": email_id,
                        "subject": msg["Subject"],
                        "operation": "reply_to_email"
                    }
                except Exception as e:
                    raise ToolError(
                        f"Failed to reply to email: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "create_draft":
                # Validate required parameters
                if not to:
                    raise ToolError(
                        "to is required for create_draft operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not subject:
                    raise ToolError(
                        "subject is required for create_draft operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                if not body:
                    raise ToolError(
                        "body is required for create_draft operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                try:
                    # Create the message
                    msg = EmailMessage()
                    msg["Subject"] = subject
                    msg["From"] = self.email_address
                    # Parse email addresses
                    parsed_to = self._parse_email_addresses(to)
                    if parsed_to:
                        msg["To"] = parsed_to
                    else:
                        raise ToolError(
                            "No valid email addresses in 'to' field",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                    
                    if cc:
                        parsed_cc = self._parse_email_addresses(cc)
                        if parsed_cc:
                            msg["Cc"] = parsed_cc
                    
                    if bcc:
                        parsed_bcc = self._parse_email_addresses(bcc)
                        if parsed_bcc:
                            msg["Bcc"] = parsed_bcc
                    
                    # Set the date
                    msg["Date"] = email.utils.formatdate(localtime=True)
                    
                    # Set the content
                    msg.set_content(body)
                    
                    # Append to drafts folder with \Draft flag
                    self.connection.append("Drafts", "\\Draft", None, msg.as_bytes())
                    
                    return {
                        "success": True,
                        "to": to,
                        "subject": subject,
                        "operation": "create_draft"
                    }
                except Exception as e:
                    raise ToolError(
                        f"Failed to create draft: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "search_emails":
                # Build search criteria
                search_parts = []
                
                if unread_only:
                    search_parts.append("UNSEEN")
                
                if sender:
                    search_parts.append(f'FROM "{sender}"')
                
                if subject:
                    search_parts.append(f'SUBJECT "{subject}"')
                
                if start_date:
                    search_parts.append(f'SINCE "{start_date}"')
                
                if end_date:
                    search_parts.append(f'BEFORE "{end_date}"')
                
                # Must have at least one search criterion
                if not search_parts:
                    raise ToolError(
                        "At least one search criterion is required",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Combine criteria with AND
                search_criteria = " ".join(search_parts)
                
                # Execute search and fetch emails
                message_ids = self._search_messages(search_criteria)
                
                # Make sure max_emails is an integer
                if isinstance(max_emails, str):
                    max_emails = int(max_emails)
                    
                emails = self._fetch_message_headers(message_ids, max_emails, load_content)
                
                # LLM handling note for search results
                search_note = """
                For search results:
                1. Display emails in a clear, numbered list
                2. For each email, show: sender, date, subject, and a brief preview
                3. If content is loaded, provide a short summary of each email's purpose
                """
                
                return {
                    "emails": emails,
                    "total": len(message_ids),
                    "showing": len(emails),
                    "criteria": search_criteria,
                    "content_loaded": load_content,
                    "search_note": search_note
                }
            
            elif operation == "list_folders":
                try:
                    # Get list of folders
                    typ, folder_data = self.connection.list()
                    
                    if typ != "OK":
                        raise ToolError(
                            "Failed to retrieve folder list",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                    
                    folders = []
                    for item in folder_data:
                        if not item:
                            continue
                        
                        decoded_item = item.decode("utf-8")
                        
                        # Parse folder data
                        match = re.match(r'^\((?P<flags>.*?)\) "(?P<delimiter>.*?)" (?P<name>.+)$', decoded_item)
                        
                        if match:
                            flags = match.group("flags")
                            delimiter = match.group("delimiter")
                            folder_name = match.group("name")
                            
                            # Remove quotes if present
                            if folder_name.startswith('"') and folder_name.endswith('"'):
                                folder_name = folder_name[1:-1]
                            
                            folders.append({
                                "name": folder_name,
                                "flags": flags
                            })
                    
                    return {
                        "folders": folders,
                        "current_folder": self.selected_folder
                    }
                except Exception as e:
                    raise ToolError(
                        f"Error listing folders: {e}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            elif operation == "mark_for_later_reply":
                # Validate required parameters
                if not email_id:
                    raise ToolError(
                        "email_id is required for mark_for_later_reply operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Get the message ID from the UUID to verify it exists
                message_id = self._get_message_id(email_id)
                if not message_id:
                    raise ToolError(
                        f"Email with ID {email_id} not found or no longer available",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Add to later reply set
                self.emails_for_later_reply.add(email_id)
                
                return {
                    "success": True,
                    "email_id": email_id,
                    "operation": "mark_for_later_reply"
                }
            
            elif operation == "get_emails_for_later_reply":
                email_ids = list(self.emails_for_later_reply)
                
                # Get details for each email
                emails = []
                for eid in email_ids:
                    msgid = self._get_message_id(eid)
                    if msgid:
                        try:
                            # Fetch headers
                            typ, data = self.connection.fetch(str(msgid), "(BODY.PEEK[HEADER])")
                            if typ == "OK" and data and data[0]:
                                header_data = data[0][1]
                                parser = email.parser.BytesParser()
                                headers = parser.parsebytes(header_data, headersonly=True)
                                
                                # Get flags
                                flags = self._get_message_flags(msgid)
                                
                                # Create header dictionary
                                email_info = {
                                    "id": eid,
                                    "from": self._decode_header(headers.get("From", "")),
                                    "subject": self._decode_header(headers.get("Subject", "")),
                                    "date": self._decode_header(headers.get("Date", "")),
                                    "flags": flags
                                }
                                
                                emails.append(email_info)
                        except Exception as e:
                            self.logger.error(f"Error fetching headers for later reply email {eid}: {e}")
                
                # LLM handling note for later reply emails
                later_reply_note = """
                For emails marked for later reply:
                1. Present these emails as a numbered list
                2. Ask the user if they want to reply to any of them now
                3. If there are no emails marked for later reply, inform the user
                """
                
                return {
                    "emails": emails,
                    "count": len(emails),
                    "later_reply_note": later_reply_note
                }
            
            else:
                raise ToolError(
                    f"Unknown operation: {operation}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
    
    def __del__(self):
        """Ensure we disconnect when the object is destroyed."""
        self._disconnect()


def create_email_tool():
    """
    Factory function to create and initialize an EmailTool instance.
    
    Returns:
        Initialized EmailTool instance
    """
    return EmailTool()
