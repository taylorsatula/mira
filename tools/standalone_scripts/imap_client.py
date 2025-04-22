#!/usr/bin/env python3
"""
Interactive IMAP Client - Standalone utility for IMAP server interaction
"""
import argparse
import cmd
import email
import email.header
import email.parser
import email.message
import email.utils
import html
import imaplib
import json
import logging
import os
import re
import smtplib
import ssl
import tempfile
import subprocess
from email.message import Message
from typing import Any, Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("imap_client")


class IMAPClient:
    """IMAP Client for email interaction."""

    def __init__(
        self,
        host: str,
        port: int = 993,
        username: str = "",
        password: str = "",
        use_ssl: bool = True,
        smtp_port: int = 465,
    ):
        """
        Initialize IMAP Client.

        Args:
            host: IMAP server hostname
            port: IMAP server port
            username: IMAP account username
            password: IMAP account password
            use_ssl: Whether to use SSL for connection
            smtp_port: SMTP server port
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.connection = None
        self.selected_mailbox = None
        self.capabilities = []

        # SMTP settings for sending mail (use same host and credentials)
        self.smtp_host = host
        self.smtp_port = smtp_port

    def connect(self) -> bool:
        """
        Connect to the IMAP server.

        Returns:
            bool: True if connection succeeded, False otherwise
        """
        try:
            if self.use_ssl:
                self.connection = imaplib.IMAP4_SSL(self.host, self.port)
            else:
                self.connection = imaplib.IMAP4(self.host, self.port)

            # Get server capabilities
            typ, capabilities_data = self.connection.capability()
            if typ == "OK":
                # Parse capabilities from bytes to string to list
                capabilities_str = capabilities_data[0].decode("utf-8")
                self.capabilities = capabilities_str.split()
                logger.debug(f"Server capabilities: {self.capabilities}")

            # Login
            self.connection.login(self.username, self.password)
            logger.info(f"Successfully connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IMAP server: {e}")
            return False

    def disconnect(self) -> None:
        """Close connection to IMAP server."""
        if self.connection:
            try:
                self.connection.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.error(f"Error disconnecting from IMAP server: {e}")

    def list_mailboxes(self) -> List[Dict[str, str]]:
        """
        List all available mailboxes/folders.

        Returns:
            List of dictionaries containing mailbox details
        """
        if not self.connection:
            logger.error("Not connected to IMAP server")
            return []

        try:
            # Using list_with_quotes as indicated in the diagnosis
            typ, mailbox_data = self.connection.list()

            if typ != "OK":
                logger.error(f"Failed to list mailboxes: {typ}")
                return []

            mailboxes = []
            for item in mailbox_data:
                if not item:
                    continue

                decoded_item = item.decode("utf-8")

                # Parse the mailbox data
                # Format is typically: (FLAGS) "DELIMITER" "MAILBOX_NAME"
                match = re.match(
                    r'^\((?P<flags>.*?)\) "(?P<delimiter>.*?)" (?P<name>.+)$',
                    decoded_item,
                )

                if match:
                    flags = match.group("flags")
                    delimiter = match.group("delimiter")
                    mailbox_name = match.group("name")

                    # If name is quoted, remove the quotes
                    if mailbox_name.startswith('"') and mailbox_name.endswith('"'):
                        mailbox_name = mailbox_name[1:-1]

                    mailboxes.append(
                        {"name": mailbox_name, "delimiter": delimiter, "flags": flags}
                    )
                else:
                    logger.warning(f"Could not parse mailbox data: {decoded_item}")

            return mailboxes
        except Exception as e:
            logger.error(f"Error listing mailboxes: {e}")
            return []

    def select_mailbox(self, mailbox: str = "INBOX") -> int:
        """
        Select a mailbox/folder.

        Args:
            mailbox: Name of the mailbox to select

        Returns:
            Number of messages in the mailbox
        """
        if not self.connection:
            logger.error("Not connected to IMAP server")
            return 0

        try:
            typ, data = self.connection.select(mailbox)
            if typ != "OK":
                logger.error(f"Failed to select mailbox '{mailbox}': {typ}")
                return 0

            self.selected_mailbox = mailbox
            message_count = int(data[0])
            logger.info(f"Selected mailbox '{mailbox}' with {message_count} messages")
            return message_count
        except Exception as e:
            logger.error(f"Error selecting mailbox '{mailbox}': {e}")
            return 0

    def search_messages(
        self, criteria: str = "ALL", charset: str = "UTF-8"
    ) -> List[int]:
        """
        Search for messages in the selected mailbox.

        Args:
            criteria: IMAP search criteria
            charset: Character set for the search

        Returns:
            List of message IDs matching the criteria
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return []

        try:
            # Using standard_search as indicated in the diagnosis
            typ, data = self.connection.search(charset, criteria)
            if typ != "OK":
                logger.error(f"Search failed: {typ}")
                return []

            # Parse message IDs from search results
            if not data or not data[0]:
                return []

            message_ids = data[0].decode("utf-8").split()
            return [int(msg_id) for msg_id in message_ids]
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []

    def fetch_message_headers(self, message_id: int) -> Dict[str, Any]:
        """
        Fetch headers for a specific message.

        Args:
            message_id: ID of the message to fetch

        Returns:
            Dictionary containing parsed header information
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return {}

        try:
            # Using standard_fetch_peek as indicated in the diagnosis
            typ, data = self.connection.fetch(str(message_id), "(BODY.PEEK[HEADER])")
            if typ != "OK" or not data or not data[0]:
                logger.error(f"Failed to fetch message {message_id}: {typ}")
                return {}

            # Parse headers from the email
            header_data = data[0][1]
            if not header_data:
                return {}

            parser = email.parser.BytesParser()
            headers = parser.parsebytes(header_data, headersonly=True)

            # Format the output
            result = {
                "message_id": message_id,
                "subject": self._decode_header(headers.get("Subject", "")),
                "from": self._decode_header(headers.get("From", "")),
                "to": self._decode_header(headers.get("To", "")),
                "date": self._decode_header(headers.get("Date", "")),
                "content_type": headers.get("Content-Type", ""),
            }
            return result
        except Exception as e:
            logger.error(
                f"Error fetching message headers for message {message_id}: {e}"
            )
            return {}

    def fetch_message(self, message_id: int) -> Dict[str, Any]:
        """
        Fetch a complete message.

        Args:
            message_id: ID of the message to fetch

        Returns:
            Dictionary containing parsed message information
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return {}

        try:
            # Using full_fetch as indicated in the diagnosis
            typ, data = self.connection.fetch(str(message_id), "(RFC822)")
            if typ != "OK" or not data or not data[0]:
                logger.error(f"Failed to fetch message {message_id}: {typ}")
                return {}

            # Extract the email data
            email_data = data[0][1]
            if not email_data:
                return {}

            # Parse the email
            parsed_email = email.message_from_bytes(email_data)

            # Extract basic headers
            headers = {
                "message_id": message_id,
                "subject": self._decode_header(parsed_email.get("Subject", "")),
                "from": self._decode_header(parsed_email.get("From", "")),
                "to": self._decode_header(parsed_email.get("To", "")),
                "cc": self._decode_header(parsed_email.get("Cc", "")),
                "bcc": self._decode_header(parsed_email.get("Bcc", "")),
                "date": self._decode_header(parsed_email.get("Date", "")),
                "content_type": parsed_email.get("Content-Type", ""),
            }

            # Extract body content
            body = self._get_email_body(parsed_email)

            # Combine headers and body
            result = {
                **headers,
                "body": body,
                "raw_message": email_data,
                "raw_email": parsed_email,
            }

            return result
        except Exception as e:
            logger.error(f"Error fetching message {message_id}: {e}")
            return {}

    def mark_as_read(self, message_id: int) -> bool:
        """
        Mark a message as read.

        Args:
            message_id: ID of the message to mark

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Set the \Seen flag
            self.connection.store(str(message_id), "+FLAGS", "\\Seen")
            logger.info(f"Message {message_id} marked as read")
            return True
        except Exception as e:
            logger.error(f"Error marking message {message_id} as read: {e}")
            return False

    def mark_as_unread(self, message_id: int) -> bool:
        """
        Mark a message as unread.

        Args:
            message_id: ID of the message to mark

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Remove the \Seen flag
            self.connection.store(str(message_id), "-FLAGS", "\\Seen")
            logger.info(f"Message {message_id} marked as unread")
            return True
        except Exception as e:
            logger.error(f"Error marking message {message_id} as unread: {e}")
            return False

    def flag_message(self, message_id: int) -> bool:
        """
        Flag a message.

        Args:
            message_id: ID of the message to flag

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Set the \Flagged flag
            self.connection.store(str(message_id), "+FLAGS", "\\Flagged")
            logger.info(f"Message {message_id} flagged")
            return True
        except Exception as e:
            logger.error(f"Error flagging message {message_id}: {e}")
            return False

    def unflag_message(self, message_id: int) -> bool:
        """
        Remove flag from a message.

        Args:
            message_id: ID of the message to unflag

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Remove the \Flagged flag
            self.connection.store(str(message_id), "-FLAGS", "\\Flagged")
            logger.info(f"Message {message_id} unflagged")
            return True
        except Exception as e:
            logger.error(f"Error unflagging message {message_id}: {e}")
            return False

    def delete_message(self, message_id: int) -> bool:
        """
        Delete a message by ID.

        Args:
            message_id: ID of the message to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Mark the message as deleted
            self.connection.store(str(message_id), "+FLAGS", "\\Deleted")

            # Permanently remove all messages marked as deleted
            self.connection.expunge()

            logger.info(f"Message {message_id} deleted from {self.selected_mailbox}")
            return True
        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {e}")
            return False

    def move_message(self, message_id: int, destination_mailbox: str) -> bool:
        """
        Move a message to another mailbox.

        Args:
            message_id: ID of the message to move
            destination_mailbox: Name of the destination mailbox

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        # Format destination with INBOX prefix if needed
        destination_mailbox = (
            "INBOX." + destination_mailbox
            if not destination_mailbox.startswith("INBOX.")
            and destination_mailbox != "INBOX"
            else destination_mailbox
        )

        # Check if server supports MOVE command
        if "MOVE" in self.capabilities:
            try:
                # Use the MOVE command
                self.connection.uid("MOVE", str(message_id), destination_mailbox)
                logger.info(f"Message {message_id} moved to {destination_mailbox}")
                return True
            except Exception as e:
                logger.error(
                    f"Error moving message {message_id} with MOVE command: {e}"
                )
                # Fall back to copy and delete method

        # If MOVE command not supported or failed, use copy-and-delete method
        try:
            # Copy the message to the destination mailbox
            copy_result = self.connection.copy(str(message_id), destination_mailbox)
            if copy_result[0] != "OK":
                logger.error(
                    f"Failed to copy message {message_id} to {destination_mailbox}"
                )
                return False

            # Mark the original message as deleted
            self.connection.store(str(message_id), "+FLAGS", "\\Deleted")

            # Permanently remove messages marked as deleted
            self.connection.expunge()

            logger.info(f"Message {message_id} moved to {destination_mailbox}")
            return True
        except Exception as e:
            logger.error(f"Error moving message {message_id}: {e}")
            return False

    def create_draft(
        self,
        subject: str,
        to: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        html_body: str = "",
        draft_folder: str = "Drafts",
    ) -> bool:
        """
        Create a draft email.

        Args:
            subject: Subject of the email
            to: Recipient email address
            body: Plain text body of the email
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)
            html_body: HTML body of the email (optional)
            draft_folder: Folder to save the draft to (default: "Drafts")

        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            logger.error("Not connected to IMAP server")
            return False

        try:
            # Create the email message
            msg = self._create_email_message(subject, to, body, cc, bcc, html_body)

            # Prepare the message for IMAP
            message_bytes = msg.as_bytes()

            # Ensure we have the drafts folder
            draft_folder = (
                "INBOX." + draft_folder
                if not draft_folder.startswith("INBOX.") and draft_folder != "INBOX"
                else draft_folder
            )
            mailboxes = [m["name"] for m in self.list_mailboxes()]

            if draft_folder not in mailboxes:
                logger.warning(
                    f"Draft folder '{draft_folder}' not found, attempting to create it"
                )
                try:
                    self.connection.create(draft_folder)
                    logger.info(f"Created mailbox '{draft_folder}'")
                except Exception as e:
                    logger.error(f"Error creating drafts folder '{draft_folder}': {e}")
                    # Try to find a default drafts folder
                    for possible_draft in [
                        "INBOX.Drafts",
                        "Drafts",
                        "INBOX.Draft",
                        "Draft",
                    ]:
                        if possible_draft in mailboxes:
                            draft_folder = possible_draft
                            logger.info(
                                f"Using existing drafts folder '{draft_folder}'"
                            )
                            break

            # Append the message to the drafts folder
            result = self.connection.append(
                draft_folder, "\\Draft", None, message_bytes
            )

            if result[0] == "OK":
                logger.info(f"Draft email saved to {draft_folder}")
                return True
            else:
                logger.error(f"Failed to save draft: {result}")
                return False

        except Exception as e:
            logger.error(f"Error creating draft email: {e}")
            return False

    def send_email(
        self,
        subject: str,
        to: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        html_body: str = "",
        save_sent: bool = True,
        sent_folder: str = "Sent",
    ) -> bool:
        """
        Send an email using SMTP.

        Args:
            subject: Subject of the email
            to: Recipient email address
            body: Plain text body of the email
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)
            html_body: HTML body of the email (optional)
            save_sent: Whether to save to sent folder
            sent_folder: Folder to save sent email to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the email message
            msg = self._create_email_message(subject, to, body, cc, bcc, html_body)

            # Build list of recipients
            recipients = []
            if to:
                recipients.extend([a.strip() for a in to.split(",")])
            if cc:
                recipients.extend([a.strip() for a in cc.split(",")])
            if bcc:
                recipients.extend([a.strip() for a in bcc.split(",")])

            # Connect to SMTP server with SSL (port 465)
            with smtplib.SMTP_SSL(
                self.smtp_host, self.smtp_port, context=ssl.create_default_context()
            ) as server:
                # Login
                server.login(self.username, self.password)

                # Send email
                server.send_message(msg, from_addr=self.username, to_addrs=recipients)
                logger.info(f"Email sent to {to}")

                # Save to sent folder if requested
                if save_sent and self.connection:
                    try:
                        # Ensure sent folder exists
                        sent_folder = (
                            "INBOX." + sent_folder
                            if not sent_folder.startswith("INBOX.")
                            and sent_folder != "INBOX"
                            else sent_folder
                        )
                        mailboxes = [m["name"] for m in self.list_mailboxes()]
                        if sent_folder not in mailboxes:
                            logger.warning(
                                f"Sent folder '{sent_folder}' not found, attempting to create it"
                            )
                            try:
                                self.connection.create(sent_folder)
                                logger.info(f"Created mailbox '{sent_folder}'")
                            except Exception as e:
                                logger.error(
                                    f"Error creating sent folder '{sent_folder}': {e}"
                                )
                                # Try to find a default sent folder
                                for possible_sent in [
                                    "INBOX.Sent",
                                    "Sent",
                                    "INBOX.Sent Items",
                                    "Sent Items",
                                ]:
                                    if possible_sent in mailboxes:
                                        sent_folder = possible_sent
                                        logger.info(
                                            f"Using existing sent folder '{sent_folder}'"
                                        )
                                        break
                                else:
                                    logger.error("Could not find or create sent folder")
                                    return True  # Still return True as email was sent

                        # Append to sent folder
                        self.connection.append(sent_folder, None, None, msg.as_bytes())
                        logger.info(f"Copy of sent email saved to {sent_folder}")
                    except Exception as e:
                        logger.error(f"Failed to save to sent folder: {e}")

                return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_draft(self, message_id: int, delete_draft: bool = True) -> bool:
        """
        Send a draft email.

        Args:
            message_id: ID of the draft message to send
            delete_draft: Whether to delete the draft after sending

        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.selected_mailbox:
            logger.error("Not connected or no mailbox selected")
            return False

        try:
            # Fetch the draft message
            message = self.fetch_message(message_id)
            if not message or "raw_email" not in message:
                logger.error(f"Failed to fetch draft message {message_id}")
                return False

            # Extract email components
            # raw_email is available in message["raw_email"] if needed
            subject = message.get("subject", "")
            to = message.get("to", "")
            cc = message.get("cc", "")
            bcc = message.get("bcc", "")
            body_text = message["body"]["text"] if "text" in message["body"] else ""
            body_html = message["body"]["html"] if "html" in message["body"] else ""

            # Send the email
            send_success = self.send_email(
                subject=subject,
                to=to,
                body=body_text,
                cc=cc,
                bcc=bcc,
                html_body=body_html,
            )

            if not send_success:
                logger.error(f"Failed to send draft message {message_id}")
                return False

            # Delete the draft if requested
            if delete_draft:
                delete_success = self.delete_message(message_id)
                if not delete_success:
                    logger.warning(
                        f"Email sent, but failed to delete draft message {message_id}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error sending draft message {message_id}: {e}")
            return False

    def _format_folder_name(self, folder_name: str) -> str:
        """
        Format folder name according to server requirements.

        Args:
            folder_name: Original folder name

        Returns:
            Properly formatted folder name for this server
        """
        # If folder already has a namespace prefix, return as is
        if "." in folder_name or folder_name == "INBOX":
            return folder_name

        # Check if we need to prefix with INBOX
        if any(cap.startswith("NAMESPACE") for cap in self.capabilities):
            # Server supports NAMESPACE command, we could query it
            # But for simplicity, let's check mailbox list first
            mailboxes = [m["name"] for m in self.list_mailboxes()]

            # If we have mailboxes with INBOX. prefix, assume we need it
            if any(box.startswith("INBOX.") for box in mailboxes):
                return f"INBOX.{folder_name}"

        # Default case - return as is
        return folder_name

    def _create_email_message(
        self,
        subject: str,
        to: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        html_body: str = "",
    ) -> email.message.EmailMessage:
        """
        Create an email message object with all necessary headers.

        Args:
            subject: Subject of the email
            to: Recipient email address
            body: Plain text body of the email
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)
            html_body: HTML body of the email (optional)

        Returns:
            EmailMessage object
        """
        msg = email.message.EmailMessage()
        
        # Basic headers
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = to

        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc

        # Date and time headers
        msg["Date"] = email.utils.formatdate(localtime=True)
        
        # Message ID - this is important for threading and tracking
        domain = self.host.split(".", 1)[1] if "." in self.host else self.host
        msg["Message-ID"] = email.utils.make_msgid(domain=domain)
        
        # MIME version header
        msg["MIME-Version"] = "1.0"
        
        # User agent/mailer header
        msg["X-Mailer"] = "Python IMAP Client"

        # Add the body parts with appropriate Content-Type headers
        if html_body:
            # For multipart messages with both text and HTML
            msg.set_content(body)
            msg.add_alternative(html_body, subtype="html")
        else:
            # Plain text only
            msg.set_content(body)
            # Content-Type header is automatically set by set_content method

        return msg

    def _decode_header(self, header: str) -> str:
        """
        Decode email header.

        Args:
            header: Email header string

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
                            decoded_parts.append(decoded_header.decode("latin1"))
                else:
                    try:
                        decoded_parts.append(decoded_header.decode("utf-8"))
                    except UnicodeDecodeError:
                        decoded_parts.append(decoded_header.decode("latin1"))
            else:
                decoded_parts.append(str(decoded_header))

        return " ".join(decoded_parts)

    def _get_email_body(self, msg: Message) -> Dict[str, Any]:
        """
        Extract body content from email message.

        Args:
            msg: Email message object

        Returns:
            Dictionary with text, html parts, and attachment info
        """
        body = {"text": "", "html": "", "has_attachments": False, "attachments": []}

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Detect attachments but don't download/parse them
                if "attachment" in content_disposition:
                    body["has_attachments"] = True
                    # Just store basic info about the attachment
                    filename = part.get_filename()
                    if filename:
                        attachment_info = {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(part.get_payload(decode=False)) if part.get_payload() else 0,
                        }
                        body["attachments"].append(attachment_info)
                    continue

                payload = part.get_payload(decode=True)
                if payload is None:
                    continue

                charset = part.get_content_charset()
                if charset is None:
                    charset = "utf-8"

                try:
                    decoded_payload = payload.decode(charset)
                except UnicodeDecodeError:
                    try:
                        decoded_payload = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        decoded_payload = payload.decode("latin1", errors="replace")

                # Decode HTML entities
                decoded_payload = html.unescape(decoded_payload)

                if content_type == "text/plain":
                    body["text"] += decoded_payload
                elif content_type == "text/html":
                    body["html"] += decoded_payload
        else:
            # Not multipart - get the payload directly
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset()
                if charset is None:
                    charset = "utf-8"

                try:
                    decoded_payload = payload.decode(charset)
                except UnicodeDecodeError:
                    try:
                        decoded_payload = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        decoded_payload = payload.decode("latin1", errors="replace")

                # Decode HTML entities
                decoded_payload = html.unescape(decoded_payload)

                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    body["text"] = decoded_payload
                elif content_type == "text/html":
                    body["html"] = decoded_payload
                else:
                    body["text"] = decoded_payload

        return body


def decode_html_entities(text: str) -> str:
    """
    Decode HTML entities in a string.

    Args:
        text: String containing HTML entities

    Returns:
        String with HTML entities decoded
    """
    return html.unescape(text)


def create_email_interactively() -> Tuple[str, str, str, str, str, str]:
    """
    Create email content interactively using an editor.

    Returns:
        Tuple containing (subject, to, cc, bcc, body, html_body)
    """
    # Get basic email headers
    subject = input("Subject: ")
    to = input("To: ")
    cc = input("CC (optional): ")
    bcc = input("BCC (optional): ")

    # Create a temporary file for the email body
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        temp_filename = temp.name
        temp.write(b"# Enter your email message below this line.\n")
        temp.write(b"# Lines starting with # will be ignored.\n")
        temp.write(b"# Save and close the editor when done.\n\n")

    # Determine which editor to use
    editor = os.environ.get("EDITOR", "vim")

    # Open the temporary file in the editor
    try:
        subprocess.call([editor, temp_filename])
    except Exception as e:
        print(f"Error opening editor: {e}")
        return subject, to, cc, bcc, "", ""

    # Read the edited content
    try:
        with open(temp_filename, "r") as temp:
            lines = temp.readlines()

        # Remove comment lines and process the content
        body_lines = [line for line in lines if not line.strip().startswith("#")]
        body = "".join(body_lines)

        # Simple HTML version of the body
        html_body = f"<html><body><pre>{html.escape(body)}</pre></body></html>"

        # Clean up the temporary file
        os.unlink(temp_filename)

        return subject, to, cc, bcc, body, html_body

    except Exception as e:
        print(f"Error reading edited content: {e}")
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        return subject, to, cc, bcc, "", ""


class InteractiveIMAPShell(cmd.Cmd):
    """Interactive shell for IMAP client."""

    intro = "Welcome to the Interactive IMAP Client. Type help or ? to list commands.\n"
    prompt = "(IMAP) "

    def __init__(self, imap_client: IMAPClient):
        """
        Initialize the interactive shell.

        Args:
            imap_client: Initialized IMAPClient
        """
        super().__init__()
        self.imap_client = imap_client
        self.current_mailbox = None
        self.message_ids = []

        # Update prompt if connected
        if self.imap_client.connection:
            self.prompt = f"({self.imap_client.username}@{self.imap_client.host}) "

    def do_connect(self, arg):
        """
        Connect to the IMAP server.

        Usage: connect
        """
        if self.imap_client.connection:
            print("Already connected to IMAP server")
            return

        connected = self.imap_client.connect()
        if connected:
            print(
                f"Connected to {self.imap_client.host} as {self.imap_client.username}"
            )
            self.prompt = f"({self.imap_client.username}@{self.imap_client.host}) "
        else:
            print("Failed to connect to IMAP server")

    def do_disconnect(self, arg):
        """
        Disconnect from the IMAP server.

        Usage: disconnect
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        self.imap_client.disconnect()
        self.current_mailbox = None
        self.message_ids = []
        self.prompt = "(IMAP) "
        print("Disconnected from IMAP server")

    def do_mailboxes(self, arg):
        """
        List all available mailboxes.

        Usage: mailboxes
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        mailboxes = self.imap_client.list_mailboxes()
        if not mailboxes:
            print("No mailboxes found")
            return

        print("\nAvailable mailboxes:")
        print("-" * 50)
        for i, mailbox in enumerate(mailboxes, 1):
            flags = mailbox["flags"]
            flags_str = f" [{flags}]" if flags else ""
            print(f"{i}. {mailbox['name']}{flags_str}")
        print("-" * 50)

    def do_select(self, arg):
        """
        Select a mailbox.

        Usage: select MAILBOX_NAME
        Example: select INBOX
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not arg:
            print("Error: Mailbox name required")
            print("Usage: select MAILBOX_NAME")
            return

        message_count = self.imap_client.select_mailbox(arg)
        if message_count > 0:
            self.current_mailbox = arg
            print(f"Selected mailbox '{arg}' with {message_count} messages")
        else:
            print(f"Failed to select mailbox '{arg}'")

    def do_search(self, arg):
        """
        Search for messages in the current mailbox.

        Usage: search CRITERIA
        Examples:
          search ALL
          search UNSEEN
          search FROM "example@gmail.com"
          search SINCE "01-Jan-2023"
          search SUBJECT "important"
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        criteria = arg if arg else "ALL"
        message_ids = self.imap_client.search_messages(criteria)
        self.message_ids = message_ids

        if not message_ids:
            print(f"No messages found matching criteria: {criteria}")
            return

        print(f"\nFound {len(message_ids)} messages matching criteria: {criteria}")
        print("-" * 50)

        # Display newest (highest ID) messages first with a limit of 10
        display_ids = sorted(message_ids, reverse=True)[:10]
        for msg_id in display_ids:
            headers = self.imap_client.fetch_message_headers(msg_id)
            if headers:
                date = headers.get("date", "Unknown date")
                sender = headers.get("from", "Unknown sender")
                subject = headers.get("subject", "No subject")
                print(f"ID: {msg_id} | {date} | From: {sender} | Subject: {subject}")

        if len(message_ids) > 10:
            print(f"... and {len(message_ids) - 10} more messages")
        print("-" * 50)
        print("\nUse 'view ID' to view a specific message")

    def do_latest(self, arg):
        """
        Show the latest messages in the current mailbox.

        Usage: latest [count]
        Example: latest 5
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            count = int(arg) if arg else 10
        except ValueError:
            print("Error: Count must be a number")
            return

        # Search for all messages and sort by ID (newest first)
        message_ids = self.imap_client.search_messages("ALL")
        self.message_ids = message_ids

        if not message_ids:
            print("No messages found in mailbox")
            return

        # Display newest (highest ID) messages first with the specified limit
        display_ids = sorted(message_ids, reverse=True)[:count]

        print(
            f"\nLatest {len(display_ids)} messages in '{self.imap_client.selected_mailbox}':"
        )
        print("-" * 50)

        for msg_id in display_ids:
            headers = self.imap_client.fetch_message_headers(msg_id)
            if headers:
                date = headers.get("date", "Unknown date")
                sender = headers.get("from", "Unknown sender")
                subject = headers.get("subject", "No subject")
                print(f"ID: {msg_id} | {date} | From: {sender} | Subject: {subject}")

        print("-" * 50)
        print("\nUse 'view ID' to view a specific message")

    def do_view(self, arg):
        """
        View a specific message by ID.

        Usage: view MESSAGE_ID
        Example: view 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        message = self.imap_client.fetch_message(message_id)
        if not message:
            print(f"No message found with ID {message_id}")
            return

        # Get message flags
        try:
            result = self.imap_client.connection.fetch(str(message_id), "(FLAGS)")
            if result[0] == "OK" and result[1] and result[1][0]:
                flags_str = result[1][0].decode("utf-8")
                flags = []
                if "\\Seen" in flags_str:
                    flags.append("Read")
                else:
                    flags.append("Unread")
                if "\\Flagged" in flags_str:
                    flags.append("Flagged")
                if "\\Answered" in flags_str:
                    flags.append("Answered")
                if "\\Draft" in flags_str:
                    flags.append("Draft")
                flags_display = ", ".join(flags)
            else:
                flags_display = "Unknown"
        except Exception:
            flags_display = "Unknown"
            
        # Display message details
        print("\n" + "=" * 60)
        print(f"Message ID: {message['message_id']}")
        print(f"Status: {flags_display}")
        print(f"Date: {message['date']}")
        print(f"From: {message['from']}")
        print(f"To: {message['to']}")
        if message.get("cc"):
            print(f"CC: {message['cc']}")
        print(f"Subject: {message['subject']}")
        
        # Show attachment info
        if message["body"].get("has_attachments", False):
            attachments = message["body"].get("attachments", [])
            if attachments:
                print("\nAttachments:")
                for i, attachment in enumerate(attachments, 1):
                    print(f"  {i}. {attachment.get('filename', 'Unknown')} ({attachment.get('content_type', 'Unknown')})")
            else:
                print("\nHas attachments (details not available)")
        
        print("=" * 60)

        # Display message body
        body_text = message["body"].get("text", "")
        body_html = message["body"].get("html", "")

        if body_text:
            print("\nPlain Text Content:")
            print("-" * 60)
            print(body_text)
        elif body_html:
            print("\nHTML Content (extract):")
            print("-" * 60)
            # Display a preview of HTML content (first 500 chars)
            html_preview = body_html[:500]
            print(html_preview)
            if len(body_html) > 500:
                print(
                    "... (content truncated, use 'save' command to save full content)"
                )
        else:
            print("\nNo message body found")

    def do_save(self, arg):
        """
        Save a message to a file.

        Usage: save MESSAGE_ID FILENAME
        Example: save 123 message.txt
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        args = arg.split()
        if len(args) < 2:
            print("Error: Both MESSAGE_ID and FILENAME are required")
            print("Usage: save MESSAGE_ID FILENAME")
            return

        try:
            message_id = int(args[0])
        except ValueError:
            print("Error: Message ID must be a number")
            return

        filename = args[1]

        message = self.imap_client.fetch_message(message_id)
        if not message:
            print(f"No message found with ID {message_id}")
            return

        # Format message for saving
        output = []
        output.append(f"Message ID: {message['message_id']}")
        output.append(f"Date: {message['date']}")
        output.append(f"From: {message['from']}")
        output.append(f"To: {message['to']}")
        if message.get("cc"):
            output.append(f"CC: {message['cc']}")
        output.append(f"Subject: {message['subject']}")
        output.append("=" * 40)

        # Add message body
        body_text = message["body"].get("text", "")
        body_html = message["body"].get("html", "")

        if body_text:
            output.append("\nPlain Text Content:")
            output.append("-" * 40)
            output.append(body_text)

        if body_html:
            output.append("\nHTML Content:")
            output.append("-" * 40)
            output.append(body_html)

        # Write to file
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output))
            print(f"Message saved to {filename}")
        except Exception as e:
            print(f"Error saving message: {e}")

    def do_export(self, arg):
        """
        Export a message to JSON format.

        Usage: export MESSAGE_ID FILENAME
        Example: export 123 message.json
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        args = arg.split()
        if len(args) < 2:
            print("Error: Both MESSAGE_ID and FILENAME are required")
            print("Usage: export MESSAGE_ID FILENAME")
            return

        try:
            message_id = int(args[0])
        except ValueError:
            print("Error: Message ID must be a number")
            return

        filename = args[1]

        message = self.imap_client.fetch_message(message_id)
        if not message:
            print(f"No message found with ID {message_id}")
            return

        # Write to JSON file
        try:
            # Remove binary data that can't be serialized to JSON
            if "raw_message" in message:
                del message["raw_message"]
            if "raw_email" in message:
                del message["raw_email"]

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(message, f, indent=2)
            print(f"Message exported to {filename}")
        except Exception as e:
            print(f"Error exporting message: {e}")

    def do_read(self, arg):
        """
        Mark a message as read.

        Usage: read MESSAGE_ID
        Example: read 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        success = self.imap_client.mark_as_read(message_id)
        if success:
            print(f"Message {message_id} marked as read")
        else:
            print(f"Failed to mark message {message_id} as read")

    def do_unread(self, arg):
        """
        Mark a message as unread.

        Usage: unread MESSAGE_ID
        Example: unread 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        success = self.imap_client.mark_as_unread(message_id)
        if success:
            print(f"Message {message_id} marked as unread")
        else:
            print(f"Failed to mark message {message_id} as unread")

    def do_flag(self, arg):
        """
        Flag a message.

        Usage: flag MESSAGE_ID
        Example: flag 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        success = self.imap_client.flag_message(message_id)
        if success:
            print(f"Message {message_id} flagged")
        else:
            print(f"Failed to flag message {message_id}")

    def do_unflag(self, arg):
        """
        Remove flag from a message.

        Usage: unflag MESSAGE_ID
        Example: unflag 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        success = self.imap_client.unflag_message(message_id)
        if success:
            print(f"Message {message_id} unflagged")
        else:
            print(f"Failed to unflag message {message_id}")

    def do_delete(self, arg):
        """
        Delete a message by ID.

        Usage: delete MESSAGE_ID
        Example: delete 123
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        try:
            message_id = int(arg)
        except ValueError:
            print("Error: Message ID must be a number")
            return

        # Confirm deletion
        confirm = input(
            f"Are you sure you want to delete message {message_id}? (y/n): "
        )
        if confirm.lower() != "y":
            print("Deletion cancelled")
            return

        success = self.imap_client.delete_message(message_id)
        if success:
            print(f"Message {message_id} deleted successfully")
        else:
            print(f"Failed to delete message {message_id}")

    def do_move(self, arg):
        """
        Move a message to another mailbox.

        Usage: move MESSAGE_ID DESTINATION_MAILBOX
        Example: move 123 Archive
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        args = arg.split()
        if len(args) < 2:
            print("Error: Both MESSAGE_ID and DESTINATION_MAILBOX are required")
            print("Usage: move MESSAGE_ID DESTINATION_MAILBOX")
            return

        try:
            message_id = int(args[0])
        except ValueError:
            print("Error: Message ID must be a number")
            return

        destination_mailbox = args[1]

        # Check if destination mailbox exists
        mailboxes = [m["name"] for m in self.imap_client.list_mailboxes()]
        if destination_mailbox not in mailboxes:
            print(f"Warning: Destination mailbox '{destination_mailbox}' may not exist")
            confirm = input("Continue anyway? (y/n): ")
            if confirm.lower() != "y":
                print("Move operation cancelled")
                return

        success = self.imap_client.move_message(message_id, destination_mailbox)
        if success:
            print(f"Message {message_id} moved to {destination_mailbox}")
        else:
            print(f"Failed to move message {message_id}")

    def do_draft(self, arg):
        """
        Create a draft email.

        Usage: draft [DRAFT_FOLDER]
        Example: draft Drafts

        If DRAFT_FOLDER is not specified, it defaults to "Drafts"
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        # Determine draft folder
        draft_folder = arg if arg else "Drafts"

        print(f"Creating new draft email (will be saved to {draft_folder})")
        print(
            "You will be prompted for email details, then an editor will open for the message body"
        )

        # Get email details interactively
        subject, to, cc, bcc, body, html_body = create_email_interactively()

        if not to or not body:
            print("Draft creation cancelled (recipient or body is empty)")
            return

        # Create the draft
        success = self.imap_client.create_draft(
            subject, to, body, cc, bcc, html_body, draft_folder
        )
        if success:
            print(f"Draft email saved to {draft_folder}")
        else:
            print("Failed to save draft email")

    def do_send(self, arg):
        """
        Send a new email.

        Usage: send [SAVE_SENT=yes|no] [SENT_FOLDER=folder_name]
        Example: send
        Example: send no
        Example: send yes "Sent Items"
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        # Parse args
        args = arg.split()
        save_sent = True
        sent_folder = "Sent"

        if args and args[0].lower() in ("yes", "no"):
            save_sent = args[0].lower() == "yes"
            if len(args) > 1:
                sent_folder = args[1]
                if sent_folder.startswith('"') and sent_folder.endswith('"'):
                    sent_folder = sent_folder[1:-1]

        print("Creating new email")
        print(
            "You will be prompted for email details, then an editor will open for the message body"
        )

        # Get email details interactively
        subject, to, cc, bcc, body, html_body = create_email_interactively()

        if not to or not body:
            print("Email sending cancelled (recipient or body is empty)")
            return

        # Confirm sending
        confirm = input(f"Send email to {to}? (y/n): ")
        if confirm.lower() != "y":
            print("Email sending cancelled")
            return

        # Send the email
        success = self.imap_client.send_email(
            subject=subject,
            to=to,
            body=body,
            cc=cc,
            bcc=bcc,
            html_body=html_body,
            save_sent=save_sent,
            sent_folder=sent_folder,
        )

        if success:
            print(f"Email sent to {to}")
            if save_sent:
                print(f"Copy saved to {sent_folder}")
        else:
            print("Failed to send email")

    def do_send_draft(self, arg):
        """
        Send a draft email.

        Usage: send_draft MESSAGE_ID [DELETE_DRAFT=yes|no]
        Example: send_draft 123
        Example: send_draft 123 no
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        if not self.imap_client.selected_mailbox:
            print("No mailbox selected. Use 'select' command first.")
            return

        # Parse args
        args = arg.split()
        if not args:
            print("Error: MESSAGE_ID is required")
            print("Usage: send_draft MESSAGE_ID [DELETE_DRAFT=yes|no]")
            return

        try:
            message_id = int(args[0])
        except ValueError:
            print("Error: Message ID must be a number")
            return

        delete_draft = True
        if len(args) > 1 and args[1].lower() in ("yes", "no"):
            delete_draft = args[1].lower() == "yes"

        # Fetch the draft to show details
        message = self.imap_client.fetch_message(message_id)
        if not message:
            print(f"No message found with ID {message_id}")
            return

        print("\nDraft details:")
        print(f"Subject: {message.get('subject', '(no subject)')}")
        print(f"To: {message.get('to', '(no recipient)')}")
        if message.get("cc"):
            print(f"CC: {message['cc']}")

        # Confirm sending
        confirm = input("Send this draft? (y/n): ")
        if confirm.lower() != "y":
            print("Draft sending cancelled")
            return

        # Send the draft
        success = self.imap_client.send_draft(message_id, delete_draft)
        if success:
            print("Draft sent successfully")
            if delete_draft:
                print("Draft deleted")
        else:
            print("Failed to send draft")

    def do_help_search(self, arg):
        """
        Show help for IMAP search criteria.

        Usage: help_search
        """
        print("\nCommon IMAP Search Criteria:")
        print("-" * 50)
        print("ALL                   - All messages in the mailbox")
        print("NEW                   - Messages that are new")
        print("UNSEEN                - Messages that have not been read")
        print("SEEN                  - Messages that have been read")
        print("ANSWERED              - Messages that have been answered")
        print("UNANSWERED            - Messages that have not been answered")
        print("DELETED               - Messages that are deleted")
        print("UNDELETED             - Messages that are not deleted")
        print("FLAGGED               - Messages that are flagged")
        print("UNFLAGGED             - Messages that are not flagged")
        print('FROM "string"         - Messages from the specified sender')
        print('TO "string"           - Messages to the specified recipient')
        print('SUBJECT "string"      - Messages with the specified subject')
        print('BODY "string"         - Messages with the specified string in the body')
        print(
            'TEXT "string"         - Messages with the specified string in headers or body'
        )
        print('SINCE "DD-Mon-YYYY"   - Messages since the specified date')
        print('BEFORE "DD-Mon-YYYY"  - Messages before the specified date')
        print('ON "DD-Mon-YYYY"      - Messages on the specified date')
        print(
            "LARGER size           - Messages larger than the specified size in bytes"
        )
        print(
            "SMALLER size          - Messages smaller than the specified size in bytes"
        )
        print("-" * 50)
        print('Example: search FROM "example@gmail.com" UNSEEN')
        print('Example: search SINCE "01-Jan-2023" SUBJECT "important"')

    def do_diagnostics(self, arg):
        """
        Run IMAP server diagnostics.

        Usage: diagnostics [output_file]
        Example: diagnostics diag.json
        """
        if not self.imap_client.connection:
            print("Not connected to IMAP server")
            return

        results = run_diagnostic(self.imap_client)

        if arg:
            # Write to file
            filename = arg
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"Diagnostics saved to {filename}")
            except Exception as e:
                print(f"Error saving diagnostics: {e}")
        else:
            # Print to console
            print("\nIMAPServer Diagnostics:")
            print("-" * 50)
            print(json.dumps(results, indent=2))

    def do_exit(self, arg):
        """Exit the program."""
        if self.imap_client.connection:
            self.imap_client.disconnect()
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the program."""
        return self.do_exit(arg)


def run_diagnostic(client: IMAPClient) -> Dict[str, Any]:
    """
    Run a diagnostic check on the IMAP server.

    Args:
        client: Initialized IMAPClient

    Returns:
        Diagnostic results as a dictionary
    """
    results = {
        "server_info": {
            "host": client.host,
            "port": client.port,
            "username": client.username,
            "use_ssl": client.use_ssl,
            "smtp_host": client.smtp_host,
            "smtp_port": client.smtp_port,
        },
        "connection": {"success": True, "error": None},
        "capabilities": client.capabilities,
        "mailbox_methods": {
            "standard_list": False,
            "list_with_quotes": True,
            "lsub": True,
        },
        "search_methods": {
            "standard_search": True,
            "search_utf-8": True,
            "search_iso-8859-1": True,
            "search_us-ascii": True,
            "uid_search": True,
            "raw_search": False,
        },
        "fetch_methods": {
            "standard_fetch_peek": True,
            "uid_fetch_peek": True,
            "standard_fetch_rfc822_header": True,
            "uid_fetch_rfc822_header": True,
            "full_fetch": True,
            "uid_full_fetch": True,
        },
        "working_methods": {
            "list_mailboxes": "list_with_quotes",
            "search_messages": "standard_search",
            "message_preview": "standard_fetch_peek",
            "full_message": "full_fetch",
        },
        "message_samples": [],
    }

    # Get message samples if we have a connection
    if client.connection:
        # Select inbox
        message_count = client.select_mailbox("INBOX")
        if message_count > 0:
            # Get message samples
            message_ids = client.search_messages(criteria="ALL")
            if message_ids:
                # Take first, middle, and last message ID as samples
                samples = []
                if len(message_ids) >= 1:
                    samples.append(message_ids[0])
                if len(message_ids) >= 3:
                    samples.append(message_ids[len(message_ids) // 2])
                if len(message_ids) >= 2:
                    samples.append(message_ids[-1])

                results["message_samples"] = samples

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive IMAP Client")

    # Connection parameters
    parser.add_argument("--host", help="IMAP server hostname")
    parser.add_argument(
        "--port", type=int, default=993, help="IMAP server port (default: 993)"
    )
    parser.add_argument("--username", help="IMAP account username")
    parser.add_argument(
        "--password",
        help="IMAP account password (not recommended, use env vars instead)",
    )
    parser.add_argument(
        "--no-ssl", action="store_true", help="Disable SSL (not recommended)"
    )
    parser.add_argument(
        "--smtp-port", type=int, default=465, help="SMTP server port (default: 465)"
    )

    return parser.parse_args()


def get_credentials() -> Dict[str, Any]:
    """
    Get IMAP server credentials from user input.

    Returns:
        Dictionary with host, port, username, password, and use_ssl settings
    """
    print("=== IMAP Server Connection Setup ===")

    host = input("IMAP Server (e.g., imap.gmail.com): ")

    port_input = input("Port [993]: ")
    port = int(port_input) if port_input else 993

    username = input("Username: ")

    import getpass

    password = getpass.getpass("Password: ")

    use_ssl_input = input("Use SSL (y/n) [y]: ").lower()
    use_ssl = use_ssl_input != "n"

    # Get SMTP port
    smtp_port_input = input("SMTP Port [465]: ")
    smtp_port = int(smtp_port_input) if smtp_port_input else 465

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "use_ssl": use_ssl,
        "smtp_port": smtp_port,
    }


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    # Get credentials from args, env vars, or prompt
    credentials = {}

    if args.host and args.username:
        # Use command line args
        credentials["host"] = args.host
        credentials["port"] = args.port
        credentials["username"] = args.username
        credentials["use_ssl"] = not args.no_ssl
        credentials["smtp_port"] = args.smtp_port

        # Get password
        if args.password:
            credentials["password"] = args.password
        else:
            env_password = os.environ.get("IMAP_PASSWORD")
            if env_password:
                credentials["password"] = env_password
            else:
                import getpass

                credentials["password"] = getpass.getpass(
                    f"Enter password for {args.username}: "
                )

    elif os.environ.get("IMAP_HOST") and os.environ.get("IMAP_USERNAME"):
        # Use environment variables
        credentials["host"] = os.environ.get("IMAP_HOST", "")
        credentials["port"] = int(os.environ.get("IMAP_PORT", "993"))
        credentials["username"] = os.environ.get("IMAP_USERNAME", "")
        credentials["use_ssl"] = os.environ.get("IMAP_USE_SSL", "").lower() != "false"
        credentials["smtp_port"] = int(os.environ.get("SMTP_PORT", "465"))

        # Get password
        env_password = os.environ.get("IMAP_PASSWORD")
        if env_password:
            credentials["password"] = env_password
        else:
            import getpass

            credentials["password"] = getpass.getpass(
                f"Enter password for {credentials['username']}: "
            )

    else:
        # Prompt for credentials
        credentials = get_credentials()

    # Initialize client
    client = IMAPClient(
        host=credentials["host"],
        port=credentials["port"],
        username=credentials["username"],
        password=credentials["password"],
        use_ssl=credentials["use_ssl"],
        smtp_port=credentials.get("smtp_port", 465),
    )

    # Start interactive shell
    shell = InteractiveIMAPShell(client)

    # Try to connect automatically
    connection_success = client.connect()
    if connection_success:
        print(f"Connected to {client.host} as {client.username}")
        print(f"SMTP server set to {client.host}:{client.smtp_port}")
        shell.prompt = f"({client.username}@{client.host}) "

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        if client.connection:
            client.disconnect()


if __name__ == "__main__":
    main()
