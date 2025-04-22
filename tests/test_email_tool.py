"""
Tests for the EmailTool class.
"""
import imaplib
import smtplib
import unittest
from unittest.mock import MagicMock, patch

from errors import ToolError
from tools.email_tool import EmailTool


class TestEmailTool(unittest.TestCase):
    """Test cases for the EmailTool class."""

    def setUp(self):
        """Set up test environment."""
        # Patch the config with test values
        config_patch = patch('tools.email_tool.config')
        self.mock_config = config_patch.start()
        self.mock_config.email.imap_server = 'test.imap.server'
        self.mock_config.email.imap_port = 993
        self.mock_config.email.smtp_server = 'test.smtp.server'
        self.mock_config.email.smtp_port = 465
        self.mock_config.email.email_address = 'test@example.com'
        self.mock_config.email_password = 'test_password'
        self.mock_config.email.use_ssl = True
        self.addCleanup(config_patch.stop)

        # Create mock IMAP connection
        self.mock_imap_conn = MagicMock()
        self.mock_imap_conn.select.return_value = ('OK', [b'1'])
        
        # Patch the IMAP4_SSL constructor
        imap_patch = patch('tools.email_tool.imaplib.IMAP4_SSL')
        self.mock_imap_class = imap_patch.start()
        self.mock_imap_class.return_value = self.mock_imap_conn
        self.addCleanup(imap_patch.stop)
        
        # Patch SMTP_SSL constructor
        smtp_patch = patch('tools.email_tool.smtplib.SMTP_SSL')
        self.mock_smtp_class = smtp_patch.start()
        self.mock_smtp = self.mock_smtp_class.return_value.__enter__.return_value
        self.addCleanup(smtp_patch.stop)
        
        # Create the tool instance
        self.email_tool = EmailTool()
        
        # Pre-connect for most tests
        self.email_tool._connect()

    def test_connect(self):
        """Test connection to IMAP server."""
        tool = EmailTool()  # New instance that's not connected yet
        
        # Test successful connection
        self.assertTrue(tool._connect())
        self.assertEqual(tool.connection, self.mock_imap_conn)
        
        # Test already connected
        prev_conn = tool.connection
        self.assertTrue(tool._connect())
        self.assertEqual(tool.connection, prev_conn)  # Should be the same object
        
        # Test failed connection
        self.mock_imap_class.side_effect = Exception("Connection failed")
        tool = EmailTool()  # New instance with the mocked error
        self.assertFalse(tool._connect())
        self.assertIsNone(tool.connection)
        
        # Reset for other tests
        self.mock_imap_class.side_effect = None

    def test_disconnect(self):
        """Test disconnection from IMAP server."""
        self.assertIsNotNone(self.email_tool.connection)
        self.email_tool._disconnect()
        self.assertIsNone(self.email_tool.connection)
        self.assertIsNone(self.email_tool.selected_folder)
        
        # Test disconnect when already disconnected
        self.email_tool._disconnect()  # Should not raise any error
        
        # Test disconnect with error
        tool = EmailTool()
        tool._connect()
        tool.connection.logout.side_effect = Exception("Logout failed")
        tool._disconnect()
        self.assertIsNone(tool.connection)

    def test_select_folder(self):
        """Test folder selection."""
        # Test successful selection
        self.assertTrue(self.email_tool._select_folder("INBOX"))
        self.assertEqual(self.email_tool.selected_folder, "INBOX")
        self.mock_imap_conn.select.assert_called_with("INBOX")
        
        # Test selecting the same folder
        self.mock_imap_conn.reset_mock()
        self.assertTrue(self.email_tool._select_folder("INBOX"))
        self.mock_imap_conn.select.assert_not_called()  # Should not reselect
        
        # Test failed selection
        self.mock_imap_conn.select.side_effect = Exception("Selection failed")
        self.assertFalse(self.email_tool._select_folder("Bad_Folder"))
        
        # Reset for other tests
        self.mock_imap_conn.select.side_effect = None

    def test_email_id_registration(self):
        """Test email ID registration and retrieval."""
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Check the UUID is returned
        self.assertIsNotNone(email_uuid)
        
        # Check the mappings were created
        self.assertEqual(self.email_tool._get_message_id(email_uuid), msg_id)
        self.assertEqual(self.email_tool.msgid_to_uuid[msg_id], email_uuid)
        
        # Test registering the same message ID returns the same UUID
        email_uuid2 = self.email_tool._register_email(msg_id)
        self.assertEqual(email_uuid, email_uuid2)
        
        # Test getting a non-existent UUID
        self.assertIsNone(self.email_tool._get_message_id("non-existent-uuid"))

    def test_get_emails(self):
        """Test retrieving emails from a folder."""
        # Mock the search and fetch responses
        self.mock_imap_conn.search.return_value = ('OK', [b'1 2 3'])
        self.mock_imap_conn.fetch.side_effect = [
            ('OK', [(b'1 (FLAGS (\\Seen))', b'')]),  # Flags for message 1
            ('OK', [(b'1', b'From: test1@example.com\r\nSubject: Test Email 1\r\nDate: Mon, 1 Jan 2025 10:00:00\r\n\r\nTest body 1')]),  # Headers for message 1
            ('OK', [(b'2 (FLAGS (\\Unseen))', b'')]),  # Flags for message 2
            ('OK', [(b'2', b'From: test2@example.com\r\nSubject: Test Email 2\r\nDate: Mon, 2 Jan 2025 11:00:00\r\n\r\nTest body 2')]),  # Headers for message 2
            ('OK', [(b'3 (FLAGS (\\Seen \\Flagged))', b'')]),  # Flags for message 3
            ('OK', [(b'3', b'From: test3@example.com\r\nSubject: Test Email 3\r\nDate: Mon, 3 Jan 2025 12:00:00\r\n\r\nTest body 3')]),  # Headers for message 3
        ]
        
        # Execute the operation
        result = self.email_tool.run(
            operation="get_emails",
            folder="INBOX",
            unread_only=False,
            load_content=True
        )
        
        # Verify results
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["showing"], 3)
        self.assertTrue(result["content_loaded"])
        self.assertEqual(len(result["emails"]), 3)
        
        # Check the search criteria
        self.mock_imap_conn.search.assert_called_with(None, "ALL")
        
        # Test with unread_only=True
        self.mock_imap_conn.reset_mock()
        self.mock_imap_conn.search.return_value = ('OK', [b'2'])  # Only message 2 is unread
        
        result = self.email_tool.run(
            operation="get_emails",
            folder="INBOX",
            unread_only=True
        )
        
        self.mock_imap_conn.search.assert_called_with(None, "UNSEEN")

    def test_get_email_content(self):
        """Test retrieving a specific email's content."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Mock the fetch response with a proper RFC822 format
        self.mock_imap_conn.fetch.side_effect = [
            # First fetch call for the full RFC822 message
            ('OK', [(b'1', b'From: sender@example.com\r\nTo: recipient@example.com\r\nSubject: Test Email\r\nDate: Mon, 1 Jan 2025 10:00:00\r\n\r\nTest email body')]),
            # Second fetch call for the flags
            ('OK', [(b'1 (FLAGS (\\Seen))', b'')])
        ]
        
        # Execute the operation
        result = self.email_tool.run(
            operation="get_email_content",
            email_id=email_uuid
        )
        
        # Verify results
        self.assertEqual(result["id"], email_uuid)
        self.assertEqual(result["from"], "sender@example.com")
        self.assertEqual(result["subject"], "Test Email")
        self.assertEqual(result["body_text"], "Test email body")
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="get_email_content",
                email_id="non-existent-uuid"
            )

    def test_mark_as_read(self):
        """Test marking an email as read."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Mock the store response
        self.mock_imap_conn.store.return_value = ('OK', [b'1 (FLAGS (\\Seen))'])
        
        # Execute the operation
        result = self.email_tool.run(
            operation="mark_as_read",
            email_id=email_uuid
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["email_id"], email_uuid)
        
        # Check the store command
        self.mock_imap_conn.store.assert_called_with(str(msg_id), "+FLAGS", "\\Seen")
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="mark_as_read",
                email_id="non-existent-uuid"
            )

    def test_mark_as_unread(self):
        """Test marking an email as unread."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Mock the store response
        self.mock_imap_conn.store.return_value = ('OK', [b'1 (FLAGS ())'])
        
        # Execute the operation
        result = self.email_tool.run(
            operation="mark_as_unread",
            email_id=email_uuid
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["email_id"], email_uuid)
        
        # Check the store command
        self.mock_imap_conn.store.assert_called_with(str(msg_id), "-FLAGS", "\\Seen")
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="mark_as_unread",
                email_id="non-existent-uuid"
            )

    def test_delete_email(self):
        """Test deleting an email."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Mock the store and expunge responses
        self.mock_imap_conn.store.return_value = ('OK', [b'1 (FLAGS (\\Deleted))'])
        self.mock_imap_conn.expunge.return_value = ('OK', [b'1'])
        
        # Execute the operation
        result = self.email_tool.run(
            operation="delete_email",
            email_id=email_uuid
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["email_id"], email_uuid)
        
        # Check the store command
        self.mock_imap_conn.store.assert_called_with(str(msg_id), "+FLAGS", "\\Deleted")
        self.mock_imap_conn.expunge.assert_called_once()
        
        # Check that the mappings were removed
        self.assertNotIn(email_uuid, self.email_tool.uuid_to_msgid)
        self.assertNotIn(msg_id, self.email_tool.msgid_to_uuid)
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="delete_email",
                email_id="non-existent-uuid"
            )

    def test_move_email(self):
        """Test moving an email to another folder."""
        # Create a new instance of the tool for this test
        tool = EmailTool()
        tool._connect()
        
        # Register a test email ID
        msg_id = 12345
        email_uuid = tool._register_email(msg_id)
        
        # The self.mock_imap_conn is passed to the constructor, so tool.connection will also be the mock
        
        # Set hasattr result for move to false
        def hasattr_move_side_effect(obj, attr):
            if attr == "move":
                return False
            return original_hasattr(obj, attr)
        
        original_hasattr = hasattr
        
        with patch('tools.email_tool.hasattr', side_effect=hasattr_move_side_effect):
            # Mock the copy, store, and expunge responses
            self.mock_imap_conn.copy.return_value = ('OK', [b'1'])
            self.mock_imap_conn.store.return_value = ('OK', [b'1 (FLAGS (\\Deleted))'])
            self.mock_imap_conn.expunge.return_value = ('OK', [b'1'])
            
            # Execute the operation
            result = tool.run(
                operation="move_email",
                email_id=email_uuid,
                destination_folder="Trash"
            )
            
            # Verify results
            self.assertTrue(result["success"])
            self.assertEqual(result["email_id"], email_uuid)
            self.assertEqual(result["destination"], "Trash")
            
            # Check the copy and store commands
            self.mock_imap_conn.copy.assert_called_with(str(msg_id), "Trash")
            self.mock_imap_conn.store.assert_called_with(str(msg_id), "+FLAGS", "\\Deleted")
            self.mock_imap_conn.expunge.assert_called_once()
            
            # Check that the mappings were removed
            self.assertNotIn(email_uuid, tool.uuid_to_msgid)
            self.assertNotIn(msg_id, tool.msgid_to_uuid)
            
            # Test with invalid email_id
            with self.assertRaises(ToolError):
                tool.run(
                    operation="move_email",
                    email_id="non-existent-uuid",
                    destination_folder="Trash"
                )
            
            # Test without destination_folder
            with self.assertRaises(ToolError):
                tool.run(
                    operation="move_email",
                    email_id=email_uuid
                )

    def test_send_email(self):
        """Test sending an email."""
        # Execute the operation
        result = self.email_tool.run(
            operation="send_email",
            to="recipient@example.com",
            subject="Test Subject",
            body="Test body content",
            cc="cc@example.com",
            bcc="bcc@example.com"
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["to"], "recipient@example.com")
        self.assertEqual(result["subject"], "Test Subject")
        
        # Check SMTP interactions
        self.mock_smtp.login.assert_called_with(self.mock_config.email.email_address, self.mock_config.email_password)
        self.mock_smtp.send_message.assert_called_once()
        
        # Check message structure from the send_message call
        msg = self.mock_smtp.send_message.call_args[0][0]
        self.assertEqual(msg["Subject"], "Test Subject")
        self.assertEqual(msg["From"], "test@example.com")
        self.assertEqual(msg["To"], "recipient@example.com")
        self.assertEqual(msg["Cc"], "cc@example.com")
        
        # Test with missing required parameters
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="send_email",
                to="recipient@example.com",
                subject="Test Subject"
                # missing body
            )

    def test_reply_to_email(self):
        """Test replying to an email."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Mock the fetch response for the original email
        self.mock_imap_conn.fetch.side_effect = [
            ('OK', [(b'1', b'From: sender@example.com\r\nTo: recipient@example.com\r\nSubject: Original Subject\r\nDate: Mon, 1 Jan 2025 10:00:00\r\nMessage-ID: <original-message-id@example.com>\r\n\r\nOriginal email body')]),
            ('OK', [(b'1 (FLAGS (\\Seen \\Answered))', b'')])  # Response for setting the \Answered flag
        ]
        
        # Execute the operation
        result = self.email_tool.run(
            operation="reply_to_email",
            email_id=email_uuid,
            body="Reply body content"
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["replied_to"], email_uuid)
        self.assertEqual(result["subject"], "Re: Original Subject")
        
        # Check SMTP interactions
        self.mock_smtp.login.assert_called_with(self.mock_config.email.email_address, self.mock_config.email_password)
        self.mock_smtp.send_message.assert_called_once()
        
        # Check message structure from the send_message call
        msg = self.mock_smtp.send_message.call_args[0][0]
        self.assertEqual(msg["Subject"], "Re: Original Subject")
        self.assertEqual(msg["From"], "test@example.com")
        self.assertEqual(msg["To"], "sender@example.com")
        self.assertEqual(msg["In-Reply-To"], "<original-message-id@example.com>")
        
        # Check that the original message was marked as answered
        self.mock_imap_conn.store.assert_called_with(str(msg_id), "+FLAGS", "\\Answered")
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="reply_to_email",
                email_id="non-existent-uuid",
                body="Reply body content"
            )
        
        # Test without body
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="reply_to_email",
                email_id=email_uuid
                # missing body
            )

    def test_create_draft(self):
        """Test creating a draft email."""
        # Mock the append response
        self.mock_imap_conn.append.return_value = ('OK', [b'APPENDUID 1 1'])
        
        # Execute the operation
        result = self.email_tool.run(
            operation="create_draft",
            to="recipient@example.com",
            subject="Draft Subject",
            body="Draft body content",
            cc="cc@example.com",
            bcc="bcc@example.com"
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["to"], "recipient@example.com")
        self.assertEqual(result["subject"], "Draft Subject")
        
        # Check append interaction
        self.mock_imap_conn.append.assert_called_once()
        
        # Test with missing required parameters
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="create_draft",
                to="recipient@example.com",
                subject="Draft Subject"
                # missing body
            )

    def test_mark_for_later_reply(self):
        """Test marking an email for later reply."""
        # Register a test email ID
        msg_id = 12345
        email_uuid = self.email_tool._register_email(msg_id)
        
        # Execute the operation
        result = self.email_tool.run(
            operation="mark_for_later_reply",
            email_id=email_uuid
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["email_id"], email_uuid)
        
        # Check that the email was added to the later reply set
        self.assertIn(email_uuid, self.email_tool.emails_for_later_reply)
        
        # Test with invalid email_id
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="mark_for_later_reply",
                email_id="non-existent-uuid"
            )

    def test_get_emails_for_later_reply(self):
        """Test getting emails marked for later reply."""
        # Register and mark test email IDs
        msg_id1 = 12345
        email_uuid1 = self.email_tool._register_email(msg_id1)
        self.email_tool.emails_for_later_reply.add(email_uuid1)
        
        msg_id2 = 67890
        email_uuid2 = self.email_tool._register_email(msg_id2)
        self.email_tool.emails_for_later_reply.add(email_uuid2)
        
        # Mock the fetch responses
        self.mock_imap_conn.fetch.side_effect = [
            ('OK', [(b'1', b'From: sender1@example.com\r\nSubject: Test Email 1\r\nDate: Mon, 1 Jan 2025 10:00:00\r\n\r\n')]),
            ('OK', [(b'1 (FLAGS (\\Seen))', b'')]),
            ('OK', [(b'2', b'From: sender2@example.com\r\nSubject: Test Email 2\r\nDate: Mon, 2 Jan 2025 11:00:00\r\n\r\n')]),
            ('OK', [(b'2 (FLAGS (\\Unseen))', b'')])
        ]
        
        # Execute the operation
        result = self.email_tool.run(
            operation="get_emails_for_later_reply"
        )
        
        # Verify results
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["emails"]), 2)
        self.assertTrue(any(e["id"] == email_uuid1 for e in result["emails"]))
        self.assertTrue(any(e["id"] == email_uuid2 for e in result["emails"]))
        
        # Test with an empty later reply set
        self.email_tool.emails_for_later_reply.clear()
        
        result = self.email_tool.run(
            operation="get_emails_for_later_reply"
        )
        
        self.assertEqual(result["count"], 0)
        self.assertEqual(len(result["emails"]), 0)

    def test_list_folders(self):
        """Test listing email folders."""
        # Mock the list response
        self.mock_imap_conn.list.return_value = ('OK', [
            b'(\\HasNoChildren) "." "INBOX"',
            b'(\\HasNoChildren) "." "Sent"',
            b'(\\HasNoChildren) "." "Drafts"',
            b'(\\HasNoChildren) "." "Trash"'
        ])
        
        # Execute the operation
        result = self.email_tool.run(
            operation="list_folders"
        )
        
        # Verify results
        self.assertEqual(len(result["folders"]), 4)
        folder_names = [f["name"] for f in result["folders"]]
        self.assertIn("INBOX", folder_names)
        self.assertIn("Sent", folder_names)
        self.assertIn("Drafts", folder_names)
        self.assertIn("Trash", folder_names)
        
        # Test with list error
        self.mock_imap_conn.list.return_value = ('NO', [b'Error listing folders'])
        
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="list_folders"
            )

    def test_search_emails(self):
        """Test searching for emails with specific criteria."""
        # Mock the search and fetch responses
        self.mock_imap_conn.search.return_value = ('OK', [b'1 2'])
        self.mock_imap_conn.fetch.side_effect = [
            ('OK', [(b'1 (FLAGS (\\Seen))', b'')]),  # Flags for message 1
            ('OK', [(b'1', b'From: test1@example.com\r\nSubject: Test Search 1\r\nDate: Mon, 1 Jan 2025 10:00:00\r\n\r\nTest search 1')]),  # Headers for message 1
            ('OK', [(b'2 (FLAGS (\\Unseen))', b'')]),  # Flags for message 2
            ('OK', [(b'2', b'From: test2@example.com\r\nSubject: Test Search 2\r\nDate: Mon, 2 Jan 2025 11:00:00\r\n\r\nTest search 2')]),  # Headers for message 2
        ]
        
        # Execute the operation
        result = self.email_tool.run(
            operation="search_emails",
            folder="INBOX",
            sender="test@example.com",
            subject="Test",
            start_date="01-Jan-2025",
            end_date="10-Jan-2025",
            unread_only=True
        )
        
        # Verify results
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["showing"], 2)
        self.assertEqual(len(result["emails"]), 2)
        
        # Check the search criteria
        expected_criteria = 'UNSEEN FROM "test@example.com" SUBJECT "Test" SINCE "01-Jan-2025" BEFORE "10-Jan-2025"'
        self.mock_imap_conn.search.assert_called_with(None, expected_criteria)
        
        # Test with no criteria
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="search_emails",
                folder="INBOX"
            )

    def test_unknown_operation(self):
        """Test handling of unknown operations."""
        with self.assertRaises(ToolError):
            self.email_tool.run(
                operation="unknown_operation"
            )


if __name__ == '__main__':
    unittest.main()