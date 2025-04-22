"""
Unit tests for the EmailTool.

This module contains tests for the EmailTool, focusing on its interface and
basic functionality. Many tests use mock objects to avoid actual network
connections to IMAP/SMTP servers.
"""
import unittest
from unittest.mock import patch, MagicMock, call

from tools.email_tool import EmailTool
from errors import ToolError


class TestEmailTool(unittest.TestCase):
    """Tests for the EmailTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch the IMAP and SMTP connections
        self.imap_patcher = patch('imaplib.IMAP4_SSL')
        self.smtp_patcher = patch('smtplib.SMTP_SSL')
        
        # Start the patchers
        self.mock_imap = self.imap_patcher.start()
        self.mock_smtp = self.smtp_patcher.start()
        
        # Create mock connection
        self.mock_connection = MagicMock()
        self.mock_imap.return_value = self.mock_connection
        
        # Create the tool
        self.tool = EmailTool()
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.imap_patcher.stop()
        self.smtp_patcher.stop()
    
    @patch('tools.email_tool.config')
    def test_initialization(self, mock_config):
        """Test that the tool initializes properly."""
        # Set up the mock config
        mock_config.email.imap_server = 'imap.example.com'
        mock_config.email.imap_port = 993
        mock_config.email.smtp_server = 'smtp.example.com'
        mock_config.email.smtp_port = 465
        mock_config.email.email_address = 'user@example.com'
        mock_config.email_password = 'password'
        mock_config.email.use_ssl = True
        mock_config.email.default_folders = {
            'inbox': 'INBOX',
            'sent': 'Sent',
            'drafts': 'Drafts',
            'trash': 'Trash'
        }
        mock_config.email.max_emails_to_fetch = 50
        mock_config.email.max_preview_length = 100
        
        # Create a new tool with the mock config
        tool = EmailTool()
        
        # Verify the tool was initialized with the correct config values
        self.assertEqual(tool.imap_server, 'imap.example.com')
        self.assertEqual(tool.imap_port, 993)
        self.assertEqual(tool.smtp_server, 'smtp.example.com')
        self.assertEqual(tool.smtp_port, 465)
        self.assertEqual(tool.email_address, 'user@example.com')
        self.assertEqual(tool.password, 'password')
        self.assertEqual(tool.use_ssl, True)
        self.assertEqual(tool.max_emails, 50)
        self.assertEqual(tool.preview_length, 100)
        
        # Verify the internal state was initialized
        self.assertIsNone(tool.connection)
        self.assertIsNone(tool.selected_folder)
        self.assertEqual(tool.uuid_to_msgid, {})
        self.assertEqual(tool.msgid_to_uuid, {})
        self.assertIn('humans', tool.email_buckets)
        self.assertIn('priority', tool.email_buckets)
        self.assertIn('notifications', tool.email_buckets)
        self.assertIn('newsletters', tool.email_buckets)
        self.assertIn('to_reply_later', tool.email_buckets)
        self.assertEqual(tool.seen_emails, set())
    
    def test_connect(self):
        """Test connecting to the IMAP server."""
        # Set up the mock
        self.mock_connection.login.return_value = ('OK', [b'Login successful'])
        self.mock_connection.select.return_value = ('OK', [b'1'])
        
        # Call the method
        result = self.tool._connect()
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the mock was called correctly
        self.mock_imap.assert_called_once_with(self.tool.imap_server, self.tool.imap_port)
        self.mock_connection.login.assert_called_once_with(self.tool.email_address, self.tool.password)
        self.mock_connection.select.assert_called_once_with('INBOX')
    
    def test_connect_failure(self):
        """Test handling of connection failures."""
        # Set up the mock to fail
        self.mock_connection.login.side_effect = Exception('Connection failed')
        
        # Call the method
        result = self.tool._connect()
        
        # Verify the result
        self.assertFalse(result)
    
    def test_disconnect(self):
        """Test disconnecting from the IMAP server."""
        # Set up the tool with a connection
        self.tool.connection = self.mock_connection
        
        # Call the method
        self.tool._disconnect()
        
        # Verify the mock was called correctly
        self.mock_connection.logout.assert_called_once()
        
        # Verify the connection was cleared
        self.assertIsNone(self.tool.connection)
    
    def test_register_email(self):
        """Test registering and retrieving email IDs."""
        # Register an email
        email_uuid = self.tool._register_email(123)
        
        # Verify the UUID was generated and mapped
        self.assertIsNotNone(email_uuid)
        self.assertIn(email_uuid, self.tool.uuid_to_msgid)
        self.assertEqual(self.tool.uuid_to_msgid[email_uuid], 123)
        self.assertIn(123, self.tool.msgid_to_uuid)
        self.assertEqual(self.tool.msgid_to_uuid[123], email_uuid)
        
        # Verify retrieving the message ID
        msg_id = self.tool._get_message_id(email_uuid)
        self.assertEqual(msg_id, 123)
        
        # Verify retrieving a non-existent UUID
        msg_id = self.tool._get_message_id('non-existent')
        self.assertIsNone(msg_id)
    
    def test_add_to_bucket(self):
        """Test adding emails to buckets."""
        # Register an email
        email_uuid = self.tool._register_email(123)
        
        # Add to a bucket
        result = self.tool._add_to_bucket(email_uuid, 'humans')
        
        # Verify the result
        self.assertTrue(result)
        self.assertIn(email_uuid, self.tool.email_buckets['humans'])
        
        # Try adding to a non-existent bucket
        result = self.tool._add_to_bucket(email_uuid, 'custom')
        
        # Verify the result
        self.assertTrue(result)
        self.assertIn('custom', self.tool.email_buckets)
        self.assertIn(email_uuid, self.tool.email_buckets['custom'])
    
    @patch('tools.email_tool.EmailTool._connect')
    @patch('tools.email_tool.EmailTool._search_messages')
    @patch('tools.email_tool.EmailTool._fetch_message_headers')
    @patch('tools.email_tool.EmailTool._categorize_emails')
    def test_get_unread_emails(self, mock_categorize, mock_fetch, mock_search, mock_connect):
        """Test retrieving unread emails."""
        # Set up the mocks
        mock_connect.return_value = True
        mock_search.return_value = [1, 2, 3]
        mock_fetch.return_value = [
            {'id': 'uuid1', 'from': 'person@example.com'},
            {'id': 'uuid2', 'from': 'service@example.com'}
        ]
        mock_categorize.return_value = {
            'humans': [0],
            'notifications': [1],
            'newsletters': [],
            'priority': []
        }
        
        # Call the method
        result = self.tool.run(operation='get_unread_emails')
        
        # Verify the result
        self.assertEqual(result['emails'], mock_fetch.return_value)
        self.assertEqual(result['categories'], mock_categorize.return_value)
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['showing'], 2)
        
        # Verify the mocks were called correctly
        mock_connect.assert_called_once()
        mock_search.assert_called_once_with('UNSEEN')
        mock_fetch.assert_called_once_with([1, 2, 3], self.tool.max_emails)
        mock_categorize.assert_called_once_with(mock_fetch.return_value)
    
    @patch('tools.email_tool.EmailTool._connect')
    def test_invalid_operation(self, mock_connect):
        """Test handling of invalid operations."""
        # Set up the mock
        mock_connect.return_value = True
        
        # Call the method with an invalid operation
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='invalid_operation')
        
        # Verify the error
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    @patch('tools.email_tool.EmailTool._connect')
    @patch('tools.email_tool.EmailTool._get_message_id')
    def test_get_email_content_missing_id(self, mock_get_message_id, mock_connect):
        """Test handling of missing email ID."""
        # Set up the mocks
        mock_connect.return_value = True
        
        # Call the method without an email ID
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='get_email_content')
        
        # Verify the error
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Call the method with an invalid email ID
        mock_get_message_id.return_value = None
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='get_email_content', email_id='invalid')
        
        # Verify the error
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)


if __name__ == '__main__':
    unittest.main()