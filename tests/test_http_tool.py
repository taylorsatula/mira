import unittest
from unittest.mock import patch, MagicMock
import json
import requests

from tools.http_tool import HTTPTool
from errors import ToolError, ErrorCode

class TestHTTPTool(unittest.TestCase):
    """Tests for the HTTP Tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = HTTPTool()
        
    @patch('requests.request')
    def test_get_request_success(self, mock_request):
        """Test that a successful GET request works correctly."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "success"}
        mock_response.text = '{"message": "success"}'
        mock_response.url = "https://example.com/api"
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        # Execute the tool
        result = self.tool.run(
            method="GET", 
            url="https://example.com/api", 
            params={"param1": "value1"}
        )
        
        # Verify the request was made correctly
        mock_request.assert_called_once_with(
            method="GET",
            url="https://example.com/api",
            params={"param1": "value1"},
            headers={},
            data=None,
            json=None,
            timeout=30,
            allow_redirects=True
        )
        
        # Verify the result is correct
        self.assertTrue(result["success"])
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["data"], {"message": "success"})
        
    @patch('requests.request')
    def test_post_request_success(self, mock_request):
        """Test that a successful POST request works correctly."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123, "created": True}
        mock_response.text = '{"id": 123, "created": true}'
        mock_response.url = "https://example.com/api/create"
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        # Execute the tool
        result = self.tool.run(
            method="POST", 
            url="https://example.com/api/create",
            json={"name": "Test Item"},
            headers={"Authorization": "Bearer token"}
        )
        
        # Verify the request was made correctly
        mock_request.assert_called_once_with(
            method="POST",
            url="https://example.com/api/create",
            params={},
            headers={"Authorization": "Bearer token"},
            data=None,
            json={"name": "Test Item"},
            timeout=30,
            allow_redirects=True
        )
        
        # Verify the result is correct
        self.assertTrue(result["success"])
        self.assertEqual(result["status_code"], 201)
        self.assertEqual(result["data"], {"id": 123, "created": True})
        
    @patch('requests.request')
    def test_response_format_text(self, mock_request):
        """Test that the text response format works correctly."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'Plain text response'
        mock_response.url = "https://example.com/api/text"
        mock_request.return_value = mock_response
        
        # Execute the tool
        result = self.tool.run(
            method="GET", 
            url="https://example.com/api/text",
            response_format="text"
        )
        
        # Verify the result format
        self.assertEqual(result["data"], "Plain text response")
        
    @patch('requests.request')
    def test_response_format_full(self, mock_request):
        """Test that the full response format works correctly."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"key": "value"}'
        mock_response.json.return_value = {"key": "value"}
        mock_response.url = "https://example.com/api"
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        # Execute the tool
        result = self.tool.run(
            method="GET", 
            url="https://example.com/api",
            response_format="full"
        )
        
        # Verify the full result format
        self.assertEqual(result["data"], '{"key": "value"}')
        self.assertEqual(result["headers"], {"Content-Type": "application/json"})
        self.assertEqual(result["json"], {"key": "value"})
        
    def test_invalid_method(self):
        """Test that invalid HTTP methods are rejected."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="INVALID",
                url="https://example.com/api"
            )
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
    def test_invalid_url(self):
        """Test that invalid URLs are rejected."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="GET",
                url="not-a-url"
            )
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
    def test_blocked_url(self):
        """Test that blocked internal URLs are rejected."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="GET",
                url="http://192.168.1.1/admin"
            )
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
    def test_invalid_timeout(self):
        """Test that invalid timeout values are rejected."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="GET",
                url="https://example.com/api",
                timeout=-10
            )
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
    @patch('requests.request')
    def test_timeout_error(self, mock_request):
        """Test handling of request timeouts."""
        # Make the request raise a Timeout exception
        mock_request.side_effect = unittest.mock.Mock(side_effect=requests.exceptions.Timeout("Request timed out"))
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="GET",
                url="https://example.com/api",
                timeout=5
            )
            
        self.assertEqual(context.exception.code, ErrorCode.API_TIMEOUT_ERROR)
        
    @patch('requests.request')
    def test_connection_error(self, mock_request):
        """Test handling of connection errors."""
        # Make the request raise a ConnectionError
        mock_request.side_effect = unittest.mock.Mock(side_effect=requests.exceptions.ConnectionError("Connection failed"))
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                method="GET",
                url="https://example.com/api"
            )
            
        self.assertEqual(context.exception.code, ErrorCode.API_CONNECTION_ERROR)
        
    @patch('requests.request')
    def test_json_parse_error(self, mock_request):
        """Test handling of JSON parse errors."""
        # Set up mock response with non-JSON text
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'Not valid JSON'
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.url = "https://example.com/api"
        mock_request.return_value = mock_response
        
        # Execute the tool with json response_format
        result = self.tool.run(
            method="GET", 
            url="https://example.com/api",
            response_format="json"
        )
        
        # Verify the result contains a warning
        self.assertEqual(result["data"], "Not valid JSON")
        self.assertIn("warning", result)
        
    def test_tool_metadata(self):
        """Test that the tool's metadata is correctly structured."""
        metadata = self.tool.get_metadata()
        
        # Check basic metadata
        self.assertEqual(metadata["name"], "http_tool")
        self.assertIn("description", metadata)
        
        # Check parameters
        self.assertIn("parameters", metadata)
        self.assertIn("method", metadata["parameters"])
        self.assertIn("url", metadata["parameters"])
        
        # Check required parameters
        self.assertIn("required_parameters", metadata)
        self.assertIn("method", metadata["required_parameters"])
        self.assertIn("url", metadata["required_parameters"])


if __name__ == '__main__':
    unittest.main()