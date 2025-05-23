"""
Tests for the extended thinking functionality in LLMBridge.

This module tests the extended thinking capability added to the LLMBridge
class, ensuring it correctly builds requests with the thinking parameter.
"""
import unittest
from unittest.mock import patch, MagicMock

from api.llm_bridge import LLMBridge
from config import config


class TestExtendedThinking(unittest.TestCase):
    """Test suite for extended thinking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original config values to restore later
        self.original_extended_thinking = getattr(config.api, "extended_thinking", False)
        self.original_budget = getattr(config.api, "extended_thinking_budget", 4096)
        
        # Mock the anthropic.Anthropic client
        self.client_patcher = patch('anthropic.Anthropic')
        self.mock_client = self.client_patcher.start()
        
        # Create a mock for the messages.create method
        self.mock_create = MagicMock()
        self.mock_client.return_value.messages.create = self.mock_create
        
        # Create a mock response
        self.mock_response = MagicMock()
        self.mock_response.content = [MagicMock(type="text", text="Test response")]
        self.mock_create.return_value = self.mock_response
        
        # Create LLMBridge instance for testing
        self.llm_bridge = LLMBridge()
        
        # Basic test messages
        self.test_messages = [
            {"role": "user", "content": "Hello"}
        ]

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patchers
        self.client_patcher.stop()
        
        # Restore original config values if they were changed
        if hasattr(config.api, "extended_thinking"):
            config.api.extended_thinking = self.original_extended_thinking
        if hasattr(config.api, "extended_thinking_budget"):
            config.api.extended_thinking_budget = self.original_budget

    def test_extended_thinking_disabled_by_default(self):
        """Test that extended thinking is disabled by default."""
        self.llm_bridge.generate_response(self.test_messages)
        
        # Check that the API was called without thinking parameter
        args, kwargs = self.mock_create.call_args
        self.assertNotIn("thinking", kwargs)

    def test_extended_thinking_from_config(self):
        """Test enabling extended thinking via global config."""
        # Enable extended thinking in config
        config.api.extended_thinking = True
        config.api.extended_thinking_budget = 5000
        
        self.llm_bridge.generate_response(self.test_messages)
        
        # Check that the API was called with thinking parameter
        args, kwargs = self.mock_create.call_args
        self.assertIn("thinking", kwargs)
        self.assertEqual(kwargs["thinking"]["type"], "enabled")
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 5000)

    def test_extended_thinking_per_request(self):
        """Test enabling extended thinking for a specific request."""
        self.llm_bridge.generate_response(
            self.test_messages,
            extended_thinking=True,
            extended_thinking_budget=8192
        )
        
        # Check that the API was called with thinking parameter
        args, kwargs = self.mock_create.call_args
        self.assertIn("thinking", kwargs)
        self.assertEqual(kwargs["thinking"]["type"], "enabled")
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 8192)

    def test_request_overrides_config(self):
        """Test that per-request settings override global config."""
        # Set config values
        config.api.extended_thinking = True
        config.api.extended_thinking_budget = 4096
        
        # Override with per-request values
        self.llm_bridge.generate_response(
            self.test_messages,
            extended_thinking=False  # Override to disable
        )
        
        # Check that the API was called without thinking parameter
        args, kwargs = self.mock_create.call_args
        self.assertNotIn("thinking", kwargs)

    def test_minimum_budget_enforced(self):
        """Test that minimum budget of 1024 tokens is enforced."""
        self.llm_bridge.generate_response(
            self.test_messages,
            extended_thinking=True,
            extended_thinking_budget=500  # Below minimum
        )
        
        # Check that the API was called with minimum budget
        args, kwargs = self.mock_create.call_args
        self.assertIn("thinking", kwargs)
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 1024)

    def test_max_tokens_adjusted(self):
        """Test that max_tokens is increased if needed for thinking budget."""
        # Set a small max_tokens
        small_max_tokens = 1000
        large_thinking_budget = 8192
        
        self.llm_bridge.generate_response(
            self.test_messages,
            max_tokens=small_max_tokens,
            extended_thinking=True,
            extended_thinking_budget=large_thinking_budget
        )
        
        # Check that max_tokens was increased
        args, kwargs = self.mock_create.call_args
        expected_max_tokens = small_max_tokens + large_thinking_budget
        self.assertEqual(kwargs["max_tokens"], expected_max_tokens)

    def test_max_tokens_not_adjusted_if_sufficient(self):
        """Test that max_tokens is not changed if already large enough."""
        # Set a large max_tokens
        large_max_tokens = 10000
        small_thinking_budget = 4096
        
        self.llm_bridge.generate_response(
            self.test_messages,
            max_tokens=large_max_tokens,
            extended_thinking=True,
            extended_thinking_budget=small_thinking_budget
        )
        
        # Check that max_tokens was not changed
        args, kwargs = self.mock_create.call_args
        self.assertEqual(kwargs["max_tokens"], large_max_tokens)

    def test_streaming_with_extended_thinking(self):
        """Test that extended thinking works with streaming enabled."""
        # Mock the streaming method
        mock_stream = MagicMock()
        self.mock_client.return_value.messages.stream = mock_stream
        
        # Create a mock stream response
        mock_stream_response = MagicMock()
        mock_stream.return_value = mock_stream_response
        
        self.llm_bridge.generate_response(
            self.test_messages,
            extended_thinking=True,
            extended_thinking_budget=4096,
            stream=True
        )
        
        # Check that the streaming API was called with thinking parameter
        args, kwargs = mock_stream.call_args
        self.assertIn("thinking", kwargs)
        self.assertEqual(kwargs["thinking"]["type"], "enabled")
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 4096)


if __name__ == '__main__':
    unittest.main()