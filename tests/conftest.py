"""
Pytest configuration and fixtures for the AI agent system tests.

This module provides shared fixtures and configuration for all tests.
"""
import os
import tempfile
import pytest
from pathlib import Path
import json
import logging

from config import Config
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from conversation import Conversation, Message
from crud import FileOperations
from stimuli import StimulusHandler


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary configuration file for testing."""
    config_file = temp_dir / "test_config.json"
    test_config = {
        "log_level": "DEBUG",
        "data_dir": str(temp_dir),
        "api": {
            "model": "test-model",
            "max_tokens": 500,
            "temperature": 0.5
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(test_config, f)
    
    yield config_file


@pytest.fixture
def test_config(temp_config_file):
    """Provide a test configuration instance."""
    return Config(temp_config_file)


@pytest.fixture
def file_ops(temp_dir):
    """Provide a FileOperations instance for testing."""
    return FileOperations(temp_dir)


@pytest.fixture
def mock_llm_bridge(monkeypatch):
    """Provide a mocked LLMBridge that doesn't make actual API calls."""
    class MockLLMBridge:
        def __init__(self):
            self.logger = logging.getLogger("mock_llm_bridge")
            self.calls = []
        
        def generate_response(self, messages, system_prompt=None, 
                              temperature=None, max_tokens=None, tools=None):
            self.calls.append({
                "messages": messages,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools
            })
            
            # Create a mock response
            mock_response = type('MockResponse', (), {
                'content': [
                    type('ContentBlock', (), {
                        'type': 'text',
                        'text': "This is a mock response from the LLM."
                    })
                ],
                'id': 'mock-response-id',
                'model': 'mock-model',
                'role': 'assistant'
            })
            return mock_response
        
        def extract_text_content(self, response):
            return "This is a mock response from the LLM."
        
        def extract_tool_calls(self, response):
            # By default, no tool calls
            return []
    
    # Replace the real LLMBridge with our mock
    monkeypatch.setattr("api.llm_bridge.LLMBridge", MockLLMBridge)
    return MockLLMBridge()


@pytest.fixture
def tool_repo():
    """Provide a ToolRepository instance for testing."""
    return ToolRepository()


@pytest.fixture
def conversation(mock_llm_bridge, tool_repo):
    """Provide a Conversation instance for testing."""
    return Conversation(
        conversation_id="test-conversation",
        system_prompt="Test system prompt",
        llm_bridge=mock_llm_bridge,
        tool_repo=tool_repo
    )


@pytest.fixture
def stimulus_handler():
    """Provide a StimulusHandler instance for testing."""
    return StimulusHandler()