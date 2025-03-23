"""
Tests for the main application module.

This module tests the main application entry point, including command-line
argument parsing, system initialization, and interactive mode.
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

import main
from errors import AgentError


def test_parse_arguments():
    """Test parsing command-line arguments."""
    # Test with no arguments
    with patch('sys.argv', ['main.py']):
        args = main.parse_arguments()
        assert args.config is None
        assert args.conversation is None
        assert args.log_level is None
    
    # Test with arguments
    with patch('sys.argv', [
        'main.py',
        '--config', 'test_config.json',
        '--conversation', 'test-conversation',
        '--log-level', 'DEBUG'
    ]):
        args = main.parse_arguments()
        assert args.config == 'test_config.json'
        assert args.conversation == 'test-conversation'
        assert args.log_level == 'DEBUG'
    
    # Test with short options
    with patch('sys.argv', [
        'main.py',
        '-c', 'test_config.json',
        '-id', 'test-conversation'
    ]):
        args = main.parse_arguments()
        assert args.config == 'test_config.json'
        assert args.conversation == 'test-conversation'


def test_setup_logging():
    """Test setting up logging."""
    # Test with default log level
    with patch('logging.basicConfig') as mock_config:
        main.setup_logging()
        mock_config.assert_called_once()
        # Default log level should be INFO
        assert mock_config.call_args[1]['level'] == 20  # INFO = 20
    
    # Test with specific log level
    with patch('logging.basicConfig') as mock_config:
        main.setup_logging('DEBUG')
        mock_config.assert_called_once()
        # DEBUG = 10
        assert mock_config.call_args[1]['level'] == 10


def test_initialize_system(monkeypatch, temp_dir):
    """Test system initialization."""
    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.config = None
    mock_args.log_level = None
    mock_args.conversation = None
    
    # Mock config to use temp directory
    monkeypatch.setattr('config.config.get', lambda key, default: temp_dir if key == 'data_dir' else default)
    
    # Test initialization
    with patch('logging.getLogger') as mock_logger:
        system = main.initialize_system(mock_args)
        
        # Check components
        assert 'file_ops' in system
        assert 'llm_bridge' in system
        assert 'tool_repo' in system
        assert 'conversation' in system
        
        # Check that components are initialized
        assert system['file_ops'].data_dir == Path(temp_dir)
        assert system['conversation'].conversation_id is not None


def test_initialize_with_conversation(monkeypatch, temp_dir, file_ops):
    """Test initialization with an existing conversation."""
    # Create a test conversation
    conversation_id = 'test-conversation'
    conversation_data = {
        'conversation_id': conversation_id,
        'system_prompt': 'Test system prompt',
        'messages': [
            {
                'role': 'user',
                'content': 'Hello',
                'id': '1',
                'created_at': 1615900000,
                'metadata': {}
            },
            {
                'role': 'assistant',
                'content': 'Hi there!',
                'id': '2',
                'created_at': 1615900001,
                'metadata': {}
            }
        ],
        'created_at': 1615900000,
        'updated_at': 1615900001
    }
    
    # Write the conversation to a file
    file_ops.write(f'conversation_{conversation_id}', conversation_data)
    
    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.config = None
    mock_args.log_level = None
    mock_args.conversation = conversation_id
    
    # Mock config to use our temp directory
    monkeypatch.setattr('config.config.get', lambda key, default: file_ops.data_dir if key == 'data_dir' else default)
    
    # Test initialization with the conversation
    with patch('logging.getLogger'):
        with patch('main.FileOperations', return_value=file_ops):
            system = main.initialize_system(mock_args)
            
            # Check that the conversation was loaded
            assert system['conversation'].conversation_id == conversation_id
            assert system['conversation'].system_prompt == 'Test system prompt'
            assert len(system['conversation'].messages) == 2


def test_save_conversation(file_ops):
    """Test saving a conversation."""
    # Create a test conversation
    mock_conversation = MagicMock()
    mock_conversation.conversation_id = 'test-conversation'
    mock_conversation.to_dict.return_value = {
        'conversation_id': 'test-conversation',
        'messages': []
    }
    
    # Save the conversation
    with patch('logging.info') as mock_log:
        main.save_conversation(file_ops, mock_conversation)
        
        # Check that the conversation was saved
        assert mock_conversation.to_dict.called
        saved_data = file_ops.read('conversation_test-conversation')
        assert saved_data['conversation_id'] == 'test-conversation'
        assert mock_log.called


def test_interactive_mode(monkeypatch):
    """Test interactive mode."""
    # Create mock components
    mock_conversation = MagicMock()
    mock_conversation.conversation_id = 'test-conversation'
    mock_conversation.generate_response.return_value = 'This is a test response'
    
    mock_file_ops = MagicMock()
    
    system = {
        'conversation': mock_conversation,
        'file_ops': mock_file_ops
    }
    
    # Test with 'exit' command
    with patch('builtins.input', return_value='exit'):
        with patch('builtins.print') as mock_print:
            main.interactive_mode(system)
            
            # Check that save was called
            assert mock_file_ops.write.called
            
            # Check that goodbye was printed
            mock_print.assert_called_with('Goodbye!')
    
    # Test with 'save' command
    with patch('builtins.input', side_effect=['save', 'exit']):
        with patch('builtins.print') as mock_print:
            main.interactive_mode(system)
            
            # Check that save was called twice (once for save, once for exit)
            assert mock_file_ops.write.call_count == 2
            
            # Check that confirmation was printed
            assert any('Conversation saved' in str(args) for args, _ in mock_print.call_args_list)
    
    # Test with 'clear' command
    with patch('builtins.input', side_effect=['clear', 'exit']):
        with patch('builtins.print') as mock_print:
            main.interactive_mode(system)
            
            # Check that clear was called
            assert mock_conversation.clear_history.called
            
            # Check that confirmation was printed
            assert any('Conversation history cleared' in str(args) for args, _ in mock_print.call_args_list)
    
    # Test with normal input
    with patch('builtins.input', side_effect=['Hello', 'exit']):
        with patch('builtins.print') as mock_print:
            main.interactive_mode(system)
            
            # Check that generate_response was called
            mock_conversation.generate_response.assert_called_with('Hello')
            
            # Check that response was printed
            assert any('This is a test response' in str(args) for args, _ in mock_print.call_args_list)


def test_main_function(monkeypatch):
    """Test the main function."""
    # Mock the necessary components
    monkeypatch.setattr('main.parse_arguments', lambda: MagicMock(config=None, log_level=None, conversation=None))
    monkeypatch.setattr('main.setup_logging', lambda *args: None)
    
    mock_system = {
        'conversation': MagicMock(),
        'file_ops': MagicMock()
    }
    monkeypatch.setattr('main.initialize_system', lambda args: mock_system)
    
    # Mock interactive mode
    mock_interactive = MagicMock()
    monkeypatch.setattr('main.interactive_mode', mock_interactive)
    
    # Test normal execution
    result = main.main()
    
    # Check that interactive mode was called
    assert mock_interactive.called
    assert mock_interactive.call_args[0][0] == mock_system
    
    # Check that the function returned 0 (success)
    assert result == 0
    
    # Test with an exception
    mock_interactive.side_effect = Exception('Test error')
    
    with patch('logging.exception') as mock_log:
        result = main.main()
        
        # Check that the exception was logged
        assert mock_log.called
        
        # Check that the function returned 1 (error)
        assert result == 1


def test_initialization_error(monkeypatch):
    """Test handling of initialization errors."""
    # Mock an initialization error
    def mock_initialize_error(args):
        raise AgentError('Test initialization error')
    
    monkeypatch.setattr('main.initialize_system', mock_initialize_error)
    monkeypatch.setattr('main.parse_arguments', lambda: MagicMock(config=None, log_level=None, conversation=None))
    monkeypatch.setattr('main.setup_logging', lambda *args: None)
    
    # Test with sys.exit mocked to avoid actually exiting
    with patch('sys.exit') as mock_exit:
        with patch('logging.getLogger'):
            with patch('builtins.print') as mock_print:
                main.main()
                
                # Check that error was printed
                assert any('Error: Test initialization error' in str(args) for args, _ in mock_print.call_args_list)
                
                # Check that exit was called with error code
                mock_exit.assert_called_once_with(1)