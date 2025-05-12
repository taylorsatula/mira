"""
Test suite for the kasa_tool module.

This module contains tests that verify the asyncio integration
of the kasa_tool with the automation engine.
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import patch, MagicMock

from tools.kasa_tool import KasaTool
from utils.async_utils import get_or_create_event_loop, run_coroutine, close_event_loop
from task_manager.automation_engine import AutomationEngine
from tools.repo import ToolRepository


def test_kasa_tool_asyncio_integration():
    """Test that KasaTool properly integrates with AsyncioToolBase."""
    # Create KasaTool instance
    kasa_tool = KasaTool()
    
    # Verify it's using the AsyncioToolBase
    assert hasattr(kasa_tool, 'run_async')
    assert callable(kasa_tool.run_async)
    assert hasattr(kasa_tool, 'get_thread_data')
    assert callable(kasa_tool.get_thread_data)
    assert hasattr(kasa_tool, 'cleanup')
    assert callable(kasa_tool.cleanup)


@patch('kasa.Discover')
def test_kasa_tool_device_discovery(mock_discover):
    """Test device discovery with thread-local storage."""
    # Setup mock device
    mock_device = MagicMock()
    mock_device.host = '192.168.1.100'
    mock_device.alias = 'Test Device'
    mock_device.is_on = True
    mock_device.device_type = 'bulb'
    mock_device.model = 'KL130'
    mock_device.mac = '00:11:22:33:44:55'
    mock_device.rssi = -60
    mock_device.device_info = MagicMock()
    mock_device.device_info.hardware_version = 'v1.0'
    mock_device.device_info.firmware_version = '1.2.3'
    mock_device.has_emeter = False
    mock_device.modules = {'Light': MagicMock()}
    mock_device.modules['Light'].brightness = 75
    
    # Configure mock discover
    mock_discover.discover = MagicMock()
    mock_discover.discover.return_value = asyncio.Future()
    mock_discover.discover.return_value.set_result({
        '192.168.1.100': mock_device
    })
    
    # Create KasaTool instance
    kasa_tool = KasaTool()
    
    # Mock device.update method to avoid actual network calls
    async def mock_update():
        return None
    mock_device.update = mock_update
    
    # Test discovery
    result = kasa_tool.run('discover_devices')
    
    # Verify result
    assert result['success'] is True
    assert len(result['devices']) == 1
    assert result['devices'][0]['id'] == '192.168.1.100'
    assert result['devices'][0]['alias'] == 'Test Device'
    
    # Verify device is cached in thread-local storage
    device_instances = kasa_tool.get_thread_data('device_instances', {})
    assert '192.168.1.100' in device_instances
    assert 'Test Device' in device_instances
    
    # Multiple threads should have their own device caches
    thread_device_exists = False
    
    def thread_func():
        nonlocal thread_device_exists
        # Get thread-specific device instances
        device_instances = kasa_tool.get_thread_data('device_instances', {})
        # This thread should not see the device
        thread_device_exists = '192.168.1.100' in device_instances
    
    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()
    
    # Verify thread isolation
    assert thread_device_exists is False


@patch('kasa.Discover')
def test_kasa_tool_cleanup(mock_discover):
    """Test that KasaTool properly cleans up resources."""
    # Setup mock device
    mock_device = MagicMock()
    mock_device.host = '192.168.1.100'
    mock_device.alias = 'Test Device'
    
    # Configure mock discover
    mock_discover.discover = MagicMock()
    mock_discover.discover.return_value = asyncio.Future()
    mock_discover.discover.return_value.set_result({
        '192.168.1.100': mock_device
    })
    
    # Create KasaTool instance
    kasa_tool = KasaTool()
    
    # Mock device.update method to avoid actual network calls
    async def mock_update():
        return None
    mock_device.update = mock_update
    
    # Run discovery to populate the device cache
    kasa_tool.run('discover_devices')
    
    # Verify device is cached
    device_instances = kasa_tool.get_thread_data('device_instances', {})
    assert len(device_instances) >= 1
    
    # Run cleanup
    kasa_tool.cleanup()
    
    # Verify devices are cleaned up
    device_instances = kasa_tool.get_thread_data('device_instances', {})
    assert len(device_instances) == 0


@patch('kasa.Discover')
def test_kasa_tool_with_automation_engine(mock_discover):
    """Test that KasaTool works with AutomationEngine's event loop management."""
    # Setup mock device
    mock_device = MagicMock()
    mock_device.host = '192.168.1.100'
    mock_device.alias = 'Test Device'
    
    # Configure mock discover
    mock_discover.discover = MagicMock()
    mock_discover.discover.return_value = asyncio.Future()
    mock_discover.discover.return_value.set_result({
        '192.168.1.100': mock_device
    })
    
    # Mock device.update method to avoid actual network calls
    async def mock_update():
        return None
    mock_device.update = mock_update
    
    # Create a tool repository with KasaTool
    tool_repo = ToolRepository()
    kasa_tool = KasaTool()
    tool_repo._tools['kasa_tool'] = kasa_tool
    tool_repo._enabled_tools.add('kasa_tool')
    
    # Create automation engine with the tool repo
    engine = AutomationEngine(tool_repo=tool_repo)
    
    # Simulate step execution with KasaTool
    tool_called = False
    
    def simulate_step_execution():
        nonlocal tool_called
        
        # Simulate what the engine would do
        get_or_create_event_loop()
        
        try:
            # Run the tool
            result = kasa_tool.run('discover_devices')
            
            # Verify tool ran successfully
            assert result['success'] is True
            tool_called = True
            
            # Verify device is cached
            device_instances = kasa_tool.get_thread_data('device_instances', {})
            assert len(device_instances) >= 1
            
        finally:
            # Clean up resources
            engine._cleanup_tools('test_automation_id')
            close_event_loop()
    
    # Run simulation in a thread like the engine would
    thread = threading.Thread(target=simulate_step_execution)
    thread.start()
    thread.join()
    
    # Verify the tool was called
    assert tool_called is True