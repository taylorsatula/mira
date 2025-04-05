"""
Kasa Smart Home devices integration tool.

This tool enables the bot to discover and control TP-Link Kasa smart home devices
on the local network, including plugs, bulbs, switches, and more.

Before using this tool, run the kasa_device_discovery.py script to scan
your network and create the device configuration files.
"""
import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context

try:
    from kasa import Discover, Device, Credentials
except ImportError:
    class Credentials:
        pass


class KasaTool(Tool):
    """
    Tool for controlling TP-Link Kasa smart home devices.
    
    This tool provides functionality to:
    1. List available devices 
    2. Turn devices on/off
    3. Get device status and information
    4. Control device-specific features like brightness, color, etc.
    """
    
    name = "kasa_tool"
    description = "Control TP-Link Kasa smart home devices on the local network"
    usage_examples = [
        {
            "input": {
                "operation": "list_devices"
            },
            "output": {
                "devices": [
                    {
                        "identifier": "living_room_lamp",
                        "alias": "Living Room Lamp", 
                        "device_type": "Plug",
                        "host": "192.168.1.100"
                    }
                ]
            }
        },
        {
            "input": {
                "operation": "get_device_info",
                "device_id": "living_room_lamp"
            },
            "output": {
                "alias": "Living Room Lamp",
                "host": "192.168.1.100",
                "model": "HS103",
                "device_type": "Plug",
                "is_on": False
            }
        },
        {
            "input": {
                "operation": "set_device_state",
                "device_id": "living_room_lamp",
                "state": True
            },
            "output": {
                "success": True,
                "message": "Device turned on",
                "device_id": "living_room_lamp",
                "alias": "Living Room Lamp"
            }
        }
    ]
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the Kasa tool with optional login credentials.
        
        Args:
            username: Username for Kasa devices requiring authentication
            password: Password for Kasa devices requiring authentication
        """
        super().__init__()
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.device_mapping_path = "data/tools/kasa_tool/device_mapping.json"
        self.latest_devices_path = "data/tools/kasa_tool/latest.json"
        
    def _load_device_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the device mapping from the configuration file.
        
        Returns:
            Dictionary mapping device identifiers to their information
        
        Raises:
            ToolError: If the mapping file doesn't exist or can't be loaded
        """
        try:
            if not os.path.exists(self.device_mapping_path):
                raise ToolError(
                    f"Device mapping file not found at {self.device_mapping_path}. "
                    "Please run kasa_device_discovery.py first.",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            with open(self.device_mapping_path, 'r') as f:
                return json.load(f)
                
        except json.JSONDecodeError:
            raise ToolError(
                f"Error parsing device mapping file at {self.device_mapping_path}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _get_device_host(self, device_id: str) -> str:
        """
        Get the host address for a device by its identifier.
        
        Args:
            device_id: Device identifier from the mapping file
            
        Returns:
            Host address (IP or hostname)
            
        Raises:
            ToolError: If the device ID is not found
        """
        device_mapping = self._load_device_mapping()
        
        if device_id not in device_mapping:
            available_devices = ", ".join(device_mapping.keys())
            raise ToolError(
                f"Device '{device_id}' not found in device mapping. "
                f"Available devices: {available_devices}",
                ErrorCode.INVALID_TOOL_INPUT
            )
            
        return device_mapping[device_id]["host"]
    
    async def _get_device_info(self, device_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific device.
        
        Args:
            device_id: Device identifier from the mapping file
            
        Returns:
            Dict with device information
        """
        host = self._get_device_host(device_id)
        credentials = None
        if self.username and self.password:
            credentials = Credentials(self.username, self.password)
            
        try:
            device = await Discover.discover_single(
                host,
                credentials=credentials
            )
            
            if not device:
                raise ToolError(
                    f"No device found at {host}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            await device.update()
            
            # Get basic device info
            info = {
                "device_id": device_id,
                "host": host,
                "alias": device.alias,
                "model": device.model,
                "device_type": str(device.device_type).replace("DeviceType.", ""),
                "is_on": device.is_on,
                "mac": device.mac,
                "rssi": device.rssi,
                "features": list(device.features.keys()),
                "modules": list(device.modules.keys())
            }
            
            # Add device-specific info based on type
            if device.has_emeter:
                emeter = device.modules.get("Energy")
                if emeter:
                    if hasattr(emeter, "current_consumption"):
                        info["current_consumption"] = emeter.current_consumption
                    if hasattr(emeter, "consumption_today"):
                        info["consumption_today"] = emeter.consumption_today
            
            # Check for light-specific features
            if "Light" in device.modules:
                light = device.modules["Light"]
                light_features = {}
                
                if hasattr(light, "brightness"):
                    light_features["brightness"] = light.brightness
                    
                if hasattr(light, "color_temp") and light.has_feature("color_temp"):
                    light_features["color_temp"] = light.color_temp
                    
                if hasattr(light, "hsv") and light.has_feature("hsv"):
                    light_features["hsv"] = str(light.hsv)
                    
                if light_features:
                    info["light_features"] = light_features
                    
            await device.disconnect()
            return info
            
        except Exception as ex:
            raise ToolError(
                f"Failed to get info for device {device_id} at {host}: {ex}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_device_state(self, device_id: str, state: bool) -> Dict[str, Any]:
        """
        Turn a device on or off.
        
        Args:
            device_id: Device identifier from the mapping file
            state: True to turn on, False to turn off
            
        Returns:
            Dict with operation result
        """
        host = self._get_device_host(device_id)
        credentials = None
        if self.username and self.password:
            credentials = Credentials(self.username, self.password)
            
        try:
            device = await Discover.discover_single(
                host,
                credentials=credentials
            )
            
            if not device:
                raise ToolError(
                    f"No device found at {host}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Set the state
            if state:
                await device.turn_on()
            else:
                await device.turn_off()
                
            # Get updated state
            await device.update()
            
            result = {
                "success": True,
                "message": f"Device turned {'on' if state else 'off'}",
                "device_id": device_id,
                "host": host,
                "alias": device.alias,
                "is_on": device.is_on,
            }
            
            await device.disconnect()
            return result
            
        except Exception as ex:
            raise ToolError(
                f"Failed to set state for device {device_id} at {host}: {ex}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_brightness(self, device_id: str, brightness: int) -> Dict[str, Any]:
        """
        Set brightness for a light device.
        
        Args:
            device_id: Device identifier from the mapping file
            brightness: Brightness level (0-100)
            
        Returns:
            Dict with operation result
        """
        host = self._get_device_host(device_id)
        credentials = None
        if self.username and self.password:
            credentials = Credentials(self.username, self.password)
            
        try:
            device = await Discover.discover_single(
                host,
                credentials=credentials
            )
            
            if not device:
                raise ToolError(
                    f"No device found at {host}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            if "Light" not in device.modules:
                raise ToolError(
                    f"Device {device_id} at {host} does not support brightness control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            light = device.modules["Light"]
            
            if not hasattr(light, "set_brightness"):
                raise ToolError(
                    f"Device {device_id} at {host} does not support brightness control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Set brightness
            await light.set_brightness(brightness)
                
            # Get updated state
            await device.update()
            
            result = {
                "success": True,
                "message": f"Brightness set to {brightness}",
                "device_id": device_id,
                "host": host,
                "alias": device.alias,
                "brightness": light.brightness,
            }
            
            await device.disconnect()
            return result
            
        except Exception as ex:
            raise ToolError(
                f"Failed to set brightness for device {device_id} at {host}: {ex}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_color_temp(self, device_id: str, color_temp: int) -> Dict[str, Any]:
        """
        Set color temperature for a light device.
        
        Args:
            device_id: Device identifier from the mapping file
            color_temp: Color temperature in Kelvin
            
        Returns:
            Dict with operation result
        """
        host = self._get_device_host(device_id)
        credentials = None
        if self.username and self.password:
            credentials = Credentials(self.username, self.password)
            
        try:
            device = await Discover.discover_single(
                host,
                credentials=credentials
            )
            
            if not device:
                raise ToolError(
                    f"No device found at {host}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            if "Light" not in device.modules:
                raise ToolError(
                    f"Device {device_id} at {host} does not support color temperature control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            light = device.modules["Light"]
            
            if not light.has_feature("color_temp"):
                raise ToolError(
                    f"Device {device_id} at {host} does not support color temperature control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Set color temperature
            await light.set_color_temp(color_temp)
                
            # Get updated state
            await device.update()
            
            result = {
                "success": True,
                "message": f"Color temperature set to {color_temp}K",
                "device_id": device_id,
                "host": host,
                "alias": device.alias,
                "color_temp": light.color_temp,
            }
            
            await device.disconnect()
            return result
            
        except Exception as ex:
            raise ToolError(
                f"Failed to set color temperature for device {device_id} at {host}: {ex}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_hsv(self, device_id: str, hue: int, saturation: int, value: int) -> Dict[str, Any]:
        """
        Set HSV color for a light device.
        
        Args:
            device_id: Device identifier from the mapping file
            hue: Hue (0-360)
            saturation: Saturation (0-100)
            value: Value/brightness (0-100)
            
        Returns:
            Dict with operation result
        """
        host = self._get_device_host(device_id)
        credentials = None
        if self.username and self.password:
            credentials = Credentials(self.username, self.password)
            
        try:
            device = await Discover.discover_single(
                host,
                credentials=credentials
            )
            
            if not device:
                raise ToolError(
                    f"No device found at {host}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            if "Light" not in device.modules:
                raise ToolError(
                    f"Device {device_id} at {host} does not support color control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            light = device.modules["Light"]
            
            if not light.has_feature("hsv"):
                raise ToolError(
                    f"Device {device_id} at {host} does not support HSV color control",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Set HSV
            await light.set_hsv(hue, saturation, value)
                
            # Get updated state
            await device.update()
            
            result = {
                "success": True,
                "message": f"Color set to HSV({hue}, {saturation}, {value})",
                "device_id": device_id,
                "host": host,
                "alias": device.alias,
                "hsv": str(light.hsv),
            }
            
            await device.disconnect()
            return result
            
        except Exception as ex:
            raise ToolError(
                f"Failed to set HSV color for device {device_id} at {host}: {ex}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def run(
        self,
        operation: str,
        device_id: Optional[str] = None,
        state: Optional[bool] = None,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        hue: Optional[int] = None,
        saturation: Optional[int] = None,
        value: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Kasa device operation.
        
        Args:
            operation: Operation to perform (list_devices, get_device_info, set_device_state, 
                       set_brightness, set_color_temp, set_hsv)
            device_id: Device identifier from the mapping file (required for device-specific operations)
            state: Device state to set (True for on, False for off)
            brightness: Brightness level (0-100)
            color_temp: Color temperature in Kelvin
            hue: Hue (0-360)
            saturation: Saturation (0-100)
            value: Value/brightness (0-100)
            
        Returns:
            Dict with operation result
            
        Raises:
            ToolError: If operation fails
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Run the appropriate operation
            if operation == "list_devices":
                device_mapping = self._load_device_mapping()
                devices_list = []
                
                for identifier, device in device_mapping.items():
                    devices_list.append({
                        "identifier": identifier,
                        "alias": device["alias"],
                        "device_type": device["device_type"],
                        "host": device["host"],
                        "model": device.get("model", "Unknown")
                    })
                
                return {"devices": devices_list}
                
            elif operation == "get_device_info":
                if not device_id:
                    raise ToolError(
                        "device_id parameter is required for get_device_info operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                result = asyncio.run(self._get_device_info(device_id))
                return result
                
            elif operation == "set_device_state":
                if not device_id:
                    raise ToolError(
                        "device_id parameter is required for set_device_state operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if state is None:
                    raise ToolError(
                        "state parameter is required for set_device_state operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                result = asyncio.run(self._set_device_state(device_id, state))
                return result
                
            elif operation == "set_brightness":
                if not device_id:
                    raise ToolError(
                        "device_id parameter is required for set_brightness operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if brightness is None:
                    raise ToolError(
                        "brightness parameter is required for set_brightness operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if not 0 <= brightness <= 100:
                    raise ToolError(
                        "brightness must be between 0 and 100",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                result = asyncio.run(self._set_brightness(device_id, brightness))
                return result
                
            elif operation == "set_color_temp":
                if not device_id:
                    raise ToolError(
                        "device_id parameter is required for set_color_temp operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if color_temp is None:
                    raise ToolError(
                        "color_temp parameter is required for set_color_temp operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                result = asyncio.run(self._set_color_temp(device_id, color_temp))
                return result
                
            elif operation == "set_hsv":
                if not device_id:
                    raise ToolError(
                        "device_id parameter is required for set_hsv operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if hue is None or saturation is None or value is None:
                    raise ToolError(
                        "hue, saturation, and value parameters are required for set_hsv operation",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if not 0 <= hue <= 360:
                    raise ToolError(
                        "hue must be between 0 and 360",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if not 0 <= saturation <= 100:
                    raise ToolError(
                        "saturation must be between 0 and 100",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                if not 0 <= value <= 100:
                    raise ToolError(
                        "value must be between 0 and 100",
                        ErrorCode.INVALID_TOOL_INPUT
                    )
                result = asyncio.run(self._set_hsv(device_id, hue, saturation, value))
                return result
                
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: list_devices, get_device_info, "
                    "set_device_state, set_brightness, set_color_temp, set_hsv",
                    ErrorCode.INVALID_TOOL_INPUT
                )