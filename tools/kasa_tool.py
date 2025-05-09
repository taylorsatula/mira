"""
Tool for controlling TP-Link Kasa smart home devices.

This tool provides an interface to discover and control Kasa smart home devices
on the local network including plugs, bulbs, switches, light strips, and multi-outlet
power strips.

Datetime handling follows the UTC-everywhere approach:
- All datetimes are stored in UTC internally
- Timezone-aware datetime objects are used consistently
- Conversion to local time happens only when displaying to users
- The utility functions from utils.timezone_utils are used consistently
"""

# Standard library imports
import asyncio
import json
import logging
import os
import hashlib
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta

# Third-party imports
import aiohttp
from pydantic import BaseModel, Field

# Import timezone utilities for UTC-everywhere approach
from utils.timezone_utils import utc_now, ensure_utc

# Local imports
from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry


# -------------------- CONFIGURATION --------------------

class KasaToolConfig(BaseModel):
    """
    Configuration for the kasa_tool.

    Defines the parameters that control the Kasa tool's behavior,
    including device discovery, caching, and authentication.
    """
    # Standard configuration parameter - all tools should include this
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled by default"
    )

    # Caching configuration
    cache_enabled: bool = Field(
        default=True,
        description="Whether to cache device information"
    )
    cache_duration: int = Field(
        default=3600,
        description="Cache duration in seconds (default: 1 hour)"
    )
    cache_directory: str = Field(
        default="data/tools/kasa_tool/cache",
        description="Directory to store cached device data"
    )

    # Device mapping configuration
    device_mapping_enabled: bool = Field(
        default=True,
        description="Whether to use device mapping file for device identification"
    )
    device_mapping_path: str = Field(
        default="data/tools/kasa_tool/device_mapping.json",
        description="Path to the device mapping file"
    )

    # Discovery configuration
    discovery_timeout: int = Field(
        default=5,
        description="Timeout in seconds for device discovery"
    )
    discovery_target: str = Field(
        default="255.255.255.255",
        description="Default target for device discovery"
    )
    attempt_discovery_when_not_found: bool = Field(
        default=True,
        description="Whether to attempt discovery when a device is not found in cache"
    )

    # Authentication
    default_username: str = Field(
        default="",
        description="Default username for devices requiring authentication"
    )
    default_password: str = Field(
        default="",
        description="Default password for devices requiring authentication"
    )

    # Operation settings
    verify_changes: bool = Field(
        default=True,
        description="Whether to verify changes after performing operations"
    )
    verification_attempts: int = Field(
        default=3,
        description="Number of attempts to verify changes"
    )
    verification_delay: float = Field(
        default=0.5,
        description="Delay in seconds between verification attempts"
    )

# Register with registry
registry.register("kasa_tool", KasaToolConfig)


# -------------------- CACHE MANAGER --------------------

class DeviceCache:
    """
    Manages caching of discovered devices to minimize network operations.
    """
    
    def __init__(self, cache_dir: str, cache_duration: int):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Cache validity duration in seconds
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """
        Get the cache file path for a given key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key for the filename to avoid invalid characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def is_valid(self, cache_path: str) -> bool:
        """
        Check if a cache file is valid and not expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is expired based on file modification time
        # Use utc_now timestamp for consistent timezone handling
        cache_mtime = os.path.getmtime(cache_path)
        cache_age = utc_now().timestamp() - cache_mtime
        return cache_age < self.cache_duration
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if available and valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        cache_path = self.get_cache_path(key)
        
        # Check if cache is valid
        if not self.is_valid(cache_path):
            return None
            
        # Read from cache
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {key}: {str(e)}")
            return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self.get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to cache data for {key}: {str(e)}")


# -------------------- MAIN TOOL CLASS --------------------

class KasaTool(Tool):
    """
    Tool for controlling TP-Link Kasa smart home devices.
    
    This tool provides functionality to discover and control Kasa smart home devices
    on your local network including plugs, switches, bulbs, light strips, and multi-outlet
    power strips.
    """
    
    name = "kasa_tool"
    
    description = """
    Controls TP-Link Kasa smart home devices on your local network. Use this tool to discover
    and manage Kasa devices like smart plugs, light bulbs, switches, and light strips.
    
    OPERATIONS:
    - discover_devices: Find all Kasa devices on your local network
      Parameters:
        target (optional): The network address to target for discovery (default: 255.255.255.255)
        timeout (optional): Timeout in seconds for discovery (default: 5)
        username (optional): Username for devices requiring authentication
        password (optional): Password for devices requiring authentication
        
    - get_device_info: Get detailed information about a specific device
      Parameters:
        device_id (required): The device identifier (IP address, hostname, or alias)
        
    - power_control: Turn a device on or off
      Parameters:
        device_id (required): The device identifier
        state (required): The desired state ("on" or "off")
        
    - set_brightness: Set the brightness of a light bulb or light strip
      Parameters:
        device_id (required): The device identifier
        brightness (required): Brightness level (0-100)
        
    - set_color: Set the color of a light bulb or light strip
      Parameters:
        device_id (required): The device identifier
        hue (required): Hue value (0-360)
        saturation (required): Saturation value (0-100)
        value (optional): Brightness value (0-100)
        
    - set_color_temp: Set the color temperature of a light bulb
      Parameters:
        device_id (required): The device identifier
        temperature (required): Color temperature in Kelvin
        
    - get_energy_usage: Get energy usage data for supported devices
      Parameters:
        device_id (required): The device identifier
        period (optional): Period to retrieve ("realtime", "today", "month")
        
    - set_device_alias: Set a new name for the device
      Parameters:
        device_id (required): The device identifier
        alias (required): New name for the device
        
    - get_child_devices: Get information about child devices for power strips
      Parameters:
        device_id (required): The device identifier of the parent device
        
    - control_child_device: Control a specific outlet on a power strip
      Parameters:
        device_id (required): The device identifier of the parent device
        child_id (required): The ID or index of the child device
        state (required): The desired state ("on" or "off")
    
    RESPONSE FORMAT:
    - All operations return a success flag and operation-specific data
    
    LIMITATIONS:
    - Only works with devices on the same local network
    - Some operations may require device-specific parameters
    - Some devices may require authentication
    """
    
    usage_examples = [
        {
            "input": {
                "operation": "discover_devices"
            },
            "output": {
                "success": True,
                "devices": [
                    {
                        "id": "192.168.1.100",
                        "alias": "Living Room Light",
                        "model": "KL130",
                        "type": "bulb",
                        "state": "on",
                        "features": ["brightness", "color", "temperature"]
                    },
                    {
                        "id": "192.168.1.101",
                        "alias": "Office Plug",
                        "model": "HS110",
                        "type": "plug",
                        "state": "off",
                        "features": ["energy"]
                    }
                ],
                "message": "Found 2 Kasa devices on the network"
            }
        },
        {
            "input": {
                "operation": "power_control",
                "device_id": "Living Room Light",
                "state": "off"
            },
            "output": {
                "success": True,
                "device": {
                    "id": "192.168.1.100",
                    "alias": "Living Room Light",
                    "model": "KL130",
                    "state": "off"
                },
                "message": "Turned Living Room Light off"
            }
        },
        {
            "input": {
                "operation": "set_brightness",
                "device_id": "Living Room Light",
                "brightness": 75
            },
            "output": {
                "success": True,
                "device": {
                    "id": "192.168.1.100",
                    "alias": "Living Room Light",
                    "state": "on",
                    "brightness": 75
                },
                "message": "Set brightness of Living Room Light to 75%"
            }
        }
    ]
    
    def __init__(self):
        """Initialize the Kasa tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("KasaTool initialized")

        # Create required directories
        from config import config
        os.makedirs(config.kasa_tool.cache_directory, exist_ok=True)

        # Create directory for device mapping if it doesn't exist
        mapping_dir = os.path.dirname(config.kasa_tool.device_mapping_path)
        os.makedirs(mapping_dir, exist_ok=True)

        # Initialize device cache
        self.cache = DeviceCache(
            config.kasa_tool.cache_directory,
            config.kasa_tool.cache_duration
        )

        # In-memory device storage
        self._device_instances = {}

        # Load device mapping if available
        self._device_mapping = self._load_device_mapping()
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a Kasa tool operation.
        
        Args:
            operation: The operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            Dict containing the operation results
            
        Raises:
            ToolError: If operation fails or parameters are invalid
        """
        with error_context(
            component_name=self.name,
            operation=operation,
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Route to the appropriate operation
            if operation == "discover_devices":
                return asyncio.run(self._discover_devices(**kwargs))
            elif operation == "get_device_info":
                return asyncio.run(self._get_device_info(**kwargs))
            elif operation == "power_control":
                return asyncio.run(self._power_control(**kwargs))
            elif operation == "set_brightness":
                return asyncio.run(self._set_brightness(**kwargs))
            elif operation == "set_color":
                return asyncio.run(self._set_color(**kwargs))
            elif operation == "set_color_temp":
                return asyncio.run(self._set_color_temp(**kwargs))
            elif operation == "get_energy_usage":
                return asyncio.run(self._get_energy_usage(**kwargs))
            elif operation == "set_device_alias":
                return asyncio.run(self._set_device_alias(**kwargs))
            elif operation == "get_child_devices":
                return asyncio.run(self._get_child_devices(**kwargs))
            elif operation == "control_child_device":
                return asyncio.run(self._control_child_device(**kwargs))
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "discover_devices, get_device_info, power_control, set_brightness, "
                    "set_color, set_color_temp, get_energy_usage, set_device_alias, "
                    "get_child_devices, control_child_device",
                    ErrorCode.TOOL_INVALID_INPUT
                )
    
    async def _discover_devices(
        self,
        target: Optional[str] = None,
        timeout: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover Kasa devices on the local network.
        
        Args:
            target: The network address to target for discovery
            timeout: Timeout in seconds for discovery
            username: Username for devices requiring authentication
            password: Password for devices requiring authentication
            
        Returns:
            Dict containing information about discovered devices
        """
        from kasa import Discover, Credentials
        from config import config
        
        # Use default values from config if not provided
        target = target or config.kasa_tool.discovery_target
        timeout = timeout or config.kasa_tool.discovery_timeout
        
        # Process credentials
        credentials = None
        if username and password:
            credentials = Credentials(username, password)
        elif config.kasa_tool.default_username and config.kasa_tool.default_password:
            credentials = Credentials(
                config.kasa_tool.default_username,
                config.kasa_tool.default_password
            )
        
        self.logger.info(f"Discovering Kasa devices on {target} with timeout {timeout}s")
        
        try:
            # Perform device discovery
            found_devices = await Discover.discover(
                target=target,
                discovery_timeout=timeout,
                credentials=credentials
            )
            
            # Update devices after discovery to get full information
            device_details = []
            for device in found_devices.values():
                try:
                    await device.update()
                    self._cache_device(device)
                    device_details.append(self._serialize_device_summary(device))
                except Exception as e:
                    self.logger.warning(f"Error updating device {device.host}: {e}")
            
            return {
                "success": True,
                "devices": device_details,
                "message": f"Found {len(device_details)} Kasa device(s) on the network"
            }
        except Exception as e:
            raise ToolError(
                f"Error discovering devices: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _get_device_info(self, device_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific device.
        
        Args:
            device_id: The device identifier (IP address, hostname, or alias)
            
        Returns:
            Dict containing device information
        """
        self.logger.info(f"Getting info for device: {device_id}")
        
        try:
            # Get device and ensure it's updated
            device = await self._get_device_by_id(device_id)
            await device.update()
            
            # Cache the device after update
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_details(device),
                "message": f"Retrieved information for {device.alias or device.host}"
            }
        except Exception as e:
            raise ToolError(
                f"Error getting device info: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _power_control(self, device_id: str, state: str) -> Dict[str, Any]:
        """
        Turn a device on or off.
        
        Args:
            device_id: The device identifier
            state: The desired state ("on" or "off")
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting power state for {device_id} to {state}")
        
        # Validate state
        if state.lower() not in ["on", "off"]:
            raise ToolError(
                f"Invalid state: {state}. Must be 'on' or 'off'",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Set the state
            if state.lower() == "on":
                await device.turn_on()
            else:
                await device.turn_off()
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    device, 
                    state.lower() == "on", 
                    "is_on",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Turned {device.alias or device.host} {state.lower()}"
            }
        except Exception as e:
            raise ToolError(
                f"Error controlling device: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_brightness(self, device_id: str, brightness: Union[int, str]) -> Dict[str, Any]:
        """
        Set the brightness of a light bulb or light strip.
        
        Args:
            device_id: The device identifier
            brightness: Brightness level (0-100)
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting brightness for {device_id} to {brightness}")
        
        # Validate and convert brightness
        try:
            brightness_value = int(brightness)
            if brightness_value < 0 or brightness_value > 100:
                raise ValueError("Brightness must be between 0 and 100")
        except (ValueError, TypeError) as e:
            raise ToolError(
                f"Invalid brightness value: {brightness}. Must be an integer between 0 and 100",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports brightness
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_brightness'):
                raise ToolError(
                    f"Device {device.alias or device.host} does not support brightness control",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Set brightness
            await light_module.set_brightness(brightness_value)
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    light_module, 
                    brightness_value, 
                    "brightness",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set brightness of {device.alias or device.host} to {brightness_value}%"
            }
        except Exception as e:
            raise ToolError(
                f"Error setting brightness: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_color(
        self, 
        device_id: str, 
        hue: Union[int, str], 
        saturation: Union[int, str], 
        value: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Set the color of a light bulb or light strip.
        
        Args:
            device_id: The device identifier
            hue: Hue value (0-360)
            saturation: Saturation value (0-100)
            value: Brightness value (0-100)
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting color for {device_id} to H:{hue} S:{saturation} V:{value}")
        
        # Validate and convert color values
        try:
            hue_value = int(hue)
            if hue_value < 0 or hue_value > 360:
                raise ValueError("Hue must be between 0 and 360")
                
            saturation_value = int(saturation)
            if saturation_value < 0 or saturation_value > 100:
                raise ValueError("Saturation must be between 0 and 100")
                
            if value is not None:
                value_value = int(value)
                if value_value < 0 or value_value > 100:
                    raise ValueError("Value must be between 0 and 100")
            else:
                value_value = None
        except (ValueError, TypeError) as e:
            raise ToolError(
                f"Invalid color values: {str(e)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports color
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_hsv'):
                raise ToolError(
                    f"Device {device.alias or device.host} does not support color control",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Set HSV
            if value_value is not None:
                await light_module.set_hsv(hue_value, saturation_value, value_value)
            else:
                # Use current brightness if value not provided
                current_hsv = getattr(light_module, 'hsv', None)
                current_value = current_hsv.value if current_hsv else 100
                await light_module.set_hsv(hue_value, saturation_value, current_value)
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set color of {device.alias or device.host} to HSV({hue_value}, {saturation_value}, {value_value or current_value})"
            }
        except Exception as e:
            raise ToolError(
                f"Error setting color: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_color_temp(self, device_id: str, temperature: Union[int, str]) -> Dict[str, Any]:
        """
        Set the color temperature of a light bulb.
        
        Args:
            device_id: The device identifier
            temperature: Color temperature in Kelvin
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting color temperature for {device_id} to {temperature}K")
        
        # Validate and convert temperature
        try:
            temperature_value = int(temperature)
            if temperature_value < 2500 or temperature_value > 9000:
                raise ValueError("Temperature must be between 2500K and 9000K")
        except (ValueError, TypeError) as e:
            raise ToolError(
                f"Invalid temperature value: {temperature}. Must be an integer between 2500 and 9000",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports color temperature
            light_module = None
            if "Light" in device.modules:
                light_module = device.modules["Light"]
            
            if not light_module or not hasattr(light_module, 'set_color_temp'):
                raise ToolError(
                    f"Device {device.alias or device.host} does not support color temperature control",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Set color temperature
            await light_module.set_color_temp(temperature_value)
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    light_module, 
                    temperature_value, 
                    "color_temp",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Set color temperature of {device.alias or device.host} to {temperature_value}K"
            }
        except Exception as e:
            raise ToolError(
                f"Error setting color temperature: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _get_energy_usage(self, device_id: str, period: str = "realtime") -> Dict[str, Any]:
        """
        Get energy usage data for supported devices.
        
        Args:
            device_id: The device identifier
            period: Period to retrieve ("realtime", "today", "month")
            
        Returns:
            Dict containing energy usage data
        """
        self.logger.info(f"Getting energy usage for {device_id} for period: {period}")
        
        # Validate period
        valid_periods = ["realtime", "today", "month"]
        if period.lower() not in valid_periods:
            raise ToolError(
                f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device supports energy monitoring
            if not device.has_emeter:
                raise ToolError(
                    f"Device {device.alias or device.host} does not support energy monitoring",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Get the energy module
            energy_module = None
            if "Energy" in device.modules:
                energy_module = device.modules["Energy"]
            else:
                raise ToolError(
                    f"Device {device.alias or device.host} does not have energy module",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Get energy data based on period
            energy_data = {}
            if period.lower() == "realtime":
                await energy_module.update()  # Ensure we have the latest data
                energy_data = {
                    "current_power": energy_module.current_consumption,
                    "voltage": getattr(energy_module, 'voltage', None),
                    "current": getattr(energy_module, 'current', None)
                }
                message = f"Current power usage for {device.alias or device.host} is {energy_data['current_power']}W"
            elif period.lower() == "today":
                energy_data = {
                    "consumption_today": energy_module.consumption_today
                }
                message = f"Energy consumption today for {device.alias or device.host} is {energy_data['consumption_today']}kWh"
            elif period.lower() == "month":
                energy_data = {
                    "consumption_month": energy_module.consumption_this_month
                }
                message = f"Energy consumption this month for {device.alias or device.host} is {energy_data['consumption_month']}kWh"
            
            return {
                "success": True,
                "device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "energy_data": energy_data,
                "message": message
            }
        except Exception as e:
            raise ToolError(
                f"Error getting energy usage: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _set_device_alias(self, device_id: str, alias: str) -> Dict[str, Any]:
        """
        Set a new name for the device.
        
        Args:
            device_id: The device identifier
            alias: New name for the device
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Setting alias for {device_id} to {alias}")
        
        # Validate alias
        if not alias or not isinstance(alias, str):
            raise ToolError(
                "Alias must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Save old alias for the response message
            old_alias = device.alias or device.host
            
            # Set new alias
            await device.set_alias(alias)
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "device": self._serialize_device_summary(device),
                "message": f"Renamed device from '{old_alias}' to '{device.alias}'"
            }
        except Exception as e:
            raise ToolError(
                f"Error setting device alias: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _get_child_devices(self, device_id: str) -> Dict[str, Any]:
        """
        Get information about child devices for power strips.
        
        Args:
            device_id: The device identifier of the parent device
            
        Returns:
            Dict containing information about child devices
        """
        self.logger.info(f"Getting child devices for {device_id}")
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device has children
            if not hasattr(device, 'children') or not device.children:
                raise ToolError(
                    f"Device {device.alias or device.host} does not have child devices",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Get information about child devices
            child_devices = []
            for child in device.children:
                child_devices.append(self._serialize_device_summary(child))
            
            return {
                "success": True,
                "parent_device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "child_devices": child_devices,
                "message": f"Found {len(child_devices)} child devices for {device.alias or device.host}"
            }
        except Exception as e:
            raise ToolError(
                f"Error getting child devices: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    async def _control_child_device(
        self, 
        device_id: str, 
        child_id: str, 
        state: str
    ) -> Dict[str, Any]:
        """
        Control a specific outlet on a power strip.
        
        Args:
            device_id: The device identifier of the parent device
            child_id: The ID or index of the child device
            state: The desired state ("on" or "off")
            
        Returns:
            Dict containing the operation result
        """
        self.logger.info(f"Controlling child device {child_id} of {device_id} with state {state}")
        
        # Validate state
        if state.lower() not in ["on", "off"]:
            raise ToolError(
                f"Invalid state: {state}. Must be 'on' or 'off'",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            device = await self._get_device_by_id(device_id)
            
            # Check if device has children
            if not hasattr(device, 'children') or not device.children:
                raise ToolError(
                    f"Device {device.alias or device.host} does not have child devices",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Get child device by ID or index
            child_device = None
            
            # First try to get by alias
            child_device = device.get_plug_by_name(child_id)
            
            # If not found by alias, try by index
            if not child_device and child_id.isdigit():
                try:
                    index = int(child_id)
                    child_device = device.get_plug_by_index(index)
                except Exception:
                    pass
            
            if not child_device:
                raise ToolError(
                    f"Child device {child_id} not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
            
            # Set the state
            if state.lower() == "on":
                await child_device.turn_on()
            else:
                await child_device.turn_off()
            
            # Verify the change
            from config import config
            if config.kasa_tool.verify_changes:
                await self._verify_change(
                    child_device, 
                    state.lower() == "on", 
                    "is_on",
                    config.kasa_tool.verification_attempts,
                    config.kasa_tool.verification_delay
                )
            
            # Update the device to get current state
            await device.update()
            
            # Cache the updated device
            self._cache_device(device)
            
            return {
                "success": True,
                "parent_device": {
                    "id": device.host,
                    "alias": device.alias,
                    "model": device.model
                },
                "child_device": self._serialize_device_summary(child_device),
                "message": f"Turned {child_device.alias or child_id} {state.lower()}"
            }
        except Exception as e:
            raise ToolError(
                f"Error controlling child device: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # -------------------- HELPER METHODS --------------------

    def _load_device_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the device mapping from file.

        Returns:
            Dictionary mapping device identifiers to their information
        """
        from config import config

        mapping_path = config.kasa_tool.device_mapping_path

        if not os.path.exists(mapping_path):
            self.logger.info(f"Device mapping file not found at {mapping_path}")
            return {}

        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                self.logger.info(f"Loaded {len(mapping)} devices from mapping file")
                return mapping
        except Exception as e:
            self.logger.warning(f"Error loading device mapping: {e}")
            return {}

    def _find_best_device_match(self, query: str) -> str:
        """
        Find the best device match in the in-memory cache based on name similarity.

        Args:
            query: The device name or identifier to match

        Returns:
            The best matching device identifier

        Raises:
            ToolError: If no match found or multiple ambiguous matches found
        """
        # Return exact match if exists
        if query in self._device_instances:
            return query

        # Normalize the query for better matching
        norm_query = query.lower().replace("the ", "").strip()

        # Track potential matches with scores
        candidates = {}

        # Look through cached devices for matches
        for device_id, device in self._device_instances.items():
            # Skip IP addresses - we're looking for name matches
            if '.' in device_id and device_id.replace('.', '').isdigit():
                continue

            # Skip empty device IDs
            if not device_id:
                continue

            # Get the device alias if available
            alias = device.alias if hasattr(device, 'alias') else ""
            if not alias:
                continue

            # Normalize the alias
            norm_alias = alias.lower().replace("the ", "").strip()

            # Score the match (higher is better)
            score = 0

            # Exact match gets highest score
            if norm_query == norm_alias:
                score = 100
            # Contained words get medium scores
            elif norm_query in norm_alias:
                score = 80
            elif norm_alias in norm_query:
                score = 70
            else:
                # Calculate word overlap
                query_words = set(norm_query.split())
                alias_words = set(norm_alias.split())

                if query_words and alias_words:
                    common_words = query_words.intersection(alias_words)
                    total_words = len(query_words.union(alias_words))

                    if common_words:
                        # Percentage of matching words
                        score = 60 * len(common_words) / total_words

            # Record candidate if score is above threshold
            if score > 50:
                # Store both score and device_id -> alias mapping
                candidates[device_id] = {"score": score, "alias": alias}
                self.logger.debug(f"Match candidate: '{device_id}' (alias: '{alias}', score: {score})")

        # If no candidates found
        if not candidates:
            raise ToolError(
                f"No device found matching '{query}'",
                ErrorCode.TOOL_NOT_FOUND
            )

        # Sort candidates by score (highest first)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

        # Check for clear winner vs ambiguous matches
        if len(sorted_candidates) == 1:
            # Only one match - return it
            best_match_id = sorted_candidates[0][0]
            alias = candidates[best_match_id]["alias"]
            score = candidates[best_match_id]["score"]
            self.logger.info(f"Found match for '{query}': '{alias}' (device_id: '{best_match_id}', score: {score})")
            return best_match_id

        # Check if there's a clear winner (significantly higher score)
        best_score = sorted_candidates[0][1]["score"]
        second_best_score = sorted_candidates[1][1]["score"] if len(sorted_candidates) > 1 else 0

        if best_score > second_best_score + 20:  # Clear winner if 20+ points higher
            best_match_id = sorted_candidates[0][0]
            alias = candidates[best_match_id]["alias"]
            self.logger.info(f"Found best match for '{query}': '{alias}' (device_id: '{best_match_id}', score: {best_score})")
            return best_match_id

        # Ambiguous matches - suggest options
        top_matches = [candidates[candidate[0]]["alias"] for candidate in sorted_candidates[:3]]
        suggestion = ", ".join([f"'{match}'" for match in top_matches])

        raise ToolError(
            f"Ambiguous device name '{query}'. Did you mean one of: {suggestion}?",
            ErrorCode.TOOL_AMBIGUOUS_INPUT
        )
    
    async def _get_device_by_id(
        self,
        device_id: str,
        force_discovery: bool = False
    ) -> Any:
        """
        Get a device by its identifier, using cache when available.

        Args:
            device_id: The device identifier (IP address, hostname, or alias)
            force_discovery: Whether to force device discovery

        Returns:
            Device object

        Raises:
            ToolError: If device cannot be found
        """
        from kasa import Device, Credentials, Discover
        from config import config

        # Check for cached instance first
        if device_id in self._device_instances and not force_discovery:
            device = self._device_instances[device_id]
            try:
                await device.update()
                self.logger.info(f"Using cached device: {device_id}")
                return device
            except Exception as e:
                self.logger.warning(f"Error updating cached device {device_id}: {e}")
                # Continue to try other methods if update fails

        # Try fuzzy matching if we have cached devices and not forcing discovery
        if self._device_instances and not force_discovery:
            try:
                # Find best match based on name similarity
                best_match_id = self._find_best_device_match(device_id)
                if best_match_id in self._device_instances:
                    device = self._device_instances[best_match_id]
                    try:
                        await device.update()
                        alias = device.alias if hasattr(device, 'alias') else best_match_id
                        self.logger.info(f"Using fuzzy-matched device: '{device_id}' -> '{alias}'")
                        # Cache this match for future use
                        self._device_instances[device_id] = device
                        return device
                    except Exception as e:
                        self.logger.warning(f"Error updating fuzzy-matched device {best_match_id}: {e}")
                else:
                    self.logger.warning(f"Found fuzzy match '{best_match_id}' but device instance not available")
            except ToolError as e:
                # Log but continue to other methods if fuzzy matching fails or is ambiguous
                self.logger.info(f"Fuzzy matching: {str(e)}")
        
        # Setup credentials for possible use later
        credentials = None
        if config.kasa_tool.default_username and config.kasa_tool.default_password:
            credentials = Credentials(
                config.kasa_tool.default_username,
                config.kasa_tool.default_password
            )

        # Only do direct connection if forcing discovery or if device ID looks like an IP address
        if force_discovery or ('.' in device_id and not device_id.startswith('.')):
            try:
                self.logger.info(f"Attempting direct connection to: {device_id}")
                # Try to connect directly
                device = await Discover.discover_single(
                    device_id,
                    credentials=credentials
                )

                if device:
                    self._device_instances[device_id] = device
                    self._device_instances[device.host] = device
                    if device.alias:
                        self._device_instances[device.alias] = device
                    return device
            except Exception as e:
                self.logger.warning(f"Error connecting directly to {device_id}: {e}")
                # Continue to try other methods if connection fails
        
        # Check for cached devices by alias
        # Loop through all devices and try to match by alias
        devices_to_check = []

        # Get from cache if available and enabled
        if config.kasa_tool.cache_enabled and not force_discovery:
            self.logger.info(f"Checking persistent cache for device: {device_id}")
            # Get all cached devices
            cache_dir = config.kasa_tool.cache_directory
            if os.path.exists(cache_dir):
                for cache_file in os.listdir(cache_dir):
                    if cache_file.endswith('.json'):
                        cache_path = os.path.join(cache_dir, cache_file)
                        try:
                            with open(cache_path, 'r') as f:
                                device_data = json.load(f)
                                if device_data.get('host') or device_data.get('alias'):
                                    devices_to_check.append(device_data)
                        except Exception as e:
                            self.logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        # Check if any cached device matches by alias
        for device_data in devices_to_check:
            if (device_data.get('alias') == device_id or 
                device_data.get('host') == device_id):
                try:
                    # Try to connect to the device
                    device = await Device.connect(
                        host=device_data.get('host'),
                        config=None
                    )
                    
                    if device:
                        # Log that we're using a device from persistent cache
                        self.logger.info(f"Using device from persistent cache: {device_id}")
                        self._device_instances[device_id] = device
                        self._device_instances[device.host] = device
                        if device.alias:
                            self._device_instances[device.alias] = device
                        return device
                except Exception as e:
                    self.logger.warning(f"Error connecting to cached device {device_data.get('host')}: {e}")
                    # Continue to next cached device
        
        # If still not found and discovery is allowed
        if config.kasa_tool.attempt_discovery_when_not_found or force_discovery:
            try:
                # Only do network discovery as a last resort when cache fails
                self.logger.info(f"Device {device_id} not found in cache. Running network discovery.")
                # Perform device discovery
                found_devices = await Discover.discover(
                    target=config.kasa_tool.discovery_target,
                    discovery_timeout=config.kasa_tool.discovery_timeout,
                    credentials=credentials
                )
                
                # Check if any discovered device matches by alias
                for host, device in found_devices.items():
                    try:
                        await device.update()
                        
                        # Cache all discovered devices
                        self._cache_device(device)
                        
                        # Add to in-memory cache
                        self._device_instances[host] = device
                        if device.alias:
                            self._device_instances[device.alias] = device
                            
                        # Check if this device matches the requested ID
                        if device.alias == device_id or host == device_id:
                            return device
                    except Exception as e:
                        self.logger.warning(f"Error updating discovered device {host}: {e}")
                        # Continue to next discovered device
            except Exception as e:
                self.logger.warning(f"Error during device discovery: {e}")
                # Continue to final error
        
        # If we get here, device was not found
        raise ToolError(
            f"Device '{device_id}' not found in cache or network. Make sure it is connected to your network or try discover_devices operation first.",
            ErrorCode.TOOL_NOT_FOUND
        )
    
    def _serialize_device_summary(self, device: Any) -> Dict[str, Any]:
        """
        Serialize basic device information for response.
        
        Args:
            device: The device object
            
        Returns:
            Dict containing basic device information
        """
        # Basic device information
        result = {
            "id": device.host,
            "alias": device.alias,
            "model": device.model,
            "state": "on" if device.is_on else "off"
        }
        
        # Add device type
        device_type = str(device.device_type).lower()
        if 'bulb' in device_type:
            result["type"] = "bulb"
        elif 'plug' in device_type:
            result["type"] = "plug"
        elif 'strip' in device_type:
            result["type"] = "strip"
        elif 'dimmer' in device_type or 'switch' in device_type:
            result["type"] = "switch"
        elif 'lightstrip' in device_type:
            result["type"] = "lightstrip"
        else:
            result["type"] = "other"
        
        # Add supported features
        features = []
        
        # Check for brightness support
        if "Light" in device.modules and hasattr(device.modules["Light"], "brightness"):
            features.append("brightness")
            result["brightness"] = device.modules["Light"].brightness
        
        # Check for color support
        if "Light" in device.modules and hasattr(device.modules["Light"], "hsv"):
            features.append("color")
            hsv = device.modules["Light"].hsv
            if hsv:
                result["color"] = {
                    "hue": hsv.hue,
                    "saturation": hsv.saturation,
                    "value": hsv.value
                }
        
        # Check for color temperature support
        if "Light" in device.modules and hasattr(device.modules["Light"], "color_temp"):
            features.append("temperature")
            result["color_temp"] = device.modules["Light"].color_temp
        
        # Check for energy monitoring support
        if device.has_emeter:
            features.append("energy")
            if "Energy" in device.modules:
                result["power"] = device.modules["Energy"].current_consumption
        
        # Add features list to result
        result["features"] = features
        
        return result
    
    def _serialize_device_details(self, device: Any) -> Dict[str, Any]:
        """
        Serialize detailed device information for response.
        
        Args:
            device: The device object
            
        Returns:
            Dict containing detailed device information
        """
        # Start with summary
        result = self._serialize_device_summary(device)
        
        # Add additional information
        result["mac"] = device.mac
        result["rssi"] = device.rssi
        result["hardware_version"] = device.device_info.hardware_version
        result["firmware_version"] = device.device_info.firmware_version
        
        # Add time information
        if hasattr(device, "time"):
            result["device_time"] = device.time.isoformat()
        if device.on_since:
            result["on_since"] = device.on_since.isoformat()
        
        # Add modules list
        result["modules"] = list(device.modules.keys())
        
        # Add features detailed information
        result["features_details"] = {}
        for feature_id, feature in device.features.items():
            result["features_details"][feature_id] = {
                "name": feature.name,
                "value": str(feature.value)
            }
        
        # Add energy information if supported
        if device.has_emeter and "Energy" in device.modules:
            energy_module = device.modules["Energy"]
            result["energy"] = {
                "current_consumption": energy_module.current_consumption,
                "consumption_today": energy_module.consumption_today,
                "consumption_month": energy_module.consumption_this_month,
                "voltage": getattr(energy_module, "voltage", None),
                "current": getattr(energy_module, "current", None)
            }
        
        # Add child devices if any
        if hasattr(device, "children") and device.children:
            result["child_devices"] = []
            for child in device.children:
                result["child_devices"].append(self._serialize_device_summary(child))
        
        return result
    
    def _cache_device(self, device: Any) -> None:
        """
        Cache a device for future use.
        
        Args:
            device: The device object to cache
        """
        from config import config
        
        if not config.kasa_tool.cache_enabled:
            return
        
        try:
            # Generate cache key from device host
            cache_key = device.host
            
            # Prepare device data for caching
            device_data = {
                "host": device.host,
                "alias": device.alias,
                "model": device.model,
                "mac": device.mac,
                "device_type": str(device.device_type),
                "last_updated": utc_now().isoformat()
            }
            
            # Add to cache
            self.cache.set(cache_key, device_data)
            
            # Also cache by alias if available
            if device.alias:
                self.cache.set(device.alias, device_data)
                
            # Store in memory cache
            self._device_instances[device.host] = device
            if device.alias:
                self._device_instances[device.alias] = device
                
        except Exception as e:
            self.logger.warning(f"Error caching device {device.host}: {e}")
    
    async def _verify_change(
        self, 
        device: Any, 
        expected_value: Any, 
        attribute: str,
        max_attempts: int = 3,
        delay: float = 0.5
    ) -> bool:
        """
        Verify a change was successful by checking the device state.
        
        Args:
            device: The device or module object
            expected_value: The expected value
            attribute: The attribute to check
            max_attempts: Maximum number of verification attempts
            delay: Delay between verification attempts in seconds
            
        Returns:
            True if the change was successful, False otherwise
        """
        for attempt in range(max_attempts):
            try:
                # If device has an update method, use it
                if hasattr(device, "update"):
                    await device.update()
                # If device has a parent with an update method, use that
                elif hasattr(device, "_device") and hasattr(device._device, "update"):
                    await device._device.update()
                    
                # Check if the attribute has the expected value
                current_value = getattr(device, attribute, None)
                if current_value == expected_value:
                    return True
                    
                # If not, wait and retry
                await asyncio.sleep(delay)
            except Exception as e:
                self.logger.warning(f"Error verifying change: {e}")
                await asyncio.sleep(delay)
                
        # If we get here, verification failed
        self.logger.warning(
            f"Failed to verify change: expected {attribute}={expected_value}, "
            f"got {getattr(device, attribute, None)}"
        )
        return False