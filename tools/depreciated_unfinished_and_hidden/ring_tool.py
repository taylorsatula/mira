"""
Ring Security System integration tool.

This tool enables integration with Ring security systems, allowing control of
alarm systems, checking device status, and listing locations.
"""
import json
import logging
import os
import time
import uuid
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import requests
from urllib.parse import urljoin

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config import config

# Constants
AUTH_URL = "https://oauth.ring.com/oauth/token"
API_VERSION = 11
CLIENT_API_BASE_URL = "https://api.ring.com/clients_api/"
DEVICE_API_BASE_URL = "https://api.ring.com/devices/v1/"
APP_API_BASE_URL = "https://app.ring.com/api/v1/"


def generate_hardware_id() -> str:
    """
    Generate a unique hardware ID for authentication.
    
    Returns:
        A unique hardware ID string
    """
    # Create a deterministic hardware ID based on machine info
    system_info = platform.uname()
    base = f"{system_info.system}-{system_info.node}-{system_info.machine}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


class RingApiClient:
    """
    Client for interacting with the Ring API.
    
    Handles authentication, token refresh, and API requests with retry logic.
    """
    
    def __init__(self, refresh_token: str = None, logger=None):
        """
        Initialize the Ring API client.
        
        Args:
            refresh_token: The OAuth refresh token for authentication
            logger: Logger instance for logging messages
        """
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expiration = None
        self.hardware_id = generate_hardware_id()
        self.logger = logger or logging.getLogger("ring_api")
    
    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the Ring API using refresh token.
        
        Returns:
            Authentication response data
            
        Raises:
            ToolError: If authentication fails
        """
        headers = {
            "hardware_id": self.hardware_id,
            "2fa-support": "true",
            "User-Agent": "android:com.ringapp",
        }
        
        data = {
            "client_id": "ring_official_android",
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "scope": "client",
        }
        
        try:
            response = requests.post(AUTH_URL, headers=headers, json=data)
            response.raise_for_status()
            auth_data = response.json()
            
            # Store authentication data
            self.access_token = auth_data["access_token"]
            
            # Calculate token expiration (subtract 60 seconds for safety)
            expires_in = auth_data.get("expires_in", 3600) - 60
            self.token_expiration = datetime.now() + timedelta(seconds=expires_in)
            
            # Return the auth data (but don't update the refresh token as it's persistent)
            return auth_data
            
        except requests.exceptions.RequestException as e:
            raise ToolError(
                f"Authentication failed: {str(e)}",
                ErrorCode.API_AUTHENTICATION_ERROR,
                {"error": str(e)}
            )
            
    def is_token_valid(self) -> bool:
        """
        Check if the current access token is valid.
        
        Returns:
            True if the token is valid, False otherwise
        """
        return (
            self.access_token is not None and
            self.token_expiration is not None and
            datetime.now() < self.token_expiration
        )
    
    def ensure_auth(self) -> None:
        """
        Ensure that authentication is valid before making API requests.
        
        Raises:
            ToolError: If authentication fails
        """
        if not self.is_token_valid():
            if not self.refresh_token:
                raise ToolError(
                    "No refresh token available for authentication",
                    ErrorCode.API_AUTHENTICATION_ERROR
                )
                
            self.authenticate()
    
    def request(self, method: str, url: str, **kwargs) -> Any:
        """
        Make an authenticated request to the Ring API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: API endpoint URL
            **kwargs: Additional request parameters
            
        Returns:
            Response data
            
        Raises:
            ToolError: If the request fails after retries
        """
        self.ensure_auth()
        
        # Prepare request
        headers = kwargs.pop("headers", {})
        headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "hardware_id": self.hardware_id,
            "User-Agent": "android:com.ringapp",
        })
        
        # Add content type for requests with body
        if "json" in kwargs:
            headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            
        # Prepare for retries
        max_retries = 3
        retry_count = 0
        backoff_factor = 2  # Exponential backoff
        
        while retry_count < max_retries:
            try:
                response = requests.request(
                    method, 
                    url, 
                    headers=headers, 
                    **kwargs
                )
                
                # Check for auth errors
                if response.status_code == 401:
                    # Token might be expired, refresh and retry
                    self.authenticate()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    retry_count += 1
                    continue
                    
                response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    return response.json()
                except ValueError:
                    # Not JSON, return text
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                retry_count += 1
                
                # Check if we should retry
                if retry_count >= max_retries:
                    raise ToolError(
                        f"Request failed after {max_retries} attempts: {str(e)}",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                        {"error": str(e), "url": url}
                    )
                
                # Exponential backoff
                wait_time = backoff_factor ** retry_count
                self.logger.warning(
                    f"Request failed, retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})"
                )
                time.sleep(wait_time)
    
    def get_locations(self) -> List[Dict[str, Any]]:
        """
        Get all available Ring locations.
        
        Returns:
            List of location data dictionaries
            
        Raises:
            ToolError: If the request fails
        """
        url = urljoin(DEVICE_API_BASE_URL, "locations")
        response = self.request("GET", url)
        
        if "user_locations" not in response:
            raise ToolError(
                "Failed to retrieve locations: Unexpected response format",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"response": response}
            )
            
        return response["user_locations"]
    
    def get_devices(self) -> Dict[str, Any]:
        """
        Get all devices from the Ring account.
        
        Returns:
            Dictionary containing device data
            
        Raises:
            ToolError: If the request fails
        """
        url = urljoin(CLIENT_API_BASE_URL, "ring_devices")
        return self.request("GET", url)
    
    def get_location_devices(self, location_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all devices for a specific location.
        
        Args:
            location_id: The ID of the location to get devices for
            
        Returns:
            Dictionary containing device data for the location
            
        Raises:
            ToolError: If the request fails
        """
        # First get all devices
        all_devices = self.get_devices()
        
        # Then filter by location
        location_devices = {
            "doorbots": [],
            "chimes": [],
            "stickup_cams": [],
            "base_stations": [],
            "beams_bridges": [],
            "other": []
        }
        
        # Extract device categories from response
        for category in location_devices:
            if category in all_devices:
                location_devices[category] = [
                    device for device in all_devices[category]
                    if device.get("location_id") == location_id
                ]
                
        return location_devices
    
    def get_security_panel_device(self, location_id: str) -> Dict[str, Any]:
        """
        Get the security panel device for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Security panel device data
            
        Raises:
            ToolError: If no security panel is found
        """
        # Get all devices for the location
        devices = self.get_location_devices(location_id)
        
        # Look for security panel in base stations
        for device in devices.get("base_stations", []):
            if device.get("kind") == "security-panel":
                return device
        
        # If no security panel found, raise error
        raise ToolError(
            f"Could not find a security panel for location {location_id}",
            ErrorCode.TOOL_EXECUTION_ERROR,
            {"location_id": location_id}
        )
    
    def get_alarm_mode(self, location_id: str) -> Dict[str, Any]:
        """
        Get the current alarm mode for a location.
        
        Args:
            location_id: The ID of the location
            
        Returns:
            Alarm mode data including the current mode
            
        Raises:
            ToolError: If the request fails or no security panel is found
        """
        try:
            # Get the security panel device for this location
            security_panel = self.get_security_panel_device(location_id)
            
            # Log the entire security panel data for debugging
            self.logger.info(f"Security panel data: {json.dumps(security_panel, indent=2)}")
            
            # Extract mode from device data
            mode = security_panel.get("mode", "unknown")
            self.logger.info(f"Extracted mode from security panel: {mode}")
            
            # Return structured response
            return {"mode": mode}
            
        except Exception as e:
            self.logger.warning(f"Failed to get mode from security panel: {str(e)}")
            # Fallback to location mode endpoint as a last resort
            try:
                url = urljoin(APP_API_BASE_URL, f"mode/location/{location_id}")
                self.logger.info(f"Trying fallback location mode API: {url}")
                response = self.request("GET", url)
                self.logger.info(f"Location mode API response: {json.dumps(response, indent=2)}")
                return response
            except Exception as fallback_error:
                self.logger.error(f"Fallback location mode API also failed: {str(fallback_error)}")
                # If both methods fail, raise the original error
                raise e
    
    def set_alarm_mode(self, location_id: str, mode: str) -> Dict[str, Any]:
        """
        Set the alarm mode for a location.
        
        Args:
            location_id: The ID of the location
            mode: The mode to set (disarmed, home, away)
            
        Returns:
            Response data
            
        Raises:
            ToolError: If the request fails or the mode is invalid
        """
        # Map mode names to Ring API values
        mode_map = {
            "disarmed": "none",
            "home": "some",
            "away": "all"
        }
        
        if mode not in mode_map:
            raise ToolError(
                f"Invalid alarm mode: {mode}. Must be one of: disarmed, home, away",
                ErrorCode.TOOL_INVALID_INPUT,
                {"valid_modes": list(mode_map.keys())}
            )
            
        ring_mode = mode_map[mode]
        
        try:
            # Approach 1: Get the security panel device to send the command directly
            # This matches TypeScript implementation in location.ts:setAlarmMode
            security_panel = self.get_security_panel_device(location_id)
            device_id = security_panel.get("id")
            
            if device_id:
                # Send command to the security panel
                # Based on sendCommandToSecurityPanel and sendCommand methods in TypeScript
                url = urljoin(CLIENT_API_BASE_URL, f"devices/{device_id}")
                
                # This payload structure matches the TypeScript implementation
                # where it uses security-panel.switch-mode command
                data = {
                    "command": {
                        "v1": [
                            {
                                "commandType": "security-panel.switch-mode",
                                "data": {
                                    "mode": ring_mode,
                                    "bypass": []
                                }
                            }
                        ]
                    }
                }
                
                self.logger.info(f"Setting mode to {ring_mode} via security panel device API")
                return self.request("PUT", url, json=data)
        except Exception as e:
            self.logger.warning(f"Failed to set mode via security panel: {str(e)}")
            
        # Approach 2: Use the location mode API (matches TypeScript setLocationMode)
        try:
            # First get the current mode to see the response format
            url = urljoin(APP_API_BASE_URL, f"mode/location/{location_id}")
            current_mode_response = self.request("GET", url)
            self.logger.info(f"Current mode response: {json.dumps(current_mode_response, indent=2)}")
            
            # Extract securityStatus to include it in our request
            security_status = current_mode_response.get("securityStatus", {})
            timestamp = int(time.time() * 1000)  # Current time in milliseconds
            
            # Prepare data with similar format to what the API expects
            data = {
                "mode": ring_mode,
                "lastUpdateTimeMS": timestamp
            }
            
            # If there's security status info, include it
            if security_status:
                security_status["md"] = ring_mode  # Update mode in security status
                security_status["lu"] = timestamp  # Update timestamp
                data["securityStatus"] = security_status
            
            self.logger.info(f"Setting mode to {ring_mode} via location mode API with payload: {json.dumps(data, indent=2)}")
            return self.request("POST", url, json=data)
        except Exception as e:
            self.logger.warning(f"Failed to set mode via location mode API: {str(e)}")
            
            # Try an alternative approach with simpler payload
            try:
                url = urljoin(APP_API_BASE_URL, f"mode/location/{location_id}")
                data = {"mode": ring_mode}
                self.logger.info(f"Trying simplified payload: {json.dumps(data, indent=2)}")
                return self.request("POST", url, json=data)
            except Exception as e2:
                self.logger.warning(f"Failed with simplified payload: {str(e2)}")
                
                # Try one more API format
                try:
                    url = urljoin(APP_API_BASE_URL, f"locations/{location_id}/mode")
                    data = {"mode": ring_mode}
                    self.logger.info(f"Trying locations mode API: {json.dumps(data, indent=2)}")
                    return self.request("POST", url, json=data)
                except Exception as e3:
                    self.logger.warning(f"Failed with locations mode API: {str(e3)}")
                    
                    raise ToolError(
                        f"Failed to set alarm mode to {mode}. All approaches failed.",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                        {"location_id": location_id, "requested_mode": mode, "error": str(e)}
                    )
    
    def verify_alarm_mode(self, location_id: str, expected_mode: str, max_attempts: int = 3) -> bool:
        """
        Verify that the alarm mode has been set correctly.
        
        Args:
            location_id: The ID of the location
            expected_mode: The expected mode (disarmed, home, away)
            max_attempts: Maximum number of verification attempts
            
        Returns:
            True if the mode matches, False otherwise
        """
        # Map mode names to Ring API values
        mode_map = {
            "disarmed": "none",
            "home": "some",
            "away": "all"
        }
        
        ring_mode = mode_map.get(expected_mode, "")
        self.logger.info(f"Verifying alarm mode - expecting: {expected_mode} (API value: {ring_mode})")
        
        # Try multiple times with a delay between attempts
        # This accounts for any delay in the mode change propagating
        for attempt in range(max_attempts):
            try:
                # Get current mode
                self.logger.info(f"Verification attempt {attempt+1}/{max_attempts}")
                current_mode_data = self.get_alarm_mode(location_id)
                self.logger.info(f"Current mode data: {json.dumps(current_mode_data, indent=2)}")
                
                # Extract current mode from response - try both direct key and nested structure
                current_mode = ""
                if isinstance(current_mode_data, dict):
                    current_mode = current_mode_data.get("mode", "")
                
                self.logger.info(f"Extracted current mode: {current_mode}, Expected: {ring_mode}")
                
                # Check if mode matches expected value (case insensitive)
                # Also try matching the friendly name directly
                if (current_mode.lower() == ring_mode.lower() or 
                    current_mode.lower() == expected_mode.lower()):
                    self.logger.info(f"Mode verified successfully as {current_mode}")
                    return True
                
                self.logger.warning(f"Mode verification failed. Got: {current_mode}, Expected: {ring_mode}")
                    
                # If not on the last attempt, wait before trying again
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)  # Exponential backoff
                    
            except Exception as e:
                self.logger.warning(f"Error verifying alarm mode (attempt {attempt+1}): {str(e)}")
                
                # If not on the last attempt, wait before trying again
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Error occurred. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        self.logger.error(f"Failed to verify alarm mode after {max_attempts} attempts")
        return False


class RingAuthenticator:
    """
    Handles Ring authentication and token management.
    """
    
    def __init__(self, data_dir: str, logger=None):
        """
        Initialize the authenticator.
        
        Args:
            data_dir: Directory to store authentication data
            logger: Logger instance for logging messages
        """
        self.data_dir = data_dir
        self.token_path = os.path.join(data_dir, "token.json")
        self.logger = logger or logging.getLogger("ring_auth")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_token(self) -> Optional[str]:
        """
        Load the refresh token from the token file or environment.
        
        Returns:
            Refresh token if available, None otherwise
        """
        # Try loading from environment first
        refresh_token = os.environ.get("RING_API_TOKEN")
        if refresh_token:
            return refresh_token
            
        # Then try the token file
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "r") as f:
                    token_data = json.load(f)
                    return token_data.get("refresh_token")
            except (json.JSONDecodeError, IOError):
                return None
                
        return None
    
    def save_token(self, token_data: Dict[str, Any]) -> None:
        """
        Save token data to the token file.
        
        Args:
            token_data: Token data to save
        """
        with open(self.token_path, "w") as f:
            json.dump(token_data, f, indent=2)
            
    def authenticate_interactive(self) -> str:
        """
        Perform interactive authentication to get a refresh token.
        
        Returns:
            New refresh token
            
        Raises:
            ToolError: If authentication fails
        """
        print("\n*** Ring Interactive Authentication ***\n")
        print("This process will generate a refresh token for the Ring API.")
        print("Please follow the steps carefully:\n")
        
        print("1. Go to https://oauth.ring.com/oauth/authorize?client_id=ring_official_android&response_type=code&scope=client")
        print("2. Log in with your Ring credentials")
        print("3. After logging in, you'll be redirected to a URL that starts with 'com.ringapp://'")
        print("4. Copy the entire URL and paste it below\n")
        
        auth_url = input("Paste the URL here: ").strip()
        
        # Extract the code from the URL
        match = re.search(r"code=([^&]+)", auth_url)
        if not match:
            raise ToolError(
                "Invalid authentication URL: Could not find authorization code",
                ErrorCode.API_AUTHENTICATION_ERROR
            )
            
        auth_code = match.group(1)
        
        # Exchange code for token
        headers = {
            "hardware_id": generate_hardware_id(),
            "User-Agent": "android:com.ringapp",
        }
        
        data = {
            "client_id": "ring_official_android",
            "grant_type": "authorization_code",
            "code": auth_code,
        }
        
        try:
            response = requests.post(AUTH_URL, headers=headers, json=data)
            response.raise_for_status()
            auth_data = response.json()
            
            # Save the token data
            self.save_token(auth_data)
            
            # Output the token for the user to save to .env
            print("\nAuthentication successful!")
            print(f"\nYour refresh token is: {auth_data['refresh_token']}")
            print("\nSave this to your .env file as RING_API_TOKEN=<token>")
            
            return auth_data["refresh_token"]
            
        except requests.exceptions.RequestException as e:
            raise ToolError(
                f"Authentication failed: {str(e)}",
                ErrorCode.API_AUTHENTICATION_ERROR,
                {"error": str(e)}
            )


class RingTool(Tool):
    """
    Tool for interacting with Ring security devices and systems.
    
    This tool provides functionality to:
    1. List Ring locations and devices
    2. Check alarm status
    3. Arm and disarm security systems
    """
    
    name = "ring_tool"
    description = """Controls Ring security devices with comprehensive management capabilities.

This tool enables interaction with Ring security systems including:

1. Device and Location Management:
   - 'list_locations': Retrieve all available Ring locations
   - 'list_devices': Retrieve all devices for a specific location
   - 'get_device_info': Get detailed status and capabilities of a specific device

2. Alarm System Control:
   - 'get_alarm_mode': Check current mode of alarm system (disarmed, home, away)
   - 'set_alarm_mode': Set alarm mode (disarmed, home, away)

Use this tool whenever you need to check the status of or control Ring security devices.
The tool requires authentication with a Ring account through a refresh token.
"""
    usage_examples = [
        {
            "input": {
                "operation": "list_locations"
            },
            "output": {
                "locations": [
                    {
                        "id": "488e4800-fcde-4493-969b-d1a06f683102",
                        "name": "Home",
                        "timezone": "America/New_York"
                    }
                ]
            }
        },
        {
            "input": {
                "operation": "list_devices",
                "location_id": "488e4800-fcde-4493-969b-d1a06f683102"
            },
            "output": {
                "devices": [
                    {
                        "id": "5ab22d5c-f456-4d19-8f55-3491face68d9",
                        "name": "Front Door",
                        "type": "doorbell"
                    }
                ]
            }
        },
        {
            "input": {
                "operation": "set_alarm_mode",
                "location_id": "488e4800-fcde-4493-969b-d1a06f683102",
                "mode": "home"
            },
            "output": {
                "success": True,
                "message": "Alarm system armed in home mode",
                "location": "Home"
            }
        }
    ]
    
    def __init__(self):
        """Initialize the Ring tool."""
        super().__init__()
        
        # Set up data directory
        self.data_dir = os.path.join("data", "tools", "ring_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize authentication and API client
        self.authenticator = RingAuthenticator(self.data_dir, self.logger)
        self.refresh_token = self.authenticator.load_token()
        self.api_client = None
    
    def _get_api_client(self) -> RingApiClient:
        """
        Get an initialized API client.
        
        Returns:
            RingApiClient instance
            
        Raises:
            ToolError: If authentication fails
        """
        if self.api_client is None:
            if not self.refresh_token:
                # Attempt interactive authentication
                self.refresh_token = self.authenticator.authenticate_interactive()
                
            self.api_client = RingApiClient(self.refresh_token, self.logger)
            
        return self.api_client
    
    def _list_locations(self) -> Dict[str, Any]:
        """
        List all available Ring locations.
        
        Returns:
            Dictionary containing locations data
        """
        client = self._get_api_client()
        locations = client.get_locations()
        
        # Format location data for output
        formatted_locations = []
        for location in locations:
            formatted_locations.append({
                "id": location["location_id"],
                "name": location["name"],
                "address": location.get("address", {}).get("address1", ""),
                "timezone": location.get("timezone", "")
            })
            
        return {"locations": formatted_locations}
    
    def _list_devices(self, location_id: str) -> Dict[str, Any]:
        """
        List all devices for a specific location.
        
        Args:
            location_id: The ID of the location to get devices for
            
        Returns:
            Dictionary containing devices data
        """
        client = self._get_api_client()
        devices_by_type = client.get_location_devices(location_id)
        
        # Combine all device types
        all_devices = []
        
        # Process doorbots (doorbells)
        for device in devices_by_type.get("doorbots", []):
            all_devices.append({
                "id": str(device["id"]),
                "name": device.get("description", "Doorbell"),
                "type": "doorbell",
                "battery_level": device.get("battery_life"),
                "firmware": device.get("firmware_version")
            })
            
        # Process cameras
        for device in devices_by_type.get("stickup_cams", []):
            all_devices.append({
                "id": str(device["id"]),
                "name": device.get("description", "Camera"),
                "type": "camera",
                "battery_level": device.get("battery_life"),
                "firmware": device.get("firmware_version")
            })
            
        # Process base stations and alarm devices
        for device in devices_by_type.get("base_stations", []):
            device_type = "alarm_base_station"
            if device.get("kind") == "security-panel":
                device_type = "security_panel"
                
            all_devices.append({
                "id": str(device["id"]),
                "name": device.get("description", "Alarm Device"),
                "type": device_type,
                "firmware": device.get("firmware_version")
            })
            
        return {"devices": all_devices}
    
    def _get_device_info(self, location_id: str, device_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific device.
        
        Args:
            location_id: The ID of the location the device belongs to
            device_id: The ID of the device to get info for
            
        Returns:
            Dictionary containing device information
            
        Raises:
            ToolError: If the device is not found
        """
        client = self._get_api_client()
        
        # Get all devices for the location
        devices_by_type = client.get_location_devices(location_id)
        
        # Search for the device in all categories
        for category, devices in devices_by_type.items():
            for device in devices:
                if str(device.get("id")) == device_id:
                    return device
                    
        raise ToolError(
            f"Device not found with ID {device_id} in location {location_id}",
            ErrorCode.TOOL_EXECUTION_ERROR,
            {"location_id": location_id, "device_id": device_id}
        )
    
    def _get_alarm_mode(self, location_id: str) -> Dict[str, Any]:
        """
        Get the current mode of the alarm system.
        
        Args:
            location_id: The ID of the location to get alarm mode for
            
        Returns:
            Dictionary containing alarm mode information
        """
        client = self._get_api_client()
        
        # Get the alarm mode
        self.logger.info(f"Requesting alarm mode for location {location_id}")
        mode_data = client.get_alarm_mode(location_id)
        self.logger.info(f"Raw alarm mode data: {json.dumps(mode_data, indent=2)}")
        
        # Get location information
        locations = client.get_locations()
        location_name = "Unknown"
        
        for location in locations:
            if location["location_id"] == location_id:
                location_name = location["name"]
                break
        
        # Map API mode to user-friendly mode
        mode_map = {
            "none": "disarmed",
            "some": "home",
            "all": "away"
        }
        
        # Check different possible mode formats from the response
        current_mode = "unknown"
        if isinstance(mode_data, dict):
            # Try different possible keys where mode might be stored
            if "mode" in mode_data:
                current_mode = mode_data["mode"]
            elif "state" in mode_data:
                current_mode = mode_data["state"]
            
        self.logger.info(f"Extracted current_mode: {current_mode}")
        friendly_mode = mode_map.get(current_mode, current_mode)
        self.logger.info(f"Mapped to friendly_mode: {friendly_mode}")
        
        response = {
            "location_name": location_name,
            "mode": friendly_mode,
            "raw_mode": current_mode
        }
        
        self.logger.info(f"Final alarm mode response: {json.dumps(response, indent=2)}")
        return response
    
    def _set_alarm_mode(self, location_id: str, mode: str) -> Dict[str, Any]:
        """
        Set the mode of the alarm system.
        
        Args:
            location_id: The ID of the location to set alarm mode for
            mode: The mode to set (disarmed, home, away)
            
        Returns:
            Dictionary containing operation result
        """
        client = self._get_api_client()
        
        # Get location information
        locations = client.get_locations()
        location_name = "Unknown"
        
        for location in locations:
            if location["location_id"] == location_id:
                location_name = location["name"]
                break
        
        # Get current mode before changing
        current_mode_data = client.get_alarm_mode(location_id)
        mode_map = {
            "none": "disarmed",
            "some": "home",
            "all": "away"
        }
        previous_mode = mode_map.get(current_mode_data.get("mode", "unknown"), "unknown")
        
        # Set the alarm mode
        client.set_alarm_mode(location_id, mode)
        
        # Verify the mode was set correctly
        success = client.verify_alarm_mode(location_id, mode)
        
        if not success:
            raise ToolError(
                f"Failed to set alarm mode to {mode}. The mode was not changed.",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"location_id": location_id, "requested_mode": mode}
            )
        
        # Mode descriptions for messaging
        mode_descriptions = {
            "disarmed": "disarmed",
            "home": "armed in home mode",
            "away": "armed in away mode"
        }
        
        return {
            "success": True,
            "message": f"Alarm system {mode_descriptions[mode]}",
            "location": location_name,
            "previous_mode": previous_mode,
            "current_mode": mode
        }
    
    def run(
        self,
        operation: str,
        location_id: Optional[str] = None,
        device_id: Optional[str] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a Ring device operation.
        
        Args:
            operation: Operation to perform (list_locations, list_devices, etc.)
            location_id: The ID of the location to operate on
            device_id: The ID of the device to operate on
            mode: Mode to set (for set_alarm_mode operation)
            
        Returns:
            Dictionary with operation result
            
        Raises:
            ToolError: If operation fails or parameters are invalid
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Run the appropriate operation
            if operation == "list_locations":
                return self._list_locations()
                
            elif operation == "list_devices":
                if not location_id:
                    raise ToolError(
                        "location_id parameter is required for list_devices operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                return self._list_devices(location_id)
                
            elif operation == "get_device_info":
                if not location_id or not device_id:
                    raise ToolError(
                        "location_id and device_id parameters are required for get_device_info operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                return self._get_device_info(location_id, device_id)
                
            elif operation == "get_alarm_mode":
                if not location_id:
                    raise ToolError(
                        "location_id parameter is required for get_alarm_mode operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                return self._get_alarm_mode(location_id)
                
            elif operation == "set_alarm_mode":
                if not location_id:
                    raise ToolError(
                        "location_id parameter is required for set_alarm_mode operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                if not mode:
                    raise ToolError(
                        "mode parameter is required for set_alarm_mode operation",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                if mode not in ["disarmed", "home", "away"]:
                    raise ToolError(
                        f"Invalid alarm mode: {mode}. Must be one of: disarmed, home, away",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                return self._set_alarm_mode(location_id, mode)
                
            else:
                valid_operations = [
                    "list_locations", 
                    "list_devices", 
                    "get_device_info",
                    "get_alarm_mode", 
                    "set_alarm_mode"
                ]
                
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: {', '.join(valid_operations)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )

# #BOOKMARK: Future enhancement - WebSocket connections and push notifications
# These features would need to be implemented for real-time status updates and notifications,
# but are not necessary for the basic security system functionality implemented here.