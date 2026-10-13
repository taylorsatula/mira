"""
Centralized configuration management package.

This package provides a single configuration interface that loads settings
from various sources, validates them, and makes them available throughout
the application.

Usage:
    from config import config
    
    # Access using attribute notation
    timeout = config.api.provider_response_timeout

    # Or using get() method with dot notation
    timeout = config.get("api.provider_response_timeout")
    
    # For required values (raises exception if missing)
    api_key = config.require("api.key")
    
    # For tool configurations
    enabled = config.web_tool.enabled
"""

from tools.registry import registry

from config.config_manager import AppConfig, config

__all__ = ["config", "AppConfig", "registry"]
