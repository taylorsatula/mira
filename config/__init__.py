"""
Centralized configuration management package.

This package provides a single configuration interface that loads settings
from various sources, validates them, and makes them available throughout
the application.

Usage:
    from config import config
    
    # Access using attribute notation
    model_name = config.api.model
    
    # Or using get() method with dot notation
    model_name = config.get("api.model")
    
    # For required values (raises exception if missing)
    api_key = config.require("api.key")
"""

from config.app_config import AppConfig, config

__all__ = ["config", "AppConfig"]