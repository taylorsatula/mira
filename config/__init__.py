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
    
    # For tool configurations
    timeout = config.sample_tool.timeout
"""

# First, initialize the registry (which has no dependencies)
from config.registry import registry

# Then, import the configuration system
from config.config_manager import AppConfig, config

# Export the public interface
__all__ = ["config", "AppConfig", "registry"]