"""
Configuration registry module.

This module provides a registry for configuration classes that is independent
of both the config system and the tools. It serves as a bridge between them,
enabling drag-and-drop tool functionality without circular dependencies.

The registry is initialized first, before any imports of config or tools,
and then both systems can interact with it independently.
"""

import logging
from typing import Dict, Type, Any, Optional
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

class ConfigRegistry:
    """
    Registry for configuration classes.
    
    This registry is independent of both the config system and the tools.
    It serves as a bridge between them, allowing tools to register their
    configuration classes without importing the config system, and allowing
    the config system to access those classes without importing tools.
    """
    
    # Class variable to store registered configurations
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, config_class: Type[BaseModel]) -> None:
        """
        Register a configuration class.
        
        Args:
            name: The name to register the configuration under
            config_class: The configuration class
        """
        logger.debug(f"Registering config class for: {name}")
        cls._registry[name] = config_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """
        Get a registered configuration class.
        
        Args:
            name: The name of the configuration to get
            
        Returns:
            The configuration class, or None if not found
        """
        return cls._registry.get(name)
    
    @classmethod
    def create_default(cls, name: str) -> Type[BaseModel]:
        """
        Create a default configuration class.
        
        Args:
            name: The name to create a default configuration for
            
        Returns:
            A new configuration class with default settings
        """
        logger.debug(f"Creating default config class for: {name}")
        
        # Generate appropriate class name
        class_name = f"{name.capitalize()}Config"
        if name.endswith('_tool'):
            # Convert snake_case to CamelCase
            parts = name.split('_')
            class_name = ''.join(part.capitalize() for part in parts[:-1]) + 'ToolConfig'
        
        # Create the model
        default_class = create_model(
            class_name,
            __base__=BaseModel,
            enabled=(bool, True),
            __doc__=f"Default configuration for {name}"
        )
        
        # Register the new class
        cls.register(name, default_class)
        
        return default_class
    
    @classmethod
    def get_or_create(cls, name: str) -> Type[BaseModel]:
        """
        Get a configuration class, creating a default if not found.
        
        Args:
            name: The name of the configuration to get
            
        Returns:
            The configuration class (existing or newly created)
        """
        config_class = cls.get(name)
        if config_class is None:
            config_class = cls.create_default(name)
        return config_class
    
    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """
        List all registered configurations.
        
        Returns:
            A dictionary mapping names to configuration class names
        """
        return {name: config_class.__name__ for name, config_class in cls._registry.items()}

# Global singleton instance
registry = ConfigRegistry()