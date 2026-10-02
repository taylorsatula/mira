
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, create_model


class UnknownToolConfigFields(ValueError):
    """Tool-config validation received fields the tool's config class does not define.

    Carries the offending field names and the valid field set so every
    validation consumer rejects unknown keys with a precise error instead of
    silently dropping them (Pydantic's default extra='ignore').
    """

    def __init__(self, tool_name: str, unknown_fields: list[str], valid_fields: list[str]) -> None:
        self.tool_name = tool_name
        self.unknown_fields = unknown_fields
        self.valid_fields = valid_fields
        super().__init__(
            f"Unknown configuration field(s) for {tool_name}: "
            f"{', '.join(unknown_fields)}; valid fields: {', '.join(valid_fields)}"
        )

    def validation_entries(self) -> list[dict[str, str]]:
        """One entry per unknown field, in the /actions/tools validation_errors shape."""
        return [
            {
                "field": field,
                "message": (
                    f"Unknown configuration field '{field}' for {self.tool_name}; "
                    f"valid fields: {', '.join(self.valid_fields)}"
                ),
                "type": "unknown_field",
            }
            for field in self.unknown_fields
        ]

class ConfigRegistry:
    """Independent registry enabling drag-and-drop tool functionality without circular dependencies."""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, config_class: Type[BaseModel]) -> None:
        cls._registry[name] = config_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        return cls._registry.get(name)

    @classmethod
    def validate_config_data(cls, tool_name: str, config: Dict[str, Any]) -> BaseModel:
        """Instantiate a tool's config class from raw data, rejecting unknown fields.

        The single strictness seam for tool-config validation — every consumer
        that validates incoming tool-config data (the PUT and
        POST /actions/tools/{tool}/validate endpoints in cns/api/tool_config.py)
        calls this instead of instantiating the class directly. Pydantic's
        default extra='ignore' would silently drop a typo'd field name while
        the request still looks accepted, so unknown keys raise
        UnknownToolConfigFields naming the keys and the valid field set.

        Strictness deliberately lives here rather than extra='forbid' on the
        config classes themselves: the read path
        (config_manager.get_tool_config) instantiates from stored per-user
        rows, and rows written by the pre-strict endpoint can contain unknown
        keys — a blanket forbid would fail every read of such a row. Writes
        reject; the stored-row merge tolerates.
        """
        config_class = cls._registry.get(tool_name)
        if config_class is None:
            raise ValueError(f"No config class registered for '{tool_name}'")
        unknown_fields = sorted(key for key in config if key not in config_class.model_fields)
        if unknown_fields:
            raise UnknownToolConfigFields(
                tool_name, unknown_fields, sorted(config_class.model_fields)
            )
        return config_class(**config)
    
    @classmethod
    def create_default(cls, name: str) -> Type[BaseModel]:
        """Creates default config class with enabled=True for unregistered tools."""
        class_name = f"{name.capitalize()}Config"
        if name.endswith('_tool'):
            parts = name.split('_')
            class_name = ''.join(part.capitalize() for part in parts[:-1]) + 'ToolConfig'
        
        default_class = create_model(
            class_name,
            __base__=BaseModel,
            enabled=(bool, True),
            __doc__=f"Default configuration for {name}"
        )
        
        cls.register(name, default_class)
        
        return default_class
    
    @classmethod
    def get_or_create(cls, name: str) -> Type[BaseModel]:
        config_class = cls.get(name)
        if config_class is None:
            config_class = cls.create_default(name)
        return config_class
    
    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        return {name: config_class.__name__ for name, config_class in cls._registry.items()}

registry = ConfigRegistry()