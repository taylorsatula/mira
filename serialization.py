"""
Serialization utility module for standardized JSON conversion.

This module provides consistent JSON serialization/deserialization
functionality across the codebase, handling common types and
supporting objects with custom to_dict() and from_dict() methods.
"""

import json
import datetime
import uuid
from typing import Any, Optional, Type, TypeVar, Dict, Union

T = TypeVar('T')


def to_json(obj: Any, indent: int = 2, **kwargs) -> str:
    """
    Serialize an object to JSON.

    Handles objects with to_dict() methods and common Python types like
    datetime.

    Args:
        obj: Object to serialize
        indent: Number of spaces for indentation (default: 2)
        **kwargs: Additional arguments to pass to json.dumps()

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, indent=indent, default=_json_serializer, **kwargs)


def from_json(
    json_str: str,
    cls: Optional[Type[T]] = None
) -> Union[T, Dict[str, Any]]:
    """
    Deserialize a JSON string to an object.

    If a class is provided and it has a from_dict() method, the method will
    be called with the parsed JSON data.

    Args:
        json_str: JSON string to deserialize
        cls: Optional class to deserialize to

    Returns:
        Instance of cls if provided and it has a from_dict() method,
        otherwise the parsed JSON data
    """
    data = json.loads(json_str)
    if cls and hasattr(cls, 'from_dict'):
        return cls.from_dict(data)
    return data


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for handling non-JSON-serializable objects.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object
    """
    # Handle objects with to_dict() method
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()

    # Handle datetime objects
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()

    # Handle date objects
    if isinstance(obj, datetime.date):
        return obj.isoformat()

    # Handle UUID objects
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # Raise TypeError for unhandled types
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")