"""
Database-specific datetime utilities for consistent UTC handling.

This module provides utilities for working with datetime fields in database models,
ensuring consistent UTC storage and timezone handling. It builds on top of
timezone_utils.py to provide database-specific functionality.

Core principles:
- All datetime fields are stored as UTC in the database
- Timezone information is not stored in the database (it's handled at application level)
- All datetime objects are converted to UTC before storage
- All conversions from UTC to local time happen at display time, not storage time
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, Optional, Type, TypeVar, Union, Callable

from sqlalchemy import Column, DateTime
from sqlalchemy.ext.declarative import DeclarativeMeta

from utils.timezone_utils import (
    ensure_utc, 
    convert_to_utc, 
    convert_from_utc, 
    utc_now, 
    format_utc_for_storage,
    format_datetime,
    parse_utc_time_string
)

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


def utc_datetime_column(
    nullable: bool = True, 
    default: Optional[Union[datetime, Callable[[], datetime]]] = None,
    onupdate: Optional[Union[datetime, Callable[[], datetime]]] = None
) -> Column:
    """
    Create a DateTime column that stores values in UTC.
    
    This is a helper function to create standardized UTC datetime columns.
    
    Args:
        nullable: Whether the column can be NULL
        default: Default value or function for INSERT
        onupdate: Default value or function for UPDATE
        
    Returns:
        SQLAlchemy Column configured for UTC datetime storage
    """
    # For default and onupdate, use utc_now if they're True
    if default is True:
        default = utc_now
    if onupdate is True:
        onupdate = utc_now
        
    return Column(DateTime, nullable=nullable, default=default, onupdate=onupdate)


def utc_created_at_column() -> Column:
    """
    Create a standardized created_at timestamp column in UTC.
    
    Returns:
        SQLAlchemy Column for created_at timestamp
    """
    return utc_datetime_column(nullable=False, default=utc_now)


def utc_updated_at_column() -> Column:
    """
    Create a standardized updated_at timestamp column in UTC.
    
    Returns:
        SQLAlchemy Column for updated_at timestamp
    """
    return utc_datetime_column(nullable=False, default=utc_now, onupdate=utc_now)


def convert_datetime_to_utc_for_storage(value: Any) -> Any:
    """
    Convert a datetime value to UTC for database storage.
    
    Args:
        value: Value to convert (only datetime objects are affected)
        
    Returns:
        UTC datetime if input is a datetime, otherwise unchanged value
    """
    if isinstance(value, datetime):
        return ensure_utc(value)
    return value


def format_db_datetime_as_iso(dt: Optional[datetime], target_tz: Optional[str] = None) -> Optional[str]:
    """
    Format a database datetime as ISO string in the specified timezone.
    
    Args:
        dt: Database datetime (assumed to be in UTC)
        target_tz: Target timezone for display
        
    Returns:
        ISO formatted string in target timezone or None if dt is None
    """
    if dt is None:
        return None
        
    # Ensure datetime has timezone info (assumed to be UTC)
    dt = ensure_utc(dt)
    
    # If target timezone is specified, convert to it
    if target_tz:
        dt = convert_from_utc(dt, target_tz)
        
    return dt.isoformat()


def format_db_datetime(
    dt: Optional[datetime], 
    format_type: str = "date_time", 
    target_tz: Optional[str] = None
) -> Optional[str]:
    """
    Format a database datetime in the specified format and timezone.
    
    Args:
        dt: Database datetime (assumed to be in UTC)
        format_type: Format type as defined in timezone_utils.TIME_FORMATS
        target_tz: Target timezone for display
        
    Returns:
        Formatted string in target timezone or None if dt is None
    """
    if dt is None:
        return None
        
    # Ensure datetime has UTC timezone info
    dt = ensure_utc(dt)
    
    # Format using timezone_utils
    return format_datetime(dt, format_type, target_tz)


def serialize_model_datetime(
    model_dict: Dict[str, Any], 
    datetime_fields: list[str],
    target_tz: Optional[str] = None
) -> Dict[str, Any]:
    """
    Serialize datetime fields in a model dictionary to ISO strings.
    
    Args:
        model_dict: Dictionary representation of a model
        datetime_fields: List of field names that contain datetime values
        target_tz: Target timezone for display
        
    Returns:
        Dictionary with datetime fields converted to strings
    """
    result = model_dict.copy()
    
    for field in datetime_fields:
        if field in result and isinstance(result[field], datetime):
            result[field] = format_db_datetime_as_iso(result[field], target_tz)
            
    return result


def deserialize_datetime_strings(
    data: Dict[str, Any], 
    datetime_fields: list[str]
) -> Dict[str, Any]:
    """
    Deserialize datetime string fields in a dictionary to UTC datetime objects.
    
    Args:
        data: Dictionary containing string datetime fields
        datetime_fields: List of field names that should be converted to datetime
        
    Returns:
        Dictionary with string fields converted to UTC datetime objects
    """
    result = data.copy()
    
    for field in datetime_fields:
        if field in result and isinstance(result[field], str):
            try:
                result[field] = parse_utc_time_string(result[field])
            except Exception as e:
                logger.warning(f"Failed to parse datetime string '{result[field]}' for field '{field}': {e}")
                
    return result


class UTCDatetimeMixin:
    """
    Mixin to add automatic UTC conversion for datetime attributes.
    
    This mixin adds a SQLAlchemy event listener to convert datetime attributes 
    to UTC before they are persisted to the database.
    """
    
    created_at = utc_created_at_column()
    updated_at = utc_updated_at_column()
    
    def __init__(self, **kwargs):
        # Convert any datetime values to UTC before initialization
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                kwargs[key] = ensure_utc(value)
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary with ISO-formatted datetime strings.
        
        Returns:
            Dict with datetime fields formatted as ISO strings
        """
        # Get model attributes
        mapper = getattr(self.__class__, '__mapper__', None)
        if mapper is None:
            return {}
            
        # Create dict with all column values
        result = {}
        for column in mapper.columns:
            key = column.key
            value = getattr(self, key)
            
            # Format datetime values
            if isinstance(value, datetime):
                value = format_db_datetime_as_iso(value)
            elif isinstance(value, date):
                value = value.isoformat()
                
            result[key] = value
            
        return result
        
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a model instance from a dictionary, converting datetime strings to UTC.
        
        Args:
            data: Dictionary with model data
            
        Returns:
            Model instance
        """
        # Find datetime fields from model
        mapper = getattr(cls, '__mapper__', None)
        if mapper is None:
            return cls(**data)
            
        datetime_fields = []
        for column in mapper.columns:
            if isinstance(column.type, DateTime):
                datetime_fields.append(column.key)
                
        # Convert datetime strings to UTC datetime objects
        processed_data = {}
        for key, value in data.items():
            if key in datetime_fields and isinstance(value, str):
                try:
                    processed_data[key] = parse_utc_time_string(value)
                except Exception as e:
                    logger.warning(f"Failed to parse datetime string '{value}' for field '{key}': {e}")
                    processed_data[key] = value
            else:
                processed_data[key] = value
                
        return cls(**processed_data)