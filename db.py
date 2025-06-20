"""
Database module for tool data storage.

This module provides a unified interface for database operations using SQLAlchemy ORM.
It focuses on storing tool-specific data in PostgreSQL with multi-user support.
"""

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Index, Boolean
from sqlalchemy import text as sql_text  # Rename to avoid shadowing by column names
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from errors import ToolError, ErrorCode, error_context
from config import config

# Configure logger
logger = logging.getLogger(__name__)

# Create base model class
Base = declarative_base()

# Type variable for generic functions
T = TypeVar('T', bound=Base)



class Database:
    """Database with automatic user scoping - no more manual user_id filtering"""
    
    _instance = None
    _engine = None
    _session_factory = None
    
    def __new__(cls):
        """
        Singleton pattern to ensure only one database connection is created.
        
        Returns:
            Database instance
        """
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Initialize the database connection and session factory.
        
        Creates the database engine, tables, and session factory.
        """
        # Get database URI from config
        db_uri = config.database.uri
        
        # Validate PostgreSQL URI
        if not db_uri.startswith("postgresql://"):
            raise ValueError("Tool database requires PostgreSQL URI")
        
        # Create engine with echo for debugging if enabled
        self._engine = create_engine(
            db_uri,
            echo=config.database.echo,
            pool_size=config.database.pool_size,
            pool_timeout=config.database.pool_timeout,
            pool_recycle=config.database.pool_recycle
        )
        
        # Create tables
        Base.metadata.create_all(self._engine)
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
        
        logger.info(f"Database initialized with URI: {db_uri}")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy Session
            
        Note:
            The caller is responsible for closing the session.
            Use with a context manager for automatic cleanup.
        """
        return self._session_factory()
    
    def add(self, obj: Base) -> Base:
        """Add with automatic user_id assignment"""
        # Automatically set user_id for user-scoped models
        if hasattr(obj, 'user_id'):
            from config.tenant import tenant
            obj.user_id = tenant.user_id
        
        with error_context(
            component_name="database",
            operation="add",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            with self.get_session() as session:
                session.add(obj)
                session.commit()
                session.refresh(obj)
                return obj
    
    def get(self, model: Type[T], id: Any) -> Optional[T]:
        with self.get_session() as session:
            query = session.query(model).filter(model.id == id)
            
            # Add user filter for user-scoped models
            if hasattr(model, 'user_id'):
                from config.tenant import tenant
                query = query.filter(model.user_id == tenant.user_id)
            
            return query.first()
    
    def query(self, model: Type[T], *filters) -> List[T]:
        """Query with automatic user filtering"""
        with self.get_session() as session:
            query = session.query(model)
            
            # Automatically add user filter for user-scoped models
            if hasattr(model, 'user_id'):
                from config.tenant import tenant
                query = query.filter(model.user_id == tenant.user_id)
            
            if filters:
                query = query.filter(*filters)
            return query.all()
    
    def update(self, obj: Base) -> Base:
        """
        Update an existing object in the database.
        
        Args:
            obj: Object to update
            
        Returns:
            The updated object
            
        Raises:
            ToolError: If the object cannot be updated
        """
        with error_context(
            component_name="database",
            operation="update",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            with self.get_session() as session:
                # Use merge and return the merged instance which is attached to the session
                merged_obj = session.merge(obj)
                session.commit()
                # Refresh the merged object instead of the original
                session.refresh(merged_obj)
                return merged_obj
    
    def delete(self, obj: Base) -> bool:
        """
        Delete an object from the database.
        
        Args:
            obj: Object to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            ToolError: If the object cannot be deleted
        """
        with error_context(
            component_name="database",
            operation="delete",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            with self.get_session() as session:
                session.delete(obj)
                session.commit()
                return True
    
    def execute(self, statement: Union[str, sql_text], params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL statement.
        
        Args:
            statement: SQL statement (string or text object)
            params: Optional parameters for the statement
            
        Returns:
            Result of the execution
            
        Raises:
            ToolError: If the statement cannot be executed
        """
        with error_context(
            component_name="database",
            operation="execute",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            with self.get_session() as session:
                if isinstance(statement, str):
                    statement = sql_text(statement)
                    
                result = session.execute(statement, params or {})
                session.commit()
                return result


