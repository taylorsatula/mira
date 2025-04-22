"""
Database module for tool data storage.

This module provides a unified interface for database operations using SQLAlchemy ORM.
It focuses on storing tool-specific data in SQLite while maintaining compatibility
with the existing JSON-based storage for conversations and messages.
"""

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic

from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, inspect
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy.sql import text

from errors import ToolError, ErrorCode, error_context
from config import config

# Configure logger
logger = logging.getLogger(__name__)

# Create base model class
Base = declarative_base()

# Type variable for generic functions
T = TypeVar('T', bound=Base)


class Customer(Base):
    """
    Customer model for storing customer data.
    
    Maps to the 'customers' table with columns that match Square's customer structure
    plus additional fields for geocoding and metadata.
    """
    __tablename__ = 'customers'

    # Primary key
    id = Column(String, primary_key=True)
    
    # Basic contact info
    given_name = Column(String)
    family_name = Column(String)
    company_name = Column(String)
    email_address = Column(String)
    phone_number = Column(String)
    
    # Address fields
    address_line1 = Column(String)
    address_line2 = Column(String)
    city = Column(String)
    state = Column(String)
    postal_code = Column(String)
    country = Column(String)
    
    # Geocoding data
    latitude = Column(Float)
    longitude = Column(Float)
    geocoded_at = Column(DateTime)
    
    # Additional data stored as JSON
    additional_data = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary matching the original JSON structure.
        
        Returns:
            Dict representation of the customer
        """
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add non-null basic attributes
        if self.given_name:
            data["given_name"] = self.given_name
        if self.family_name:
            data["family_name"] = self.family_name
        if self.company_name:
            data["company_name"] = self.company_name
        if self.email_address:
            data["email_address"] = self.email_address
        if self.phone_number:
            data["phone_number"] = self.phone_number
            
        # Add address if any fields are present
        address_fields = {
            "address_line_1": self.address_line1,
            "address_line_2": self.address_line2,
            "locality": self.city,
            "administrative_district_level_1": self.state,
            "postal_code": self.postal_code, 
            "country": self.country
        }
        
        if any(address_fields.values()):
            data["address"] = {k: v for k, v in address_fields.items() if v}
            
        # Add geocoding data if available
        if self.latitude and self.longitude:
            data["geocoding_data"] = {
                "coordinates": {
                    "lat": self.latitude,
                    "lng": self.longitude
                }
            }
            if self.geocoded_at:
                data["geocoding_data"]["geocoded_at"] = int(self.geocoded_at.timestamp())
        
        # Add any additional data
        if self.additional_data:
            for key, value in self.additional_data.items():
                # Don't overwrite existing keys
                if key not in data:
                    data[key] = value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """
        Create a Customer instance from a dictionary.
        
        Args:
            data: Dictionary representation of a customer
            
        Returns:
            Customer instance
        """
        customer = cls(id=data["id"])
        
        # Basic attributes
        customer.given_name = data.get("given_name")
        customer.family_name = data.get("family_name")
        customer.company_name = data.get("company_name")
        customer.email_address = data.get("email_address")
        customer.phone_number = data.get("phone_number")
        
        # Address handling
        address = data.get("address", {})
        if address and isinstance(address, dict):
            customer.address_line1 = address.get("address_line_1")
            customer.address_line2 = address.get("address_line_2")
            customer.city = address.get("locality")
            customer.state = address.get("administrative_district_level_1")
            customer.postal_code = address.get("postal_code")
            customer.country = address.get("country")
        
        # Geocoding data
        geocoding_data = data.get("geocoding_data", {})
        if geocoding_data and isinstance(geocoding_data, dict):
            coordinates = geocoding_data.get("coordinates", {})
            if coordinates and isinstance(coordinates, dict):
                customer.latitude = coordinates.get("lat")
                customer.longitude = coordinates.get("lng")
            
            geocoded_at = geocoding_data.get("geocoded_at")
            if geocoded_at:
                if isinstance(geocoded_at, (int, float)):
                    customer.geocoded_at = datetime.fromtimestamp(geocoded_at)
        
        # Timestamps
        if "created_at" in data:
            if isinstance(data["created_at"], str):
                try:
                    customer.created_at = datetime.fromisoformat(data["created_at"])
                except ValueError:
                    customer.created_at = datetime.utcnow()
            elif isinstance(data["created_at"], (int, float)):
                customer.created_at = datetime.fromtimestamp(data["created_at"])
                
        if "updated_at" in data:
            if isinstance(data["updated_at"], str):
                try:
                    customer.updated_at = datetime.fromisoformat(data["updated_at"])
                except ValueError:
                    customer.updated_at = datetime.utcnow()
            elif isinstance(data["updated_at"], (int, float)):
                customer.updated_at = datetime.fromtimestamp(data["updated_at"])
        
        # Store all other fields in additional_data
        excluded_keys = {
            "id", "given_name", "family_name", "company_name", "email_address", 
            "phone_number", "address", "geocoding_data", "created_at", "updated_at"
        }
        
        additional_data = {}
        for key, value in data.items():
            if key not in excluded_keys:
                additional_data[key] = value
                
        if additional_data:
            customer.additional_data = additional_data
            
        return customer


class Database:
    """
    Database manager class providing a simplified interface for database operations.
    
    This class handles database connection management, session management,
    and provides a unified API for CRUD operations on database models.
    """
    
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
        
        # Ensure data directory exists
        db_path = db_uri.replace('sqlite:///', '')
        if db_path.startswith('/'):
            # Absolute path
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)
        
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
        """
        Add a new object to the database.
        
        Args:
            obj: Object to add
            
        Returns:
            The added object
            
        Raises:
            ToolError: If the object cannot be added
        """
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
        """
        Get an object by ID.
        
        Args:
            model: Model class
            id: Primary key value
            
        Returns:
            The object or None if not found
        """
        with self.get_session() as session:
            # Use session.get instead of query.get to avoid deprecation warning
            return session.get(model, id)
    
    def query(self, model: Type[T], *filters) -> List[T]:
        """
        Query objects with optional filters.
        
        Args:
            model: Model class
            *filters: SQLAlchemy filter conditions
            
        Returns:
            List of objects matching the filters
        """
        with self.get_session() as session:
            query = session.query(model)
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
    
    def execute(self, statement: Union[str, text], params: Optional[Dict[str, Any]] = None) -> Any:
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
                    statement = text(statement)
                    
                result = session.execute(statement, params or {})
                session.commit()
                return result


def migrate_customers_from_json() -> Dict[str, Any]:
    """
    Migrate customer data from JSON to the database.
    
    Returns:
        Status report with counts and summary
        
    Raises:
        ToolError: If migration fails
    """
    with error_context(
        component_name="database",
        operation="migrate_customers",
        error_class=ToolError,
        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
        logger=logger
    ):
        # Path to customer directory JSON
        data_dir = Path(config.paths.data_dir)
        cache_dir = data_dir / "tools" / "customer_tool"
        json_path = cache_dir / "customer_directory.json"
        
        if not json_path.exists():
            return {
                "success": False,
                "message": "No customer directory found to migrate",
                "customers_migrated": 0
            }
        
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                directory = json.load(f)
                
            # Check structure
            if not isinstance(directory, dict) or "customers" not in directory:
                return {
                    "success": False,
                    "message": "Invalid customer directory format",
                    "customers_migrated": 0
                }
                
            customers = directory.get("customers", {})
            
            # Create backup before migration
            backup_path = json_path.with_suffix('.json.bak')
            with open(backup_path, 'w') as f:
                json.dump(directory, f)
                
            logger.info(f"Created backup of customer directory at {backup_path}")
            
            # Create database instance
            db = Database()
            
            # Track migration status
            migrated_count = 0
            errors = []
            
            # Process each customer
            for customer_id, customer_data in customers.items():
                try:
                    # Skip if customer has no ID
                    if not customer_id or customer_id != customer_data.get("id"):
                        customer_data["id"] = customer_id
                        
                    # Create customer model from data
                    customer = Customer.from_dict(customer_data)
                    
                    # Check if customer already exists
                    existing = db.get(Customer, customer_id)
                    
                    if existing:
                        # Update existing customer
                        db.update(customer)
                    else:
                        # Add new customer
                        db.add(customer)
                        
                    migrated_count += 1
                except Exception as e:
                    errors.append(f"Error migrating customer {customer_id}: {str(e)}")
                    logger.error(f"Error migrating customer {customer_id}: {e}", exc_info=True)
            
            # Report results
            success = len(errors) == 0
            message = f"Migration complete. {migrated_count} customers migrated."
            if errors:
                message += f" {len(errors)} errors occurred."
                
            return {
                "success": success,
                "message": message,
                "customers_migrated": migrated_count,
                "errors": errors if errors else None
            }
                
        except Exception as e:
            logger.error(f"Error during customer migration: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Migration failed: {str(e)}",
                "customers_migrated": 0
            }