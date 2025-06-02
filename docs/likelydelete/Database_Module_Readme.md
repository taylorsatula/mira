# Database Module Documentation

## Overview

The database module provides SQLite support for the Bot With Memory project. It primarily serves as a storage solution for tool-specific data, while preserving the existing JSON-based conversation storage.

## Key Components

### Database Class

The `Database` class is a singleton that provides a centralized interface for database operations. It handles connection management, session creation, and provides a clean API for CRUD operations.

```python
from db import Database

# Get a database instance
db = Database()

# Perform operations
customer = db.get(Customer, "customer_id")
```

### Base Models

All database models inherit from the SQLAlchemy `Base` class:

```python
from db import Base, Column, String

class MyModel(Base):
    __tablename__ = 'my_models'
    id = Column(String, primary_key=True)
    name = Column(String)
```

### Customer Model

The module provides a `Customer` model that maps to the customers table, with full compatibility with the existing JSON structure. It includes:

- Basic contact fields (name, email, phone)
- Address fields
- Geocoding support (latitude/longitude)
- Additional data storage as JSON

## Usage

### Basic CRUD Operations

```python
from db import Database, Customer

# Get database instance
db = Database()

# Create
customer = Customer(id="cust123", given_name="John", family_name="Doe")
db.add(customer)

# Read
customer = db.get(Customer, "cust123")

# Update
customer.email_address = "john.doe@example.com"
db.update(customer)

# Delete
db.delete(customer)
```

### Querying

```python
from db import Database, Customer
from sqlalchemy import or_

# Basic query with filters
db = Database()
customers = db.query(Customer, 
                    or_(Customer.given_name == "John",
                        Customer.family_name == "Doe"))

# Raw SQL query
result = db.execute("SELECT * FROM customers WHERE email_address LIKE ?", 
                    {"email": "%example.com"})
```

### Converting Between Models and Dictionaries

```python
# Convert dictionary to model
customer_dict = {"id": "cust123", "given_name": "John", "family_name": "Doe"}
customer = Customer.from_dict(customer_dict)

# Convert model to dictionary
customer_dict = customer.to_dict()
```

## Data Migration

The module includes utilities for migrating existing JSON data to the database:

### Command-line Migration

Run the migration script to move data from JSON to SQLite:

```bash
python scripts/migrate_json_to_db.py --customers
```

### Programmatic Migration

```python
from db import migrate_customers_from_json

# Migrate customer data
result = migrate_customers_from_json()
print(f"Migrated {result['customers_migrated']} customers")
```

## Integration with Tools

Tools can leverage the database through their implementation. For example, the `CustomerToolDB` class provides all the functionality of the original `CustomerTool` but uses the database for persistence.

The database implementation handles:

1. Automatic migration of existing data
2. Transparent caching for performance
3. Direct SQL queries for efficient searches
4. Compatibility with the original tool interface

## Configuration

Database configuration is managed through the main config system:

```python
# in config/config.py
class DatabaseConfig(BaseModel):
    uri: str = Field(
        default="sqlite:///data/app.db",
        description="Database connection URI"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL commands for debugging"
    )
    # ... other database config options
```

## Extending with New Models

To add support for other tools:

1. Define a new model in `db.py` that inherits from `Base`
2. Add appropriate columns and relationships
3. Implement `to_dict()` and `from_dict()` methods
4. Update the tool to use the database interface

Example:

```python
class Note(Base):
    __tablename__ = 'notes'
    id = Column(String, primary_key=True)
    title = Column(String)
    content = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            title=data.get("title"),
            content=data.get("content")
        )
```