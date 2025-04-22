# SQLite Database for Bot With Memory

This implementation adds SQLite database support to the Bot With Memory project, specifically for tool data storage. The primary focus is to provide structured data storage while maintaining compatibility with the existing conversation handling.

## Features

- SQLite-based storage using SQLAlchemy ORM
- Full compatibility with existing tool interfaces
- Migration utilities for existing JSON data
- Comprehensive test coverage
- Simple, direct SQL query capabilities
- Flexible extension model for future tools

## Key Components

1. **Database Module** (`db.py`): Core database functionality
   - Connection management
   - Model definitions (Customer model)
   - CRUD operations
   - Migration utilities

2. **CustomerToolDB** (`tools/customer_tool_db.py`): Database-backed customer tool
   - Transparent caching
   - Automatic data migration
   - Efficient SQL-based searching
   - Location-based queries

3. **Migration Utilities**
   - Command-line migration script (`scripts/migrate_json_to_db.py`)
   - Programmatic migration function (`migrate_customers_from_json()`)

4. **Documentation**
   - Implementation guide
   - API documentation
   - Usage examples

## Getting Started

### Setup

The database functionality requires SQLAlchemy, which is already included in the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Configuration

Database configuration is managed through the existing config system:

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
    # ... other settings
```

### Migrating Existing Data

To migrate existing JSON data to the database:

```bash
python scripts/migrate_json_to_db.py --customers
```

### Using the Database-Backed Tool

The CustomerToolDB class is registered with the name `customer_db_tool` and coexists with the original JSON-based `customer_tool`:

```python
from tools.customer_tool_db import CustomerToolDB

# Create the tool instance
customer_db_tool = CustomerToolDB()

# Use it just like the original CustomerTool but with different name
result = customer_db_tool.run("search_customers", kwargs=json.dumps({"query": "John"}))
```

#### Tool Name Differences

- Original JSON-based tool: `customer_tool`
- SQLite database-based tool: `customer_db_tool`

Both tools provide the same interface and can be used interchangeably, but the database-backed version offers better performance for large datasets.

## Architecture

### Database Schema

**Customers Table**
- `id` (primary key, string)
- Basic contact fields (`given_name`, `family_name`, `email_address`, etc.)
- Address fields (`address_line1`, `city`, etc.)
- Geocoding fields (`latitude`, `longitude`, `geocoded_at`)
- Metadata fields (`created_at`, `updated_at`)
- `additional_data` (JSON field for extensibility)

### Interface Design

The database module provides a simple, consistent interface:

```python
Database.get_session() -> Session
Database.add(obj) -> obj
Database.get(model, id) -> obj
Database.query(model, *filters) -> [obj]
Database.update(obj) -> obj
Database.delete(obj) -> bool
Database.execute(statement, params) -> result
```

## Extension Strategy

When adding support for other tools:

1. Define a new model in `db.py` that inherits from `Base`
2. Add appropriate columns and relationships
3. Implement `to_dict()` and `from_dict()` methods
4. Update the tool to use the database interface

## Testing

Comprehensive tests are provided:

```bash
python -m pytest tests/test_db.py
python -m pytest tests/test_customer_tool_db.py
```