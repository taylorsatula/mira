"""
Mock pgvector for testing without PostgreSQL.
"""

from unittest.mock import MagicMock
from sqlalchemy import Column as RealColumn
from sqlalchemy.types import TypeDecorator, String


class MockVector(TypeDecorator):
    """Mock Vector type that stores as JSON string."""
    impl = String
    cache_ok = True
    
    def __init__(self, dim=None):
        self.dim = dim
        super().__init__()
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value.tolist() if hasattr(value, 'tolist') else value)
    
    def process_result_value(self, value, dialect):
        return value


def setup_pgvector_mocks():
    """Set up all pgvector mocks for testing."""
    import sys
    
    # Create mock module
    mock_pgvector = MagicMock()
    mock_pgvector_sqlalchemy = MagicMock()
    
    # Add the Vector class
    mock_pgvector_sqlalchemy.Vector = MockVector
    
    # Register the mocks
    sys.modules['pgvector'] = mock_pgvector
    sys.modules['pgvector.sqlalchemy'] = mock_pgvector_sqlalchemy
    
    return mock_pgvector, mock_pgvector_sqlalchemy