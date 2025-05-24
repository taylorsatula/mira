"""
Initialize LT_Memory database schema.

Run this script to set up the PostgreSQL database for LT_Memory.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text
from lt_memory.models.base import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database(database_url: str):
    """
    Initialize the LT_Memory database.
    
    Args:
        database_url: PostgreSQL connection URL
    """
    logger.info("Initializing LT_Memory database...")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create extensions
    with engine.connect() as conn:
        logger.info("Creating PostgreSQL extensions...")
        
        # Create pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("✓ pgvector extension")
        
        # Create UUID extension
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        logger.info("✓ uuid-ossp extension")
        
        conn.commit()
    
    # Create all tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(engine)
    logger.info("✓ All tables created")
    
    # Create initial indexes
    with engine.connect() as conn:
        logger.info("Creating additional indexes...")
        
        # Check if index exists before creating
        check_index = """
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'memory_passages' 
        AND indexname = 'idx_memory_passages_embedding'
        """
        
        result = conn.execute(text(check_index)).fetchone()
        if not result:
            # Create IVFFlat index for vector search
            conn.execute(text("""
            CREATE INDEX idx_memory_passages_embedding 
            ON memory_passages 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
            """))
            logger.info("✓ Vector search index created")
        else:
            logger.info("✓ Vector search index already exists")
        
        conn.commit()
    
    # Verify installation
    with engine.connect() as conn:
        # Check tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        
        tables = conn.execute(text(tables_query)).fetchall()
        logger.info(f"\nCreated {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        # Check extensions
        ext_query = """
        SELECT extname, extversion 
        FROM pg_extension 
        WHERE extname IN ('vector', 'uuid-ossp')
        """
        
        extensions = conn.execute(text(ext_query)).fetchall()
        logger.info(f"\nInstalled extensions:")
        for ext in extensions:
            logger.info(f"  - {ext[0]} v{ext[1]}")
    
    logger.info("\n✅ Database initialization complete!")


def main():
    """Main entry point."""
    # Get database URL from environment or use default
    database_url = os.getenv(
        "LT_MEMORY_DATABASE_URL",
        "postgresql://mira:mira@localhost/lt_memory"
    )
    
    logger.info(f"Database URL: {database_url}")
    
    try:
        init_database(database_url)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()