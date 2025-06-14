#!/usr/bin/env python3
"""
Flush all embeddings from the vector database.

This script removes all vector embeddings from the PostgreSQL database,
allowing the system to re-embed everything using the new embedding provider
on the next startup.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from config import config
from lt_memory.utils.pg_vector_store import PGVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def flush_embeddings(dry_run: bool = False):
    """
    Flush all embeddings from the vector database.
    
    Args:
        dry_run: If True, show what would be deleted without actually deleting
    """
    logger.info("Starting embeddings flush process...")
    
    try:
        # Get database connection
        db_url = config.memory.database_url
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Check current embedding counts
            tables = ['memory_blocks', 'memory_passages']
            
            for table in tables:
                # Count current embeddings
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                logger.info(f"Table {table}: {count} records with embeddings")
                
                if not dry_run and count > 0:
                    # Clear embeddings by setting them to NULL
                    logger.info(f"Clearing embeddings from {table}...")
                    conn.execute(text(f"UPDATE {table} SET embedding = NULL"))
                    conn.commit()
                    logger.info(f"✓ Cleared embeddings from {table}")
                elif dry_run:
                    logger.info(f"[DRY RUN] Would clear embeddings from {table}")
        
        # Also clear the vector index if it exists
        vector_store = PGVectorStore(
            connection_string=db_url,
            dimension=config.memory.embedding_dim
        )
        
        if not dry_run:
            logger.info("Clearing vector indices...")
            # The vector store will rebuild indices on next use
            with engine.connect() as conn:
                # Drop and recreate any vector indices
                try:
                    conn.execute(text("DROP INDEX IF EXISTS idx_memory_blocks_embedding"))
                    conn.execute(text("DROP INDEX IF EXISTS idx_memory_passages_embedding"))
                    conn.commit()
                    logger.info("✓ Dropped vector indices (will be recreated on next use)")
                except Exception as e:
                    logger.warning(f"Could not drop indices (may not exist): {e}")
        
        logger.info("\n" + "="*50)
        if dry_run:
            logger.info("DRY RUN COMPLETE - No changes made")
            logger.info("Run without --dry-run to actually flush embeddings")
        else:
            logger.info("EMBEDDINGS FLUSH COMPLETE!")
            logger.info("The system will re-embed all content on next startup")
            logger.info("This process may take some time depending on data volume")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error flushing embeddings: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Flush all embeddings from the vector database")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.confirm:
        print("\n" + "!"*50)
        print("WARNING: This will delete ALL embeddings from the database!")
        print("The system will need to re-embed all content on next startup.")
        print("!"*50 + "\n")
        
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Operation cancelled.")
            sys.exit(0)
    
    flush_embeddings(dry_run=args.dry_run)


if __name__ == "__main__":
    main()