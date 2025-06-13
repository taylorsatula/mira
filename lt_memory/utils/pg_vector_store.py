"""
PostgreSQL-native vector store using pgvector extension.

Installation:
    pip install pgvector psycopg2-binary numpy
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PGVectorStore:
    """
    PostgreSQL-native vector store using pgvector.
    
    Features:
    - Native pgvector support for similarity search
    - Connection pooling for better performance
    - Automatic index optimization
    - Context manager support
    """
    
    def __init__(self, 
                 connection_string: str, 
                 dimension: int = 1024,
                 pool_size: int = 5):
        """
        Initialize vector store with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string
            dimension: Vector dimension size
            pool_size: Number of connections in the pool
        """
        self.connection_string = connection_string
        self.dimension = dimension
        
        # Create connection pool
        self._pool = SimpleConnectionPool(
            1, pool_size,
            connection_string,
            cursor_factory=RealDictCursor
        )
        
        # Register vector type for all connections
        self._init_connections()
        
        # Verify pgvector extension
        self._verify_pgvector()
    
    def _init_connections(self):
        """Initialize all connections in the pool with vector type."""
        # Get all connections and register vector type
        for i in range(self._pool.minconn):
            conn = self._pool.getconn()
            register_vector(conn)
            self._pool.putconn(conn)
    
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)
    
    def _verify_pgvector(self) -> None:
        """Verify pgvector extension is installed."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                
                if not result:
                    raise RuntimeError(
                        "pgvector extension not found. Please install with: CREATE EXTENSION vector;"
                    )
                
                logger.info("pgvector extension verified")
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10, 
               table: str = "memory_passages",
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using pgvector.
        
        Args:
            query_embedding: Query vector (numpy array)
            k: Number of results to return
            table: Table name to search
            filters: Optional filters to apply
            
        Returns:
            List of (id, similarity_score) tuples
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[0]} != {self.dimension}")
        
        # Ensure float32 for consistency
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Build query with filters
        query_parts = [f"""
            SELECT id,
                   1 - (embedding <=> %s) as similarity,
                   importance_score
            FROM {table}
            WHERE embedding IS NOT NULL
        """]
        
        params = [query_embedding]
        
        # Add filter conditions
        if filters:
            if "source" in filters:
                query_parts.append("AND source = %s")
                params.append(filters["source"])
            
            if "min_importance" in filters:
                query_parts.append("AND importance_score >= %s")
                params.append(filters["min_importance"])
            
            if "created_after" in filters:
                query_parts.append("AND created_at >= %s")
                params.append(filters["created_after"])
            
            if "min_similarity" in filters:
                query_parts.append("AND 1 - (embedding <=> %s) >= %s")
                params.extend([query_embedding, filters["min_similarity"]])
        
        # Complete query
        query_parts.extend([
            f"ORDER BY embedding <=> %s",
            "LIMIT %s"
        ])
        params.extend([query_embedding, k])
        
        complete_sql = "\n".join(query_parts)
        
        # Execute query
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(complete_sql, params)
                rows = cur.fetchall()
                
                # Collect results and passage IDs
                passage_ids = []
                results = []
                
                for row in rows:
                    passage_ids.append(str(row['id']))
                    results.append((str(row['id']), float(row['similarity'])))
                
                # Update access counts if we have results
                if passage_ids:
                    update_sql = f"""
                    UPDATE {table} 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ANY(%s::uuid[])
                    """
                    cur.execute(update_sql, (passage_ids,))
                    conn.commit()
        
        logger.debug(f"Vector search returned {len(results)} results")
        return results
    
    def optimize_index(self, 
                      table: str = "memory_passages", 
                      lists: Optional[int] = None) -> None:
        """
        Optimize IVFFlat index for better performance.
        
        Args:
            table: Table name with vector column
            lists: Number of lists for IVFFlat index (auto-calculated if None)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Get row count if lists not specified
                if lists is None:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cur.fetchone()['count'] or 0
                    
                    # PostgreSQL recommendation: lists = rows/1000
                    lists = max(1, min(1000, row_count // 1000))
                    
                    logger.info(f"Auto-calculated lists={lists} for {row_count} rows")
                
                # Set maintenance work memory
                cur.execute("SET maintenance_work_mem = '512MB'")
                
                # Recreate index
                cur.execute(f"DROP INDEX IF EXISTS idx_{table}_embedding")
                cur.execute(f"""
                    CREATE INDEX idx_{table}_embedding 
                    ON {table} 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = %s)
                """, (lists,))
                
                conn.commit()
                
                logger.info(f"Optimized vector index on {table} with lists={lists}")
    
    def analyze_performance(self, 
                           query_embedding: np.ndarray, 
                           table: str = "memory_passages") -> Dict[str, Any]:
        """
        Analyze query performance for troubleshooting.
        
        Args:
            query_embedding: Sample query vector
            table: Table to analyze
            
        Returns:
            Performance metrics and query plan
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Get query plan
                explain_sql = f"""
                EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT id, embedding <=> %s as distance
                FROM {table}
                ORDER BY embedding <=> %s
                LIMIT 10
                """
                
                cur.execute(explain_sql, (query_embedding, query_embedding))
                result = cur.fetchone()
                plan = result['QUERY PLAN'][0] if result else {}
                
                # Get index statistics
                cur.execute("""
                    SELECT 
                        schemaname,
                        relname as tablename,
                        indexrelname as indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE relname = %s
                """, (table,))
                stats = cur.fetchall()
                
                # Get table size information
                cur.execute("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size(%s::regclass)) as total_size,
                        pg_size_pretty(pg_relation_size(%s::regclass)) as table_size,
                        pg_size_pretty(pg_indexes_size(%s::regclass)) as index_size
                """, (table, table, table))
                size_result = cur.fetchone()
                
                return {
                    "query_plan": plan,
                    "index_stats": list(stats),
                    "table_size": dict(size_result) if size_result else {},
                    "execution_time_ms": plan.get("Execution Time", 0) if plan else 0
                }
    
    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, '_pool') and self._pool:
            try:
                self._pool.closeall()
            except:
                pass  # Pool might already be closed
            finally:
                self._pool = None
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on context exit."""
        self.close()
    
    def __del__(self):
        """Clean up connections on deletion."""
        self.close()


# Utility functions for common operations
def create_vector_table(connection_string: str, 
                       table_name: str = "memory_passages",
                       dimension: int = 1024):
    """
    Create a vector table with all necessary columns and indexes.
    
    This is a utility to help set up new tables.
    """
    conn = psycopg2.connect(connection_string)
    register_vector(conn)
    
    try:
        with conn.cursor() as cur:
            # Create extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT,
                    embedding vector({dimension}),
                    source TEXT,
                    importance_score FLOAT DEFAULT 0.5,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata 
                ON {table_name} USING gin (metadata)
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_created 
                ON {table_name} (created_at DESC)
            """)
            
            conn.commit()
            logger.info(f"Created vector table: {table_name}")
            
    finally:
        conn.close()