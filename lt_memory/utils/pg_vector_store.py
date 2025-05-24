"""
PostgreSQL-native vector store using pgvector extension.

Provides efficient similarity search with native PostgreSQL operators.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)


class PGVectorStore:
    """
    PostgreSQL-native vector store using pgvector.
    
    Leverages pgvector's native operators for efficient similarity search
    with IVFFlat indexing support.
    """
    
    def __init__(self, engine: Engine, dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            engine: SQLAlchemy engine connected to PostgreSQL
            dimension: Vector dimension size
        """
        self.engine = engine
        self.dimension = dimension
        
        # Verify pgvector extension
        self._verify_pgvector()
    
    def _verify_pgvector(self) -> None:
        """Verify pgvector extension is installed."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            
            if not result:
                raise RuntimeError(
                    "pgvector extension not found. Please install with: CREATE EXTENSION vector;"
                )
            
            logger.info("pgvector extension verified")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               table: str = "memory_passages",
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using pgvector.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            table: Table name to search
            filters: Optional filters to apply
            
        Returns:
            List of (id, similarity_score) tuples
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[0]} != {self.dimension}")
        
        # Base query using cosine similarity
        query = f"""
        SELECT id, 
               1 - (embedding <=> %s::vector) as similarity,
               importance_score
        FROM {table}
        WHERE embedding IS NOT NULL
        """
        
        params = [query_embedding.tolist()]
        
        # Apply filters
        if filters:
            if "source" in filters:
                query += " AND source = %s"
                params.append(filters["source"])
            
            if "min_importance" in filters:
                query += " AND importance_score >= %s"
                params.append(filters["min_importance"])
            
            if "created_after" in filters:
                query += " AND created_at >= %s"
                params.append(filters["created_after"])
            
            if "min_similarity" in filters:
                query += " AND 1 - (embedding <=> %s::vector) >= %s"
                params.extend([query_embedding.tolist(), filters["min_similarity"]])
        
        # Order by similarity and limit
        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding.tolist(), k])
        
        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            
            # Update access count for retrieved passages
            passage_ids = []
            results = []
            
            for row in result:
                passage_ids.append(str(row.id))
                results.append((str(row.id), float(row.similarity)))
            
            # Batch update access counts
            if passage_ids:
                update_query = f"""
                UPDATE {table} 
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = ANY(%s::uuid[])
                """
                conn.execute(text(update_query), [passage_ids])
                conn.commit()
        
        logger.debug(f"Vector search returned {len(results)} results")
        return results
    
    def optimize_index(self, table: str = "memory_passages", 
                      lists: Optional[int] = None) -> None:
        """
        Optimize IVFFlat index for better performance.
        
        Args:
            table: Table name with vector column
            lists: Number of lists for IVFFlat index (auto-calculated if None)
        """
        with self.engine.connect() as conn:
            # Get row count to determine optimal lists
            if lists is None:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                row_count = result[0] if result else 0
                
                # PostgreSQL recommendation: lists = rows/1000
                # Minimum 1, maximum 1000
                lists = max(1, min(1000, row_count // 1000))
                
                logger.info(f"Auto-calculated lists={lists} for {row_count} rows")
            
            # Set maintenance work memory for index creation
            conn.execute(text("SET maintenance_work_mem = '512MB'"))
            
            # Drop existing index
            conn.execute(text(f"DROP INDEX IF EXISTS idx_{table}_embedding"))
            
            # Create optimized index
            create_index = f"""
            CREATE INDEX idx_{table}_embedding 
            ON {table} 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = %s)
            """
            conn.execute(text(create_index), [lists])
            conn.commit()
            
            logger.info(f"Optimized vector index on {table} with lists={lists}")
    
    def analyze_performance(self, query_embedding: np.ndarray, 
                           table: str = "memory_passages") -> Dict[str, Any]:
        """
        Analyze query performance for troubleshooting.
        
        Args:
            query_embedding: Sample query vector
            table: Table to analyze
            
        Returns:
            Performance metrics and query plan
        """
        with self.engine.connect() as conn:
            # Get query plan
            explain_query = f"""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
            SELECT id, embedding <=> %s::vector as distance
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT 10
            """
            
            result = conn.execute(
                text(explain_query), 
                [query_embedding.tolist(), query_embedding.tolist()]
            ).fetchone()
            
            plan = result[0][0] if result else {}
            
            # Get index statistics
            stats_query = f"""
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE tablename = %s
            """
            
            stats = conn.execute(text(stats_query), [table]).fetchall()
            
            # Get table size
            size_query = f"""
            SELECT 
                pg_size_pretty(pg_total_relation_size(%s::regclass)) as total_size,
                pg_size_pretty(pg_relation_size(%s::regclass)) as table_size,
                pg_size_pretty(pg_indexes_size(%s::regclass)) as index_size
            """
            
            size_result = conn.execute(
                text(size_query), 
                [table, table, table]
            ).fetchone()
            
            return {
                "query_plan": plan,
                "index_stats": [dict(row) for row in stats],
                "table_size": dict(size_result) if size_result else {},
                "execution_time_ms": plan.get("Execution Time", 0) if plan else 0
            }