# LT_Memory Implementation Guide for MIRA
*A comprehensive guide for implementing persistent, self-managed memory inspired by MemGPT*

## Overview

This guide documents the implementation of a sophisticated memory system for MIRA that combines the elegance of MemGPT's self-editing memory with advanced features from Letta, while leveraging MIRA's existing architecture. The goal is to give MIRA continuous cognitive existence through persistent, searchable, and evolving memory.

### Core Concept

The LT_Memory system provides three layers of memory:
1. **Core Memory**: Always-visible, self-editable context blocks
2. **Recall Memory**: Searchable conversation history
3. **Archival Memory**: Vector-indexed long-term knowledge storage

Unlike Letta's platform approach, this implementation treats memory as a modular component that enhances MIRA without requiring architectural changes.

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                      MIRA Main Process                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Conversation│───▶│WorkingMemory │◀───│  LT_Memory    │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                              ▲                    │          │
│                              │                    ▼          │
│  ┌─────────────┐            │           ┌───────────────┐  │
│  │   LLMBridge │            │           │MemoryManager │  │
│  └─────────────┘            │           └───────────────┘  │
│                              │                    │          │
│                              ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Automation Engine                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Hourly    │  │    Daily    │  │   Weekly    │  │  │
│  │  │ Processing  │  │Consolidation│  │   Review    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        PostgreSQL                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Blocks  │  │ Passages │  │ Entities │  │Relations │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ History  │  │ Vectors  │  │Snapshots │  │ Indexes  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **MemoryManager**: Central orchestrator for all memory operations
2. **BlockManager**: Handles core memory blocks with versioning
3. **PassageManager**: Manages archival memory with vector search
4. **EntityManager**: Builds and maintains knowledge graph
5. **ConsolidationEngine**: Handles memory optimization and summarization
6. **MemoryBridge**: Interfaces between WorkingMemory and LT_Memory
7. **BatchProcessor**: Processes conversations in scheduled batches

## Prerequisites

### Required Dependencies

```toml
# Add to requirements.txt
sqlalchemy>=2.0.0
alembic>=1.13.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
onnxruntime>=1.16.0
transformers>=4.35.0
numpy>=1.24.0
networkx>=3.0
jinja2>=3.1.0
```

### PostgreSQL Setup

```bash
# Install PostgreSQL and pgvector
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector

# Create database
sudo -u postgres psql
CREATE DATABASE lt_memory;
CREATE USER mira WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lt_memory TO mira;
\c lt_memory
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

### ONNX Model Setup

```bash
# Download and convert model to ONNX format
python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 \
  --feature=default onnx/
```

### Configuration

```python
# config/config.py additions
class MemoryConfig(BaseModel):
    """LT_Memory configuration settings."""
    
    # PostgreSQL connection
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "LT_MEMORY_DATABASE_URL",
            "postgresql://mira:secure_password@localhost/lt_memory"
        ),
        description="PostgreSQL connection URL"
    )
    
    # Database pool settings
    db_pool_size: int = Field(
        default=10,
        description="Database connection pool size"
    )
    db_pool_max_overflow: int = Field(
        default=20,
        description="Maximum overflow connections"
    )
    
    # Core memory settings
    core_memory_blocks: Dict[str, int] = Field(
        default={"persona": 2048, "human": 2048, "system": 1024},
        description="Core memory blocks and their character limits"
    )
    
    # ONNX model settings
    onnx_model_path: str = Field(
        default="onnx/model.onnx",
        description="Path to ONNX model file"
    )
    onnx_tokenizer: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Tokenizer name for ONNX model"
    )
    embedding_dim: int = Field(
        default=384,
        description="Embedding dimension size"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    # Search settings
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval"
    )
    max_search_results: int = Field(
        default=10,
        description="Maximum number of search results"
    )
    
    # Batch processing settings
    batch_process_hours: int = Field(
        default=1,
        description="Hours of conversations to process per batch"
    )
    
    # Consolidation settings
    consolidation_threshold: int = Field(
        default=1000,
        description="Message count before triggering consolidation"
    )
    max_memory_age_days: int = Field(
        default=90,
        description="Days before memory enters cold storage"
    )
    
    # Knowledge graph settings
    entity_extraction_enabled: bool = Field(
        default=True,
        description="Enable entity extraction and knowledge graph"
    )
    relationship_inference_enabled: bool = Field(
        default=True,
        description="Enable relationship inference between entities"
    )
```

## Implementation Details

### Phase 1: Core Database Models

The system uses PostgreSQL with pgvector for efficient vector operations. All models use UUID primary keys and JSONB for flexible metadata storage.

### Phase 2: Memory Management

The MemoryManager orchestrates all operations, with specialized managers for blocks, passages, and entities. Each manager handles its own domain with clear interfaces.

### Phase 3: ONNX Embeddings

ONNX Runtime provides optimized embedding generation with CPU-specific optimizations. The embedding cache prevents redundant computations.

### Phase 4: Batch Processing

Conversations are processed in scheduled batches rather than real-time, allowing for more sophisticated analysis and reducing system overhead.

### Phase 5: Automation Integration

The automation engine handles all scheduled memory tasks, from hourly processing to weekly reviews, maintaining memory health automatically.

## Testing Strategy

### Unit Tests
- Test core memory operations (append, replace, insert, rethink)
- Verify character limit enforcement
- Validate version tracking
- Test embedding generation and caching

### Integration Tests
- Test working memory bridge updates
- Verify automation execution
- Test batch conversation processing
- Validate entity extraction and relationships

### Performance Tests
- Benchmark embedding generation speed
- Test vector search scalability
- Measure batch processing throughput
- Monitor database query performance

## Best Practices

### 1. Memory Hygiene
- Schedule regular consolidation to prevent unbounded growth
- Set appropriate importance thresholds for archival
- Monitor embedding cache size and clear periodically
- Use batch processing windows during low-activity periods

### 2. Performance Optimization
- Use PostgreSQL connection pooling
- Enable query plan caching
- Optimize pgvector indexes based on data size
- Batch embeddings generation for efficiency

### 3. Privacy and Security
- Never store sensitive information in embeddings
- Use PostgreSQL row-level security if needed
- Regular backups with point-in-time recovery
- Encrypt database connections with SSL

### 4. Monitoring
- Track memory operation latencies
- Monitor database growth and vacuum status
- Alert on consolidation failures
- Watch for embedding dimension mismatches

## Troubleshooting

### Issue: Slow vector searches

**Solution:**
```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM memory_passages 
ORDER BY embedding <=> '[...]'::vector 
LIMIT 10;

-- Optimize index lists based on data size
DROP INDEX idx_passage_embedding;
CREATE INDEX idx_passage_embedding ON memory_passages 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);  -- Adjust based on row count
```

### Issue: ONNX model loading errors

**Solution:**
```python
# Verify ONNX model
import onnx
model = onnx.load("onnx/model.onnx")
onnx.checker.check_model(model)

# Check runtime compatibility
import onnxruntime
print(onnxruntime.get_device())
print(onnxruntime.get_available_providers())
```

### Issue: Memory not appearing in context

**Solution:**
```python
# Check automation execution
SELECT * FROM automation_executions 
WHERE automation_name LIKE '%Memory%' 
ORDER BY executed_at DESC LIMIT 10;

# Manually trigger update
memory_bridge.update_working_memory()
print(working_memory.get_prompt_content())
```

### Issue: Database connection pool exhaustion

**Solution:**
```python
# Monitor connections
SELECT count(*) FROM pg_stat_activity 
WHERE datname = 'lt_memory';

# Adjust pool settings
config.memory.db_pool_size = 20
config.memory.db_pool_max_overflow = 40
```

## Migration Path

### From Existing System

1. **Export conversation history** to JSON format
2. **Run migration script** to populate initial memories
3. **Enable automations** one at a time
4. **Monitor performance** and adjust settings
5. **Gradually increase** processing frequency

### Future Enhancements

1. **Multi-model embeddings** for improved retrieval
2. **Hierarchical memory organization** with topics
3. **Cross-conversation pattern mining**
4. **Memory compression** for older passages
5. **Distributed processing** for large deployments

## Conclusion

This implementation gives MIRA a sophisticated memory system that:
- Provides persistent, self-editing core memory like MemGPT
- Uses PostgreSQL and pgvector for production-ready performance
- Leverages ONNX for optimized embedding generation
- Processes memories in efficient scheduled batches
- Integrates seamlessly with MIRA's automation engine
- Scales from personal assistant to enterprise deployment

The system achieves "continuous cognitive existence" while remaining modular and maintainable, allowing MIRA to develop its own memory personality over time.