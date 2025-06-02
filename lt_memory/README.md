# LT_Memory Package

Long-term memory system for MIRA providing persistent, self-managed memory capabilities inspired by MemGPT.

## Overview

LT_Memory provides three layers of memory:
1. **Core Memory**: Always-visible, self-editable context blocks
2. **Recall Memory**: Searchable conversation history  
3. **Archival Memory**: Vector-indexed long-term knowledge storage

## Quick Start

### Prerequisites

1. PostgreSQL with pgvector extension:
```bash
sudo apt-get install postgresql postgresql-contrib postgresql-14-pgvector
```

2. Create database:
```sql
CREATE DATABASE lt_memory;
CREATE USER mira WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lt_memory TO mira;
```

3. Set environment variable:
```bash
export LT_MEMORY_DATABASE_URL="postgresql://mira:secure_password@localhost/lt_memory"
```

4. Setup ONNX model:
```bash
python scripts/setup_onnx.py
```

### Initialize Database

```bash
python lt_memory/migrations/init_database.py
```

### Integration with MIRA

In your `main.py`, add:

```python
from lt_memory.integration import initialize_lt_memory

# In initialize_system function:
lt_memory = initialize_lt_memory(config, working_memory, tool_repo, automation_controller)
```

## Core Features

### Self-Editing Memory Functions

- `core_memory_append`: Add content to memory blocks
- `core_memory_replace`: Replace specific content
- `memory_insert`: Insert at specific line numbers
- `memory_rethink`: Complete rewrite of blocks

### Archival Search

- Vector similarity search using pgvector
- Importance-based filtering
- Source tracking (conversations, documents, etc.)

### Knowledge Graph

- Automatic entity extraction
- Relationship inference
- Graph traversal queries

### Automated Processing

- Hourly conversation archival
- Daily memory consolidation
- Weekly memory review
- Importance score updates

## Architecture

```
MemoryManager (orchestrator)
├── BlockManager (core memory)
├── PassageManager (archival memory)
├── EntityManager (knowledge graph)
├── ConsolidationEngine (optimization)
└── BatchProcessor (scheduled processing)
```

## Memory Tool Usage

The `lt_memory` tool provides these operations:

```python
# Append to core memory
{"operation": "core_memory_append", "label": "human", "content": "User prefers Python"}

# Search archival memory
{"operation": "search_archival", "query": "programming languages", "limit": 5}

# Get entity information
{"operation": "get_entity_info", "entity_name": "Python"}

# Process recent conversations
{"operation": "process_recent_conversations", "hours": 1}
```

## Automations

Pre-configured automations handle memory maintenance:

1. **Hourly Processing**: Archives recent conversations
2. **Daily Consolidation**: Optimizes memory storage
3. **Weekly Review**: Updates user understanding
4. **Health Monitoring**: Checks system status

## Performance Considerations

- Uses PostgreSQL connection pooling
- ONNX-optimized embeddings with caching
- IVFFlat indexes for fast vector search
- Batch processing to reduce overhead

## Troubleshooting

### Check system health:
```python
from lt_memory.integration import check_lt_memory_requirements
status = check_lt_memory_requirements()
print(status)
```

### View memory statistics:
```bash
# Use the lt_memory tool
{"operation": "get_memory_stats"}
```

### Common Issues

1. **Slow vector searches**: Optimize IVFFlat index lists parameter
2. **High memory usage**: Clear embedding cache periodically
3. **Missing conversations**: Check batch processor status

## Development

### Running Tests
```bash
pytest tests/test_lt_memory*.py
```

### Adding New Memory Types
1. Create model in `models/base.py`
2. Add manager in `managers/`
3. Extend tool interface in `tools/memory_tool.py`

### Custom Entity Extractors
Override the default regex-based extractor:
```python
memory_manager.entity_manager.set_extractor(CustomEntityExtractor())
```

## Configuration

See `config/config.py` for all available settings:
- Database connection parameters
- Embedding model selection
- Search thresholds
- Consolidation policies
- Batch processing windows