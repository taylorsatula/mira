# Letta Memory System Integration Guide

This document outlines the key components from Letta's memory management system that can be extracted and integrated into the botwithmemory codebase.

## 1. Memory Structure and Models

### Memory Data Models

**Purpose:** Defines the data structures for different types of memory (core blocks, archival entries).

**Source Files:**
- `letta/schemas/memory.py` - Contains the Memory base class and core memory implementations
- `letta/schemas/passage.py` - Defines the passage structure for archival memory
- `letta/schemas/block.py` - Block data structure for core memory segments

### Memory Integration with Agent State

**Purpose:** Shows how memory is integrated into the agent state and accessed.

**Source Files:**
- `letta/schemas/agent.py` - Contains the AgentState class which includes memory references

## 2. Core Memory Management

### Memory Block Operations

**Purpose:** Functions for manipulating core memory blocks (append, replace, rethink).

**Source Files:**
- `letta/functions/function_sets/base.py` - Contains core memory operation functions
- `letta/memory.py` - Memory utility functions including summarization

### Memory Prompt Construction

**Purpose:** Logic to compile memory blocks into prompt components.

**Source Files:**
- `letta/schemas/memory.py` - The `compile()` method shows how memory blocks are formatted
- `letta/agents/letta_agent.py` - The `_rebuild_memory` method shows memory integration

## 3. Archival Memory and Vector Search

### Passage Management

**Purpose:** Handles creating, storing, and retrieving memory passages.

**Source Files:**
- `letta/services/passage_manager.py` - Core service for passage operations
- `letta/orm/passage.py` - Database models for passages

### Embedding and Search

**Purpose:** Vector embedding creation and similarity search.

**Source Files:**
- `letta/embeddings.py` - Contains embedding models and text chunking
- `letta/functions/function_sets/base.py` - The `archival_memory_search` function

## 4. Memory Integration with Conversation

### Memory Context Window Management

**Purpose:** Logic to manage what memory goes into the context window.

**Source Files:**
- `letta/agents/letta_agent.py` - The `_rebuild_memory` method shows memory refresh
- `letta/services/agent_manager.py` - The `refresh_memory` method

### Summarization for Context Management

**Purpose:** Summarizes conversation history to maintain memory within context limits.

**Source Files:**
- `letta/memory.py` - The `summarize_messages` function
- `letta/services/summarizer/summarizer.py` - Summarization service

## 5. Memory-Related Tools

### Memory Tool Definitions

**Purpose:** Tool definitions for interacting with memory systems.

**Source Files:**
- `letta/functions/function_sets/base.py` - Contains memory-related tool functions
- `letta/functions/functions.py` - Tool registration and metadata

### Memory Tool Implementations

**Purpose:** Actual implementations of memory operations triggered by tools.

**Source Files:**
- `letta/functions/function_sets/base.py` - Implementation of core memory tools
- `letta/services/passage_manager.py` - Implementation of archival memory operations

## Implementation Strategy

This integration can be approached in phases:

1. **Foundation Phase:** Implement the core memory data structures and basic block operations
2. **Core Memory Phase:** Integrate core memory with the conversation context
3. **Archival Phase:** Implement the vector-based archival memory and search
4. **Tool Integration Phase:** Add memory-related tools to the agent's toolset
5. **Context Optimization Phase:** Implement advanced memory context management

The existing topic_changed flag and conversation management in botwithmemory provide excellent integration points for this memory system.