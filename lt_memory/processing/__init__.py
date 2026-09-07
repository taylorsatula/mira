"""
Memory processing pipeline - clean, surgical implementation.

This package contains the rebuilt extraction pipeline with proper separation of concerns:
- memory_processor: Parse and validate LLM responses
- extraction_engine: Build extraction payloads and prompts
- execution_strategy: Execute directly through model_config=batch
- consolidation_handler: Handle memory consolidation
- orchestrator: High-level workflow coordination
"""

from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.processing.extraction_engine import ExtractionEngine, ExtractionPayload
from lt_memory.processing.execution_strategy import (
    DirectExecutionStrategy,
    create_execution_strategy
)
from lt_memory.processing.consolidation_handler import ConsolidationHandler
from lt_memory.processing.orchestrator import ExtractionOrchestrator

__all__ = [
    "MemoryProcessor",
    "ExtractionEngine",
    "ExtractionPayload",
    "DirectExecutionStrategy",
    "create_execution_strategy",
    "ConsolidationHandler",
    "ExtractionOrchestrator",
]
