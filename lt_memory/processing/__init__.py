"""
The memory processing package: extraction, response parsing, storage, consolidation.

Each module owns one stage of the pipeline and is wired together by
``lt_memory/factory.py``; in-repo callers import those submodules directly.
This root only re-exports the seven ``__all__`` names, so a name added to a
submodule and used by a caller is not necessarily reachable from here.
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
