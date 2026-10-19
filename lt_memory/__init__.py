"""
The package's single import surface: the factory, the four service classes,
and the model types, re-exported for callers that want one import path.

Code inside `lt_memory/` and the consumers in this repo import
`lt_memory.<submodule>` directly, so this list is not load-bearing for them.
Every name in `__all__` is imported above it; a name added to `__all__`
without a matching import resolves only when a caller asks for it, not at
package import, so the omission stays invisible until that one import fails.
"""
import logging

from lt_memory.factory import LTMemoryFactory, get_lt_memory_factory
from lt_memory.db_access import LTMemoryDB
from lt_memory.vector_ops import VectorOps
from lt_memory.linking import LinkingService
from lt_memory.proactive import ProactiveService
from lt_memory.models import (
    Memory,
    ExtractedMemory,
    MemoryLink,
    Entity,
    ProcessingChunk,
    RelationshipType,
    MemoryLinkEntry,
    EntityLinkEntry,
    AnnotationEntry,
    LinkMetadata,
    TraversalResult,
    UserMemorySettings,
    MemoryPageResult,
    MemoryContext,
    MemoryContextSnapshot,
    MemoryDict,
)

logger = logging.getLogger(__name__)

__all__ = [
    'LTMemoryFactory',
    'get_lt_memory_factory',

    'LTMemoryDB',
    'VectorOps',
    'LinkingService',
    'ProactiveService',

    'Memory',
    'ExtractedMemory',
    'MemoryLink',
    'Entity',
    'ProcessingChunk',

    'RelationshipType',
    'MemoryLinkEntry',
    'EntityLinkEntry',
    'AnnotationEntry',
    'LinkMetadata',
    'TraversalResult',
    'UserMemorySettings',
    'MemoryPageResult',
    'MemoryContext',
    'MemoryContextSnapshot',
    'MemoryDict',
]
