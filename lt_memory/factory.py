"""
Factory for creating and managing LT_Memory service instances.

Replaces global singleton pattern with explicit dependency management,
enabling easier testing and clearer lifecycle control.
"""
import logging
from typing import Optional

from lt_memory.db_access import LTMemoryDB
from lt_memory.vector_ops import VectorOps
from lt_memory.linking import LinkingService
from lt_memory.proactive import ProactiveService
from lt_memory.hub_discovery import HubDiscoveryService
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.processing.extraction_engine import ExtractionEngine
from lt_memory.processing.execution_strategy import create_execution_strategy
from lt_memory.processing.orchestrator import ExtractionOrchestrator
from lt_memory.processing.consolidation_handler import ConsolidationHandler
from utils.database_session_manager import LTMemorySessionManager, get_shared_session_manager

logger = logging.getLogger(__name__)

# Singleton instance
_lt_memory_factory_instance: Optional['LTMemoryFactory'] = None


class LTMemoryFactory:
    """
    Creates and manages all LT_Memory service instances with explicit dependencies.

    Services use module-level constants for algorithm tuning (no config injection).
    The factory wires infrastructure dependencies (DB, embeddings, LLM provider)
    and manages service lifecycle.

    Memory extraction uses the fixed ``batch`` model config through LLMProvider.
    """

    def __init__(
        self,
        session_manager: LTMemorySessionManager,
        embeddings_provider,
        llm_provider,
        conversation_repo
    ):
        logger.info("Initializing LTMemoryFactory")

        self._session_manager = session_manager
        self._embeddings_provider = embeddings_provider
        self._llm_provider = llm_provider
        self._conversation_repo = conversation_repo

        # Late-registered CNS callback: invoked by store_and_tend_extraction
        # after memories are stored, so the SegmentCollapseHandler (which owns
        # tool_repo + event_bus) can spawn the MemoryCuratorAgent in integration
        # mode. None until registered; store_and_tend_extraction is None-safe.
        # lt_memory never imports from agents/ — the callback is the seam.
        self.on_memories_stored = None

        # Track initialization order for reverse cleanup
        self._service_init_order = []

        # Build dependency graph in order
        self._init_services()

        logger.info("LTMemoryFactory initialization complete")

    def _init_services(self) -> None:
        """Initialize all services in dependency order, tracking for reverse cleanup."""
        try:
            # Layer 1: Database access (no dependencies)
            logger.debug("Initializing LTMemoryDB...")
            self.db = LTMemoryDB(self._session_manager)
            self._service_init_order.append(self.db)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LTMemoryDB: {e}") from e

        try:
            # Layer 2: Vector operations (depends on db)
            logger.debug("Initializing VectorOps...")
            self.vector_ops = VectorOps(
                embeddings_provider=self._embeddings_provider,
                db=self.db,
            )
            self._service_init_order.append(self.vector_ops)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VectorOps: {e}") from e

        try:
            # Layer 3: Core services (depend on db + vector_ops)
            logger.debug("Initializing LinkingService...")
            self.linking = LinkingService(
                vector_ops=self.vector_ops,
                db=self.db
            )
            self._service_init_order.append(self.linking)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LinkingService: {e}") from e

        try:
            # Layer 3.5: New processing components (depend on db + vector_ops)
            logger.debug("Initializing MemoryProcessor...")
            self.memory_processor = MemoryProcessor(
                vector_ops=self.vector_ops
            )
            self._service_init_order.append(self.memory_processor)

            logger.debug("Initializing ExtractionEngine...")
            self.extraction_engine = ExtractionEngine(
                db=self.db
            )
            self._service_init_order.append(self.extraction_engine)

            logger.debug("Initializing ExecutionStrategy...")
            self.execution_strategy = create_execution_strategy(
                extraction_engine=self.extraction_engine,
                memory_processor=self.memory_processor,
                vector_ops=self.vector_ops,
                db=self.db,
                llm_provider=self._llm_provider,
                linking_service=self.linking
            )
            self._service_init_order.append(self.execution_strategy)

            logger.debug("Initializing ExtractionOrchestrator...")
            self.extraction_orchestrator = ExtractionOrchestrator(
                extraction_engine=self.extraction_engine,
                execution_strategy=self.execution_strategy,
                continuum_repo=self._conversation_repo,
                db=self.db,
            )
            self._service_init_order.append(self.extraction_orchestrator)

            logger.debug("Initializing ConsolidationHandler...")
            self.consolidation_handler = ConsolidationHandler(
                vector_ops=self.vector_ops,
                db=self.db
            )
            self._service_init_order.append(self.consolidation_handler)

            logger.debug("Initializing HubDiscoveryService...")
            self.hub_discovery = HubDiscoveryService(
                db=self.db
            )
            self._service_init_order.append(self.hub_discovery)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize processing components: {e}") from e

        try:
            logger.debug("Initializing ProactiveService...")
            self.proactive = ProactiveService(
                vector_ops=self.vector_ops,
                linking_service=self.linking,
                db=self.db,
                hub_discovery=self.hub_discovery
            )
            self._service_init_order.append(self.proactive)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ProactiveService: {e}") from e

        logger.debug("All LT_Memory services initialized")

    def cleanup(self) -> None:
        """
        Clean up all service resources.

        Call this during application shutdown to release resources properly.
        Services are cleaned up in reverse initialization order automatically.
        """
        logger.info("Cleaning up LTMemoryFactory")

        # Cleanup in reverse initialization order
        for service in reversed(self._service_init_order):
            if service and hasattr(service, 'cleanup'):
                try:
                    service.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up {service.__class__.__name__}: {e}")

        self._service_init_order.clear()
        logger.info("LTMemoryFactory cleanup complete")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LTMemoryFactory(services=[db, vector_ops, linking, "
            f"memory_processor, extraction_engine, execution_strategy, extraction_orchestrator, "
            f"consolidation_handler, hub_discovery, proactive])"
        )


def get_lt_memory_factory(
    session_manager: LTMemorySessionManager = None,
    embeddings_provider = None,
    llm_provider = None,
    conversation_repo = None,
    force_new: bool = False
) -> LTMemoryFactory:
    """
    Get or create the singleton LTMemoryFactory instance.

    Uses MIRA's sequential initialization contract: all singletons are created
    during app startup (main.py:lifespan) in a single thread. Not thread-safe
    during initialization.

    Args:
        session_manager: Database session manager (required on first call)
        embeddings_provider: Embeddings provider (required on first call)
        llm_provider: LLM provider (required on first call)
        conversation_repo: Continuum repository (required on first call)
        force_new: Force creation of a new instance (for testing)

    Returns:
        LTMemoryFactory singleton instance

    Raises:
        RuntimeError: If called without required arguments on first call
    """
    global _lt_memory_factory_instance

    if force_new and _lt_memory_factory_instance:
        logger.info("Forcing cleanup of existing LTMemoryFactory")
        _lt_memory_factory_instance.cleanup()
        _lt_memory_factory_instance = None

    if _lt_memory_factory_instance is None:
        if not all([session_manager, embeddings_provider,
                    llm_provider, conversation_repo]):
            raise RuntimeError(
                "First call to get_lt_memory_factory requires all arguments: "
                "session_manager, embeddings_provider, llm_provider, "
                "conversation_repo"
            )

        logger.info("Creating new LTMemoryFactory singleton")
        _lt_memory_factory_instance = LTMemoryFactory(
            session_manager=session_manager,
            embeddings_provider=embeddings_provider,
            llm_provider=llm_provider,
            conversation_repo=conversation_repo
        )

    return _lt_memory_factory_instance
