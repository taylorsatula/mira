"""
The single extraction execution path, and the only storage path it writes through.

``DirectExecutionStrategy.execute_extraction`` builds one payload per chunk,
calls ``generate_response(model_config="batch")`` synchronously, hands the raw
response text to ``memory_processor``, and stores through
``store_and_tend_extraction``. There is one strategy; ``create_execution_strategy``
returns it. No deferred or remote submission layer exists, so the segment's
mark in the orchestrator waits on a completed call, not on a queued one.

``store_and_tend_extraction`` orders durability before tending: the embedding
write plus INSERT is a plain non-idempotent insert, and the orchestrator marks
the segment only after this whole call returns. Every post-store step — entity
links, candidate hints, the ``on_memories_stored`` callback — is therefore
tolerate-and-log; a raise from one of them would surface as an unmarked segment
whose next sweep re-inserts the same memories.

Global memories (scope="global", offered only when the store is active) are
written to the store and the local copy BEFORE the personal path runs, and
never reach ``store_and_tend_extraction``: the store write is the only step
whose failure must abort the segment before any personal row lands, because
the sweep re-runs the whole segment and the store answers a repeated POST
with ``duplicate``. A failure after personal rows are inserted would
re-insert them. A deterministic rejection (4xx other than 408/429) demotes
the memory to personal instead of aborting the segment — retrying cannot
help, and the user keeps the memory. Global rows get no entity links,
candidate hints, or curator callback — no retrieval path from a global row
follows them.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID, uuid4

from clients.global_memory_store_client import GlobalMemoryStoreError
from clients.llm_provider import LLMProvider
from lt_memory.db_access import LTMemoryDB
from lt_memory.linking import LinkingService
from lt_memory.models import MEMORY_SCOPE_GLOBAL, ExtractedMemory, ProcessingChunk
from lt_memory.processing.extraction_engine import ExtractionEngine, ExtractionPayload
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps

logger = logging.getLogger(__name__)


def persist_llm_entities(
    user_id: str,
    memories: list[ExtractedMemory],
    memory_ids: list[UUID],
    db: LTMemoryDB,
) -> int:
    """Persist LLM-extracted entity links for stored memories.

    The one sanctioned entity-linking treatment for every producer that
    stores memories outside consolidation: extraction storage
    (``store_and_tend_extraction``) and the pending-manual drain
    (``cns/services/segment_collapse_handler.py``) both route their entity
    links through here, so the two store paths cannot diverge on linking.
    """
    if len(memories) != len(memory_ids):
        raise ValueError(
            f"Memory/ID length mismatch: {len(memories)} memories vs {len(memory_ids)} IDs"
        )

    total_links = 0
    for memory, memory_id in zip(memories, memory_ids):
        seen_names: set[str] = set()
        for entity_dict in memory.entities:
            entity_name = entity_dict["name"]
            if entity_name in seen_names:
                continue
            seen_names.add(entity_name)
            entity = db.get_or_create_entity(
                name=entity_name,
                entity_type=entity_dict.get("type", "UNKNOWN"),
                user_id=user_id,
            )
            db.link_memory_to_entity(
                memory_id=memory_id,
                entity_id=entity.id,
                entity_name=entity_name,
                entity_type=entity.entity_type,
                user_id=user_id,
            )
            total_links += 1
    return total_links


def _build_candidate_hints(
    memories: list[ExtractedMemory],
    memory_ids: list[UUID],
    linking: LinkingService,
) -> dict[str, list[dict]]:
    """Combine deterministic and extraction-provided relationship hints."""
    from utils.tag_parser import format_memory_id

    if len(memories) != len(memory_ids):
        raise ValueError(
            f"Candidate-hints length mismatch: {len(memories)} vs {len(memory_ids)}"
        )

    hints: dict[str, list[dict]] = {}
    for memory, memory_id in zip(memories, memory_ids):
        refs = list(linking.find_candidate_hints(memory_id))
        for related in memory.related_memory_ids:
            refs.append(
                {
                    "memory_id": format_memory_id(related["id"]),
                    "bond": related.get("bond", ""),
                    "discovery_signal": "extraction",
                    "similarity": None,
                }
            )
        if refs:
            hints[str(memory_id)] = refs
    return hints


def store_and_tend_extraction(
    *,
    user_id: str,
    segment_id: Optional[str],
    memories: list[ExtractedMemory],
    vector_ops: VectorOps,
    db: LTMemoryDB,
    linking: LinkingService,
) -> list[UUID]:
    """Store extraction output and notify the integration curator.

    Commit-after-store ordering: store_memories is a plain INSERT, not
    idempotent, and the retry sweep marks a segment processed only after this
    whole call returns. Any raise from the post-store steps below would leave
    the segment unmarked, so the sweep would re-enter and duplicate the rows.
    The memories are the deliverable and are durable once stored; failures in
    the tending steps are tolerated-and-logged and never propagate.
    """
    memory_ids = vector_ops.store_memories_with_embeddings(memories)

    try:
        persist_llm_entities(user_id, memories, memory_ids, db)
    except Exception:
        logger.exception(
            "Entity persistence failed after %d memories were stored; "
            "memories remain durable, entity links skipped for this run",
            len(memory_ids),
        )

    candidate_hints: dict[str, list[dict]] = {}
    try:
        candidate_hints = _build_candidate_hints(memories, memory_ids, linking)
    except Exception:
        logger.warning(
            "Candidate hint discovery failed after %d memories were stored; "
            "memories remain durable, discovery-based hints skipped for this run",
            len(memory_ids),
            exc_info=True,
        )

    try:
        from lt_memory.factory import get_lt_memory_factory

        callback = get_lt_memory_factory().on_memories_stored
        if callback:
            callback(
                user_id=user_id,
                segment_id=segment_id,
                memory_ids=memory_ids,
                memories=memories,
                candidate_hints=candidate_hints,
            )
    except Exception:
        logger.exception(
            "on_memories_stored callback failed after %d memories were stored; "
            "memories remain durable, curator skipped for this run",
            len(memory_ids),
        )
    return memory_ids


class DirectExecutionStrategy:
    """Execute extraction synchronously with model_config=batch."""

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        llm_provider: LLMProvider,
        linking_service: LinkingService,
    ) -> None:
        self.extraction_engine = extraction_engine
        self.memory_processor = memory_processor
        self.vector_ops = vector_ops
        self.db = db
        self.llm_provider = llm_provider
        self.linking = linking_service

    def execute_extraction(self, user_id: str, chunks: list[ProcessingChunk]) -> str:
        """Extract and store every chunk or raise on any failed dependency."""
        built_payload = False
        for chunk in chunks:
            payload = self.extraction_engine.build_extraction_payload(chunk)
            if not payload.user_prompt:
                continue
            built_payload = True
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": payload.user_prompt}],
                system_prompt=payload.system_prompt,
                model_config="batch",
            )
            response_text = self.llm_provider.extract_text_content(response)
            memory_ids = self._process_and_store_memories(
                user_id,
                response_text,
                payload,
                segment_id=str(chunk.segment_id) if chunk.segment_id else None,
            )
            logger.info(
                "Direct extraction chunk %d stored %d memories",
                chunk.chunk_index,
                len(memory_ids),
            )

        if not built_payload:
            raise ValueError(
                f"No valid extraction payloads built from {len(chunks)} chunks for user {user_id}"
            )
        return f"direct_{uuid4()}"

    def _process_and_store_memories(
        self,
        user_id: str,
        response_text: str,
        payload: ExtractionPayload,
        segment_id: Optional[str],
    ) -> list[UUID]:
        result = self.memory_processor.process_extraction_response(
            response_text=response_text,
            short_to_uuid=payload.short_to_uuid,
            memory_context=payload.memory_context,
            segment_id=segment_id,
            global_scope_offered=payload.global_scope_offered,
        )
        if not result.memories:
            return []

        global_memories = [m for m in result.memories if m.scope == MEMORY_SCOPE_GLOBAL]
        personal_memories = [m for m in result.memories if m.scope != MEMORY_SCOPE_GLOBAL]

        from lt_memory.global_store import write_global_memory

        stored_ids: list[UUID] = []
        for memory in global_memories:
            try:
                stored_ids.append(write_global_memory(memory, UUID(user_id), self.db))
            except GlobalMemoryStoreError as e:
                # A deterministic rejection (4xx other than 408/429) will not heal
                # by retrying: demote the memory to personal so the segment can
                # complete and the user keeps the memory. Outages (5xx, transport)
                # raise, leaving the segment unmarked for the sweep.
                if e.status_code is not None and 400 <= e.status_code < 500 and e.status_code not in (408, 429):
                    logger.warning(
                        "Global memory store rejected a memory with HTTP %s; storing it as personal: %s",
                        e.status_code, e,
                    )
                    personal_memories.append(memory)
                else:
                    raise

        if not personal_memories:
            return stored_ids
        stored_ids.extend(
            store_and_tend_extraction(
                user_id=user_id,
                segment_id=segment_id,
                memories=personal_memories,
                vector_ops=self.vector_ops,
                db=self.db,
                linking=self.linking,
            )
        )
        return stored_ids


def create_execution_strategy(
    extraction_engine: ExtractionEngine,
    memory_processor: MemoryProcessor,
    vector_ops: VectorOps,
    db: LTMemoryDB,
    llm_provider: LLMProvider,
    linking_service: LinkingService,
) -> DirectExecutionStrategy:
    """Create the single supported direct extraction strategy."""
    return DirectExecutionStrategy(
        extraction_engine,
        memory_processor,
        vector_ops,
        db,
        llm_provider,
        linking_service,
    )
