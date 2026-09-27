"""
Consolidation handler - memory consolidation with link transfer.

Handles consolidating multiple memories into one while preserving all
relationship information through link bundle transfer, source memory
outbound link rewriting, and proper archival of consolidated memories.
"""
import logging
import statistics
from typing import List, Dict, Any
from uuid import UUID

from lt_memory.models import ExtractedMemory
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class ConsolidationHandler:
    """
    Handle memory consolidation with link bundle transfer.

    Single Responsibility: Consolidate multiple memories into one,
    preserving all relationship information.

    Pure business logic - no I/O decisions or orchestration.
    """

    def __init__(self, vector_ops: VectorOps, db: LTMemoryDB):
        """
        Initialize consolidation handler.

        Args:
            vector_ops: Vector operations for memory storage
            db: Database access for memory queries and updates
        """
        self.vector_ops = vector_ops
        self.db = db

    def execute_consolidation(
        self,
        old_memory_ids: List[UUID],
        consolidated_text: str,
        user_id: str,
        merge_note: str | None = None,
    ) -> UUID:
        """
        Consolidate multiple memories into one with link preservation.

        Creates new consolidated memory with median importance from source
        memories, transfers all link bundles (inbound, outbound, entity),
        rewrites source memory references to point to new memory, and
        archives old memories.

        Args:
            old_memory_ids: Memory UUIDs to consolidate
            consolidated_text: Text of consolidated memory
            user_id: User ID
            merge_note: LLM's note on what was preserved/elided (stored as annotation)

        Returns:
            UUID of newly created consolidated memory

        Raises:
            ValueError: If old memories cannot be loaded
            RuntimeError: If consolidated memory storage fails
        """
        # Step 1: Load old memories
        old_memories = self.db.get_memories_by_ids(old_memory_ids, user_id=user_id)

        # Validate ALL requested memories were found (not just some)
        if len(old_memories) != len(old_memory_ids):
            found_ids = {m.id for m in old_memories}
            missing_ids = set(old_memory_ids) - found_ids
            raise ValueError(
                f"Failed to load {len(missing_ids)} of {len(old_memory_ids)} memories for consolidation. "
                f"Missing IDs: {missing_ids}. Cannot consolidate with incomplete memory set."
            )

        # Step 2: Calculate median importance from old memories
        importance_scores = [m.importance_score for m in old_memories if m.importance_score]
        median_importance = statistics.median(importance_scores) if importance_scores else 0.5

        logger.debug(
            f"Consolidating {len(old_memories)} memories with median importance {median_importance:.3f}"
        )

        # Step 3: Collect all unique links from old memories
        all_inbound_links = []
        all_outbound_links = []
        all_entity_links = []

        for memory in old_memories:
            all_inbound_links.extend(memory.inbound_links)
            all_outbound_links.extend(memory.outbound_links)
            all_entity_links.extend(memory.entity_links)

        # Step 4: Deduplicate links (exclude self-references, keep most recent)
        old_memory_id_strs = {str(mid) for mid in old_memory_ids}

        # Deduplicate inbound links by UUID, keeping most recent link
        unique_inbound = {}
        for link in all_inbound_links:
            uuid = link['uuid']
            if uuid in old_memory_id_strs:
                continue  # Skip self-references to old memories
            if uuid not in unique_inbound:
                unique_inbound[uuid] = link
            else:
                # Keep most recent link
                if link.get('created_at', '') > unique_inbound[uuid].get('created_at', ''):
                    unique_inbound[uuid] = link

        # Deduplicate outbound links by UUID, keeping most recent link
        unique_outbound = {}
        for link in all_outbound_links:
            uuid = link['uuid']
            if uuid in old_memory_id_strs:
                continue
            if uuid not in unique_outbound:
                unique_outbound[uuid] = link
            else:
                # Keep most recent link
                if link.get('created_at', '') > unique_outbound[uuid].get('created_at', ''):
                    unique_outbound[uuid] = link

        # Deduplicate entity links by UUID
        unique_entities = {
            link['uuid']: link for link in all_entity_links
        }

        logger.debug(
            f"Collected link bundles: {len(unique_inbound)} inbound, "
            f"{len(unique_outbound)} outbound, {len(unique_entities)} entities"
        )

        # Step 5: Preserve segment provenance from source memories
        # Pick earliest source segment for primary tracing column;
        # full set is recorded in the consolidation annotation.
        source_segment_ids = [
            m.source_segment_id for m in old_memories
            if m.source_segment_id is not None
        ]
        # Earliest by creation time — old_memories are sorted by importance_score DESC
        # from get_memories_by_ids, so find the earliest created_at explicitly
        earliest_segment_id = None
        if source_segment_ids:
            segment_by_created = sorted(
                [(m.created_at, m.source_segment_id) for m in old_memories if m.source_segment_id],
                key=lambda x: x[0]
            )
            earliest_segment_id = segment_by_created[0][1]

        # Step 5b: Create consolidated memory with median importance
        consolidated_memory = ExtractedMemory(
            text=consolidated_text,
            importance_score=median_importance,
            source_segment_id=earliest_segment_id,
        )

        # Step 6: Store consolidated memory with embeddings
        new_ids = self.vector_ops.store_memories_with_embeddings(
            [consolidated_memory]
        )

        if not new_ids:
            raise RuntimeError("Failed to store consolidated memory")

        new_memory_id = new_ids[0]
        new_memory_id_str = str(new_memory_id)

        # Step 7: Transfer link bundles to new consolidated memory
        if unique_inbound or unique_outbound or unique_entities:
            self.db.update_memory(new_memory_id, {
                'inbound_links': list(unique_inbound.values()),
                'outbound_links': list(unique_outbound.values()),
                'entity_links': list(unique_entities.values())
            }, user_id=user_id)

            logger.info(
                f"Transferred links to consolidated memory {new_memory_id}: "
                f"{len(unique_inbound)} inbound, {len(unique_outbound)} outbound, "
                f"{len(unique_entities)} entities"
            )

        # Step 7b: Carry access stats from merged memories (min activity-days
        # is the most conservative decay position)
        last_accessed = max(
            (m.last_accessed for m in old_memories if m.last_accessed),
            default=None
        )
        activity_days = min(
            (m.activity_days_at_last_access for m in old_memories
             if m.activity_days_at_last_access is not None),
            default=None
        )
        self.db.update_memory(new_memory_id, {
            'access_count': sum(m.access_count for m in old_memories),
            'mention_count': sum(m.mention_count for m in old_memories),
            'last_accessed': last_accessed,
            'activity_days_at_last_access': activity_days,
        }, user_id=user_id)

        # Step 7c: Store merge note as annotation (includes segment provenance)
        if merge_note:
            source_uuids = [str(mid) for mid in old_memory_ids]
            annotation: Dict[str, Any] = {
                'text': merge_note,
                'created_at': utc_now().isoformat(),
                'source': 'consolidation',
                'archived_source_ids': source_uuids,
            }
            # Preserve all source segment IDs for full lineage tracing
            if source_segment_ids:
                annotation['source_segment_ids'] = list({str(sid) for sid in source_segment_ids})
            self.db.update_memory(new_memory_id, {
                'annotations': [annotation]
            }, user_id=user_id)

        # Step 8: Rewrite links in affected memories to point at the new
        # memory. Outbound side: memories that linked TO old memories.
        # Inbound side: targets of the old memories' outbound links still
        # record the old (soon-archived) ids in their inbound_links.
        affected_ids = {
            UUID(link['uuid']) for link in all_inbound_links
            if link['uuid'] not in old_memory_id_strs
        }
        affected_ids.update(
            UUID(link['uuid']) for link in unique_outbound.values()
        )

        for affected_id in affected_ids:
            affected = self.db.get_memory(affected_id, user_id=user_id)
            if not affected:
                continue

            updates = {}

            rewritten_outbound = []
            for link in affected.outbound_links:
                if link['uuid'] in old_memory_id_strs:
                    updated_link = link.copy()
                    updated_link['uuid'] = new_memory_id_str
                    rewritten_outbound.append(updated_link)
                else:
                    rewritten_outbound.append(link)
            if rewritten_outbound != affected.outbound_links:
                # Links to multiple consolidated sources collapse onto the
                # single new target; write the (memory, link) pair once.
                # Keep the most recent link, mirroring the Step 4 dedupe.
                deduped_outbound = []
                for link in rewritten_outbound:
                    if link['uuid'] == new_memory_id_str:
                        existing = next(
                            (l for l in deduped_outbound if l['uuid'] == new_memory_id_str),
                            None
                        )
                        if existing is None:
                            deduped_outbound.append(link)
                        elif link.get('created_at', '') > existing.get('created_at', ''):
                            deduped_outbound[deduped_outbound.index(existing)] = link
                    else:
                        deduped_outbound.append(link)
                updates['outbound_links'] = deduped_outbound

            rewritten_inbound = []
            for link in affected.inbound_links:
                if link['uuid'] in old_memory_id_strs:
                    updated_link = link.copy()
                    updated_link['uuid'] = new_memory_id_str
                    rewritten_inbound.append(updated_link)
                else:
                    rewritten_inbound.append(link)
            if rewritten_inbound != affected.inbound_links:
                # Same dedupe as the outbound side: two old memories both
                # linking here collapse to one inbound entry for the target.
                deduped_inbound = []
                for link in rewritten_inbound:
                    if link['uuid'] == new_memory_id_str:
                        existing = next(
                            (l for l in deduped_inbound if l['uuid'] == new_memory_id_str),
                            None
                        )
                        if existing is None:
                            deduped_inbound.append(link)
                        elif link.get('created_at', '') > existing.get('created_at', ''):
                            deduped_inbound[deduped_inbound.index(existing)] = link
                    else:
                        deduped_inbound.append(link)
                updates['inbound_links'] = deduped_inbound

            if updates:
                self.db.update_memory(affected_id, updates, user_id=user_id)

        if affected_ids:
            logger.debug(f"Rewrote links for {len(affected_ids)} affected memories")

        # Step 9: Archive the old memories
        for old_id in old_memory_ids:
            self.db.archive_memory(old_id, user_id=user_id)

        logger.info(
            f"Consolidated {len(old_memory_ids)} memories into {new_memory_id} "
            f"(median importance: {median_importance:.3f}): {consolidated_text[:80]}..."
        )

        return new_memory_id
