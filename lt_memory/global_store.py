"""
Boot-time coordination with the global memory store.

The store is a standalone service holding the shared memory set. At boot the
instance checks that the store embeds the same way this instance does, then
replaces its local `global_memories` copy with the store's contents. Turn-time
retrieval never touches this module; it reads the local copy through the
existing UNION queries.

State is decided once at boot and held for the life of the process:

- unconfigured — `global_memory_store_url` is empty; nothing happens.
- active — the check and the pull both succeeded; writes go to the store.
- inactive — configured but the token is missing, the store is unreachable,
  or the check failed; the local copy keeps its last contents and nothing new
  is shared until the next restart.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np

from clients.global_memory_store_client import (
    GlobalMemoryStoreClient,
    GlobalMemoryStoreError,
    StoreIdentity,
)
from clients.vault_client import get_api_key
from config import config
from lt_memory.models import ExtractedMemory, GlobalMemoryRow

logger = logging.getLogger(__name__)

# Measured 2026-11-13: cosine between this instance's encode_deep and the
# store's embed_documents is 0.99999998 for both the remote provider
# (emb-qwen3) and the local provider (mdbr-leaf-ir-asym); the residual is
# float16-vs-float32 rounding. The floor sits far below that and far above
# any cross-model agreement.
PROBE_AGREEMENT_FLOOR = 0.99


class GlobalMemoryStoreMismatch(RuntimeError):
    """The store and this instance do not share an embedding space."""


@dataclass(frozen=True)
class GlobalStoreBootResult:
    configured: bool
    active: bool
    memories_loaded: int


_client: Optional[GlobalMemoryStoreClient] = None
_active = False


def global_store_is_active() -> bool:
    return _active


def check_store_compatibility(client: GlobalMemoryStoreClient, embeddings_provider) -> tuple[StoreIdentity, float]:
    """Verify the store embeds the same way this instance does.

    Raises GlobalMemoryStoreMismatch naming both models, both dimension
    counts, and the measured cosine when they do not. Returns the store's
    identity and the measured probe cosine on success.
    """
    from clients.embeddings_provider import load_embedding_config

    instance_model = load_embedding_config().model
    identity = client.fetch_identity()

    if identity.dimensions != embeddings_provider.dimensions:
        raise GlobalMemoryStoreMismatch(
            f"global memory store reports {identity.dimensions} dimensions (model {identity.model!r}); "
            f"this instance embeds with {instance_model!r} at {embeddings_provider.dimensions} dimensions"
        )

    probe = embeddings_provider.encode_deep([identity.probe_text])
    instance_vec = np.asarray(probe[0], dtype=np.float64)
    store_vec = np.asarray(identity.probe_vector, dtype=np.float64)
    denom = np.linalg.norm(instance_vec) * np.linalg.norm(store_vec)
    cosine = float(np.dot(instance_vec, store_vec) / denom) if denom else 0.0
    if cosine < PROBE_AGREEMENT_FLOOR:
        raise GlobalMemoryStoreMismatch(
            f"global memory store probe disagreement: cosine {cosine:.6f} is below the "
            f"{PROBE_AGREEMENT_FLOOR} floor (instance model {instance_model!r}, "
            f"store model {identity.model!r})"
        )
    return identity, cosine


def replace_global_memories_from_store(db, embeddings_provider) -> GlobalStoreBootResult:
    """Pull the store's full memory set into the local copy.

    Raises GlobalMemoryStoreError, GlobalMemoryStoreMismatch, or the Vault
    error for a missing token; the caller decides how to degrade. On any
    failure the local copy is left untouched.
    """
    global _client, _active

    url = config.system.global_memory_store_url.strip()
    if not url:
        _active = False
        return GlobalStoreBootResult(configured=False, active=False, memories_loaded=0)

    token = get_api_key("global_memory_store")
    client = GlobalMemoryStoreClient(url, token)
    try:
        identity, _cosine = check_store_compatibility(client, embeddings_provider)
        rows = list(client.iter_memories())
        for row in rows:
            if len(row.embedding) != identity.dimensions:
                raise GlobalMemoryStoreError(
                    f"store memory {row.id} carries {len(row.embedding)} dimensions, expected {identity.dimensions}"
                )
        local_rows = [
            GlobalMemoryRow(
                id=row.id,
                text=row.text,
                embedding=row.embedding,
                happens_at=row.happens_at,
                created_at=row.created_at,
            )
            for row in rows
        ]
        loaded = db.replace_global_memories(local_rows)
    except BaseException:
        client.close()
        _client = None
        _active = False
        raise

    _client = client
    _active = True
    logger.info(
        "Global memory store active: %d memories loaded (store model %s, %d dimensions)",
        loaded, identity.model, identity.dimensions,
    )
    return GlobalStoreBootResult(configured=True, active=True, memories_loaded=loaded)


def write_global_memory(memory: ExtractedMemory, contributor_user_id: UUID, db) -> UUID:
    """Write one extracted global memory to the store and the local copy."""
    if _client is None or not _active:
        raise GlobalMemoryStoreError("global memory store is not active in this process")
    result = _client.add_memory(
        text=memory.text,
        happens_at=memory.happens_at,
        contributor_user_id=str(contributor_user_id),
    )
    stored = result.memory
    db.insert_global_memory(
        GlobalMemoryRow(
            id=stored.id,
            text=stored.text,
            embedding=stored.embedding,
            happens_at=stored.happens_at,
            created_at=stored.created_at,
        )
    )
    return stored.id


def close_global_store() -> None:
    global _client, _active
    if _client is not None:
        _client.close()
    _client = None
    _active = False
