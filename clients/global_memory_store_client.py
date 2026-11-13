"""
HTTP client for the global memory store.

The store is a standalone service (see global_memory_store/ in this repo) that
holds the shared memory set every instance reads at boot and writes at
extraction time. The client is deliberately small: identity, paged list,
add. Deletes are an operator action over HTTP, not something the instance does.

Every failure mode raises GlobalMemoryStoreError — transport errors, non-2xx
answers, and bodies that do not match the contract. Callers that want the
instance to keep serving on store failure catch that one type.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional
from uuid import UUID

import httpx

from utils import http_client

logger = logging.getLogger(__name__)


class GlobalMemoryStoreError(RuntimeError):
    """The store could not be reached, answered non-2xx, or answered off-contract.

    ``status_code`` carries the HTTP status for a non-2xx answer, or None for a
    transport failure. Callers use it to tell a deterministic rejection (4xx)
    from an outage (5xx, transport): the former will not heal by retrying.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class StoreIdentity:
    model: str
    dimensions: int
    probe_text: str
    probe_vector: list[float]
    memory_count: int


@dataclass(frozen=True)
class StoredGlobalMemory:
    id: UUID
    seq: int
    text: str
    embedding: list[float]
    happens_at: Optional[datetime]
    created_at: datetime


@dataclass(frozen=True)
class AddMemoryResult:
    status: str  # "created" | "duplicate"
    memory: StoredGlobalMemory


def _parse_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise GlobalMemoryStoreError(f"store returned {field} that is not a string: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise GlobalMemoryStoreError(f"store returned {field} that is not ISO 8601: {value!r}") from e
    if parsed.tzinfo is None:
        raise GlobalMemoryStoreError(f"store returned {field} without a UTC offset: {value!r}")
    return parsed


def _parse_memory(raw: Any) -> StoredGlobalMemory:
    if not isinstance(raw, dict):
        raise GlobalMemoryStoreError(f"store returned a memory that is not an object: {raw!r}")
    try:
        memory_id = UUID(str(raw["id"]))
        seq = raw["seq"]
        text = raw["text"]
        embedding = raw["embedding"]
        created_at = _parse_datetime(raw["created_at"], "created_at")
    except KeyError as e:
        raise GlobalMemoryStoreError(f"store returned a memory missing {e}") from e
    except ValueError as e:
        raise GlobalMemoryStoreError(f"store returned a memory with a bad id: {e}") from e
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise GlobalMemoryStoreError(f"store returned a memory with a non-integer seq: {seq!r}")
    if not isinstance(text, str) or not text:
        raise GlobalMemoryStoreError("store returned a memory with empty text")
    if not isinstance(embedding, list) or not embedding or not all(isinstance(v, (int, float)) for v in embedding):
        raise GlobalMemoryStoreError("store returned a memory with a malformed embedding")
    happens_at_raw = raw.get("happens_at")
    happens_at = _parse_datetime(happens_at_raw, "happens_at") if happens_at_raw is not None else None
    return StoredGlobalMemory(
        id=memory_id,
        seq=seq,
        text=text,
        embedding=[float(v) for v in embedding],
        happens_at=happens_at,
        created_at=created_at,
    )


class GlobalMemoryStoreClient:
    """Synchronous client; one instance per process, created at boot."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0, max_retries: int = 2):
        self._base_url = base_url.rstrip("/")
        self._http = http_client.Client(
            timeout=timeout,
            max_retries=max_retries,
            headers={"Authorization": f"Bearer {token}"},
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, f"{self._base_url}{path}", **kwargs)
        except httpx.HTTPError as e:
            raise GlobalMemoryStoreError(f"{method} {path} failed: {e}") from e
        if response.status_code >= 400:
            raise GlobalMemoryStoreError(
                f"{method} {path} returned HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as e:
            raise GlobalMemoryStoreError(f"{method} {path} returned a non-JSON body") from e

    def fetch_identity(self) -> StoreIdentity:
        body = self._request("GET", "/v1/identity")
        if not isinstance(body, dict):
            raise GlobalMemoryStoreError(f"store identity is not an object: {body!r}")
        try:
            model = body["model"]
            dimensions = body["dimensions"]
            probe_text = body["probe_text"]
            probe_vector = body["probe_vector"]
            memory_count = body["memory_count"]
        except KeyError as e:
            raise GlobalMemoryStoreError(f"store identity is missing {e}") from e
        if not isinstance(model, str) or not model:
            raise GlobalMemoryStoreError(f"store identity model is not a non-empty string: {model!r}")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise GlobalMemoryStoreError(f"store identity dimensions is not a positive integer: {dimensions!r}")
        if not isinstance(probe_text, str) or not probe_text:
            raise GlobalMemoryStoreError("store identity probe_text is missing")
        if not isinstance(probe_vector, list) or len(probe_vector) != dimensions:
            raise GlobalMemoryStoreError(
                f"store identity probe_vector has {len(probe_vector) if isinstance(probe_vector, list) else 'no'} entries, expected {dimensions}"
            )
        if not all(isinstance(v, (int, float)) for v in probe_vector):
            raise GlobalMemoryStoreError("store identity probe_vector contains a non-number")
        if isinstance(memory_count, bool) or not isinstance(memory_count, int) or memory_count < 0:
            raise GlobalMemoryStoreError(f"store identity memory_count is not a non-negative integer: {memory_count!r}")
        return StoreIdentity(
            model=model,
            dimensions=dimensions,
            probe_text=probe_text,
            probe_vector=[float(v) for v in probe_vector],
            memory_count=memory_count,
        )

    def iter_memories(self, page_size: int = 500) -> Iterator[StoredGlobalMemory]:
        after = 0
        while True:
            body = self._request("GET", "/v1/memories", params={"after": after, "limit": page_size})
            if not isinstance(body, dict):
                raise GlobalMemoryStoreError(f"store memories page is not an object: {body!r}")
            memories = body.get("memories")
            next_after = body.get("next_after")
            if not isinstance(memories, list):
                raise GlobalMemoryStoreError("store memories page is missing the memories list")
            for raw in memories:
                yield _parse_memory(raw)
            if next_after is None:
                return
            if isinstance(next_after, bool) or not isinstance(next_after, int) or next_after <= after:
                raise GlobalMemoryStoreError(f"store returned a non-advancing next_after: {next_after!r}")
            after = next_after

    def add_memory(
        self,
        text: str,
        happens_at: Optional[datetime],
        contributor_user_id: str,
    ) -> AddMemoryResult:
        if happens_at is not None and happens_at.tzinfo is None:
            raise ValueError("happens_at must be timezone-aware")
        body = self._request(
            "POST",
            "/v1/memories",
            json={
                "text": text,
                "happens_at": happens_at.isoformat() if happens_at else None,
                "contributor_user_id": str(contributor_user_id),
            },
        )
        if not isinstance(body, dict):
            raise GlobalMemoryStoreError(f"store add_memory response is not an object: {body!r}")
        status = body.get("status")
        if status not in ("created", "duplicate"):
            raise GlobalMemoryStoreError(f"store add_memory returned unknown status: {status!r}")
        return AddMemoryResult(status=status, memory=_parse_memory(body.get("memory")))
