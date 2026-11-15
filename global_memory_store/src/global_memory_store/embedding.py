"""The store's document-side encoder.

CROSS-FILE-COUPLING: this is a second implementation of the document-side
encode in the MIRA instance (clients/embeddings_provider.py — the `_encode`
methods behind `encode_deep`). The two must agree on the vector for a given
text, and the instance checks that at every boot (lt_memory/global_store.py).
A change to either side — model, normalization, input handling — must be made
in both, or the boot check fails.
"""

import threading

import httpx
import numpy as np

from .config import EmbeddingConfig

_REMOTE_TIMEOUT_SECONDS = 30.0
_LOCAL_BATCH_SIZE = 32

# The local model is loaded once per process; loading is expensive.
_local_model = None
_local_model_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """The configured provider did not produce one finite unit vector per text."""


def embed_documents(texts: list[str], embedding: EmbeddingConfig, api_key: str | None) -> np.ndarray:
    """Unit-length float32 vectors, shape (len(texts), embedding.dimensions)."""
    if not texts:
        return np.empty((0, embedding.dimensions), dtype=np.float32)
    if embedding.provider == "remote":
        vectors = _embed_remote(texts, embedding, api_key)
    elif embedding.provider == "local":
        vectors = _embed_local(texts, embedding)
    else:
        raise EmbeddingError(f"unknown embedding provider {embedding.provider!r}")
    return _as_unit_vectors(vectors, embedding.dimensions, embedding.provider)


def _embed_remote(texts: list[str], embedding: EmbeddingConfig, api_key: str | None) -> np.ndarray:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            embedding.endpoint_url,
            json={"model": embedding.model, "input": texts},
            headers=headers,
            timeout=_REMOTE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        raise EmbeddingError(f"embeddings endpoint {embedding.endpoint_url} unreachable: {e}") from e
    if response.status_code >= 400:
        raise EmbeddingError(
            f"embeddings endpoint {embedding.endpoint_url} returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    try:
        body = response.json()
    except ValueError as e:
        raise EmbeddingError(
            f"embeddings endpoint {embedding.endpoint_url} returned a non-JSON body: {response.text[:300]!r}"
        ) from e

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != len(texts):
        got = len(data) if isinstance(data, list) else repr(body)[:300]
        raise EmbeddingError(
            f"embeddings endpoint {embedding.endpoint_url} must return {len(texts)} results under 'data'; got {got}"
        )

    vectors: list[list[float] | None] = [None] * len(texts)
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(texts) or vectors[index] is not None:
            raise EmbeddingError(
                f"embeddings endpoint {embedding.endpoint_url} returned a result with a missing, "
                f"out-of-range, or duplicate index: {index!r}"
            )
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError(
                f"embeddings endpoint {embedding.endpoint_url} result {index} carries no embedding list"
            )
        vectors[index] = vector

    try:
        return np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise EmbeddingError(
            f"embeddings endpoint {embedding.endpoint_url} returned vectors that are not equal-length number lists: {e}"
        ) from e


def _embed_local(texts: list[str], embedding: EmbeddingConfig) -> np.ndarray:
    global _local_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingError(
            "the local embedding provider needs sentence-transformers; "
            "install the 'local-embeddings' extra"
        ) from e
    with _local_model_lock:
        if _local_model is None:
            _local_model = SentenceTransformer(embedding.model)
        model = _local_model
    return np.asarray(model.encode_document(texts, batch_size=_LOCAL_BATCH_SIZE), dtype=np.float32)


def _as_unit_vectors(vectors: np.ndarray, dimensions: int, provider: str) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[1] != dimensions:
        raise EmbeddingError(
            f"{provider} provider produced shape {vectors.shape}; configured dimensions are {dimensions}"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingError(f"{provider} provider produced non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EmbeddingError(f"{provider} provider produced a zero vector; it has no direction to compare")
    return (vectors / norms).astype(np.float32)
