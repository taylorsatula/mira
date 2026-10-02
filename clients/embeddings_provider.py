"""
Embeddings provider: the one path from text to vectors.

Two implementations behind `EmbeddingsProvider`:

- `LocalEmbeddingsProvider`: MongoDB/mdbr-leaf-ir-asym through
  sentence-transformers, 768 dimensions, asymmetric — `encode_realtime()` runs
  the query encoder (mdbr-leaf-ir), `encode_deep()` the document encoder
  (snowflake-arctic-embed-m-v1.5).
- `RemoteEmbeddingsProvider`: any OpenAI-compatible `POST /v1/embeddings`
  endpoint. One model serves both roles.

Which implementation runs, and at what dimensionality, is fixed per install by
the one-row `embedding_config` table (deploy/mira_service_schema.sql). The
installer fills the row by asking this module (`describe_for_installer`, via
deploy/lib/embedding_config.sh), which probes a remote endpoint for its vector
length; the same number sizes every vector column. Vectors from different models are not comparable, so the
schema refuses to change the row once any vector is stored.

Output contract, both implementations: float16 ndarrays, unit length; shape
(dimensions,) for a str input, (n, dimensions) for a list.
"""
import hashlib
import logging
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np

from config import config
from utils import http_client

logger = logging.getLogger(__name__)

LOCAL_EMBEDDING_MODEL = "MongoDB/mdbr-leaf-ir-asym"
LOCAL_EMBEDDING_DIMENSIONS = 768

# pgvector's HNSW and IVFFlat indexes accept at most 2,000 dimensions for the
# vector type, and every embedding column carries one of those indexes.
MAX_EMBEDDING_DIMENSIONS = 2000

# Remote request ceilings for a hung or failing endpoint — bounds, not measured
# latencies. utils/http_client retries 429/5xx/connect errors and caps each
# retry sleep at 30 s.
_REMOTE_TIMEOUT_SECONDS = 30
_REMOTE_MAX_RETRIES = 2
_HTTP_MAX_RETRY_SLEEP_SECONDS = 30

# Ceiling for one CPU-bound local encode, so a wedged encoder fails its caller
# instead of hanging it.
_LOCAL_ENCODE_BOUND_SECONDS = 60

_PROBE_TEXT = "MIRA embedding dimension probe"

EmbeddingRole = Literal["query", "document"]


class EmbeddingOutputError(RuntimeError):
    """A provider produced something other than the vectors that were asked for."""


class EmbeddingCache:
    """
    Valkey-backed embedding cache with 15-minute TTL.

    Raises if Valkey is unreachable - embedding cache requires Valkey.
    """

    def __init__(self, key_prefix: str = "embedding"):
        self.logger = logging.getLogger("embedding_cache")
        self.key_prefix = key_prefix
        from clients.valkey_client import get_valkey_client
        self.valkey = get_valkey_client()  # Raises if Valkey unreachable
        self.logger.debug(f"Embedding cache initialized with Valkey backend (prefix: {key_prefix})")

    def _get_cache_key(self, text: str) -> str:
        return f"{self.key_prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.

        Returns None if key not found (cache miss).
        Raises if Valkey operation fails.
        """
        cache_key = self._get_cache_key(text)
        cached_data = self.valkey.valkey_binary.get(cache_key)
        if cached_data:
            return np.frombuffer(cached_data, dtype=np.float16)

        return None  # Cache miss

    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache embedding with 15-minute TTL.

        Raises if Valkey operation fails.
        """
        cache_key = self._get_cache_key(text)
        embedding_bytes = embedding.astype(np.float16).tobytes()
        self.valkey.valkey_binary.setex(cache_key, 900, embedding_bytes)


@dataclass(frozen=True)
class EmbeddingConfig:
    """The install's `embedding_config` row."""

    provider: Literal["local", "remote"]
    model: str
    endpoint_url: str | None
    api_key_name: str | None
    dimensions: int


def load_embedding_config() -> EmbeddingConfig:
    """Read the one-row `embedding_config` table; raises unless exactly one row."""
    from clients.postgres_client import PostgresClient

    rows = PostgresClient("mira_service").execute_query(
        "SELECT provider, model, endpoint_url, api_key_name, dimensions FROM embedding_config"
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"embedding_config must hold exactly one row, found {len(rows)}. "
            "The installer seeds it when it applies deploy/mira_service_schema.sql."
        )
    row = rows[0]
    return EmbeddingConfig(
        provider=row["provider"],
        model=row["model"],
        endpoint_url=row["endpoint_url"],
        api_key_name=row["api_key_name"],
        dimensions=row["dimensions"],
    )


class EmbeddingsProvider(ABC):
    """Text to vectors at the install's fixed dimensionality.

    Single-text encodes are cached in Valkey per role. Subclasses implement
    `_encode` over a non-empty list; shape is enforced here, so a provider can
    never hand a caller (or a vector column) the wrong dimensionality.
    """

    def __init__(self, dimensions: int, cache_enabled: bool):
        self.dimensions = dimensions
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger("embeddings")
        if cache_enabled:
            self.query_cache = EmbeddingCache(key_prefix=f"embedding_{dimensions}_query")
            self.doc_cache = EmbeddingCache(key_prefix=f"embedding_{dimensions}_doc")
        else:
            self.query_cache = None
            self.doc_cache = None

    @property
    @abstractmethod
    def max_encode_seconds(self) -> float:
        """Longest one encode of up to `embeddings_batch_size` texts can legitimately take."""

    @abstractmethod
    def _encode(self, texts: List[str], role: EmbeddingRole) -> np.ndarray:
        """(len(texts), dimensions) unit-length vectors for a non-empty list."""

    def encode_realtime(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Query encoding, for retrieval queries."""
        return self._encode_cached(texts, "query", self.query_cache)

    def encode_deep(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Document encoding, for memories and segment summaries."""
        return self._encode_cached(texts, "document", self.doc_cache)

    def _encode_cached(
        self,
        texts: Union[str, List[str]],
        role: EmbeddingRole,
        cache: Optional[EmbeddingCache],
    ) -> np.ndarray:
        single = isinstance(texts, str)
        if single and cache is not None:
            cached = cache.get(texts)
            if cached is not None:
                self.logger.debug(f"{role} encode: cache hit (text_len={len(texts)})")
                return cached

        batch = [texts] if single else list(texts)
        if not batch:
            return np.empty((0, self.dimensions), dtype=np.float16)

        vectors = self._encode(batch, role).astype(np.float16)
        if vectors.shape != (len(batch), self.dimensions):
            raise EmbeddingOutputError(
                f"{type(self).__name__} produced shape {vectors.shape} for {len(batch)} texts; "
                f"this install's embedding_config fixes {self.dimensions} dimensions"
            )

        result = vectors[0] if single else vectors
        if single and cache is not None:
            cache.set(texts, result)
        return result


# Serializes every call into a local model. PyTorch's MPS backend is NOT
# thread-safe, and sentence-transformers auto-selects it on Apple Silicon
# (`model.device == mps:0`, verified live on macOS 15 / torch 2.x). Two
# concurrent encodes — exactly the two-thread path in
# cns/services/orchestrator.py:_compute_embeddings_parallel — race in the
# Metal shader-kernel cache, observed live both as a hang past the 60 s encode
# bound and as a SIGSEGV inside MetalShaderLibrary::exec_unary_kernel.
# Module-level rather than per-instance: every provider in this process drives
# the same device and must serialize against the same lock. Single-threaded
# MPS is correct and fast (8 sequential encodes in 0.22 s measured), so the
# serialization costs nothing measurable.
_LOCAL_INFERENCE_LOCK = threading.Lock()


class LocalEmbeddingsProvider(EmbeddingsProvider):
    """mdbr-leaf-ir-asym through sentence-transformers (asymmetric, 768 dimensions)."""

    def __init__(self, cache_enabled: bool = True):
        super().__init__(LOCAL_EMBEDDING_DIMENSIONS, cache_enabled)

        from sentence_transformers import SentenceTransformer

        # LOCAL MODEL INVARIANT: the query/document pairing is contract —
        # encode_query() for retrieval queries, encode_document() for stored
        # text. An install's stored vectors belong to the model named in its
        # embedding_config row; changing LOCAL_EMBEDDING_MODEL makes every
        # existing local install refuse to boot (build_embeddings_provider)
        # until its vectors are regenerated.
        # Offline-first load: when the model is already in the local HF cache
        # this skips the Hub availability check (~2s network per process load,
        # paid by both the POST gate child and the serving process). A cache miss
        # falls back to the online load so fresh installs still download.
        try:
            self.model = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                cache_folder=None,  # Uses default HuggingFace cache directory
                local_files_only=True,
            )
        except Exception:
            self.model = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                cache_folder=None,
            )

        self.logger.toast("LocalEmbeddingsProvider initialized")

    @property
    def max_encode_seconds(self) -> float:
        return _LOCAL_ENCODE_BOUND_SECONDS

    def _encode(self, texts: List[str], role: EmbeddingRole) -> np.ndarray:
        batch_size = config.lt_memory.embeddings_batch_size
        # Serialized: the MPS backend is not thread-safe (_LOCAL_INFERENCE_LOCK).
        with _LOCAL_INFERENCE_LOCK:
            if role == "query":
                return self.model.encode_query(texts, batch_size=batch_size)
            return self.model.encode_document(texts, batch_size=batch_size)


def request_embeddings(
    http: http_client.Client,
    endpoint_url: str,
    model: str,
    headers: dict[str, str],
    texts: List[str],
) -> np.ndarray:
    """One `POST /v1/embeddings` call: float32 array (len(texts), d) in input order.

    Each result is placed by its `index` field, never by list position. Raises
    HTTPStatusError on a non-2xx status (after http_client retries), and
    EmbeddingOutputError when the body is not exactly one finite vector per
    input, all of one length.
    """
    response = http.post(endpoint_url, json={"model": model, "input": texts}, headers=headers)
    if response.status_code >= 400:
        raise http_client.HTTPStatusError(
            f"Embeddings endpoint {endpoint_url} returned HTTP {response.status_code}: "
            f"{response.text[:300]}",
            request=response.request,
            response=response,
        )

    try:
        body = response.json()
    except ValueError as e:
        raise EmbeddingOutputError(
            f"Embeddings endpoint {endpoint_url} returned a non-JSON body: {response.text[:300]!r}"
        ) from e

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != len(texts):
        raise EmbeddingOutputError(
            f"Embeddings endpoint {endpoint_url} must return {len(texts)} results under 'data'; "
            f"got {len(data) if isinstance(data, list) else repr(body)[:300]}"
        )

    vectors: list[list[float] | None] = [None] * len(texts)
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(texts) or vectors[index] is not None:
            raise EmbeddingOutputError(
                f"Embeddings endpoint {endpoint_url} returned a result with a missing, "
                f"out-of-range, or duplicate index: {index!r}"
            )
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingOutputError(
                f"Embeddings endpoint {endpoint_url} result {index} carries no embedding list"
            )
        vectors[index] = vector

    try:
        array = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise EmbeddingOutputError(
            f"Embeddings endpoint {endpoint_url} returned vectors that are not equal-length number lists: {e}"
        ) from e
    if array.ndim != 2 or not np.isfinite(array).all():
        raise EmbeddingOutputError(
            f"Embeddings endpoint {endpoint_url} returned non-finite or malformed vectors (shape {array.shape})"
        )
    return array


class RemoteEmbeddingsProvider(EmbeddingsProvider):
    """Any OpenAI-compatible `POST /v1/embeddings` endpoint; one model for both roles."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: str | None,
        dimensions: int,
        cache_enabled: bool = True,
    ):
        super().__init__(dimensions, cache_enabled)
        self.endpoint_url = endpoint_url
        self.model_name = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client.Client(timeout=_REMOTE_TIMEOUT_SECONDS, max_retries=_REMOTE_MAX_RETRIES)
        self.logger.toast(f"RemoteEmbeddingsProvider initialized ({model} at {endpoint_url}, {dimensions}d)")

    @property
    def max_encode_seconds(self) -> float:
        # Every attempt times out and every retry sleeps the maximum.
        return (
            (_REMOTE_MAX_RETRIES + 1) * _REMOTE_TIMEOUT_SECONDS
            + _REMOTE_MAX_RETRIES * _HTTP_MAX_RETRY_SLEEP_SECONDS
        )

    def _encode(self, texts: List[str], role: EmbeddingRole) -> np.ndarray:
        batch_size = config.lt_memory.embeddings_batch_size
        vectors = np.concatenate([
            request_embeddings(self._http, self.endpoint_url, self.model_name, self._headers, texts[start:start + batch_size])
            for start in range(0, len(texts), batch_size)
        ])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise EmbeddingOutputError(
                f"Embeddings endpoint {self.endpoint_url} returned a zero vector; it has no direction to compare"
            )
        return vectors / norms


def build_embeddings_provider(embedding_config: EmbeddingConfig, cache_enabled: bool = True) -> EmbeddingsProvider:
    """The provider the install's `embedding_config` row names; construction failures propagate."""
    if embedding_config.provider == "local":
        if (
            embedding_config.model != LOCAL_EMBEDDING_MODEL
            or embedding_config.dimensions != LOCAL_EMBEDDING_DIMENSIONS
        ):
            raise RuntimeError(
                f"embedding_config names local model {embedding_config.model!r} at "
                f"{embedding_config.dimensions} dimensions, but this build's local provider is "
                f"{LOCAL_EMBEDDING_MODEL!r} at {LOCAL_EMBEDDING_DIMENSIONS}. Stored vectors belong "
                "to the model in the row; refusing to embed with a different one."
            )
        return LocalEmbeddingsProvider(cache_enabled=cache_enabled)

    if embedding_config.provider == "remote":
        if not embedding_config.endpoint_url:
            raise RuntimeError("embedding_config provider 'remote' has no endpoint_url")
        from clients.vault_client import get_api_key

        api_key = get_api_key(embedding_config.api_key_name) if embedding_config.api_key_name else None
        return RemoteEmbeddingsProvider(
            endpoint_url=embedding_config.endpoint_url,
            model=embedding_config.model,
            api_key=api_key,
            dimensions=embedding_config.dimensions,
            cache_enabled=cache_enabled,
        )

    raise RuntimeError(f"Unknown embedding_config provider {embedding_config.provider!r}")


_provider: EmbeddingsProvider | None = None
_provider_lock = threading.Lock()


def get_embeddings_provider(cache_enabled: bool = True) -> EmbeddingsProvider:
    """Process-wide provider built from the `embedding_config` row; failures propagate."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = build_embeddings_provider(load_embedding_config(), cache_enabled=cache_enabled)
    return _provider


def describe_for_installer(argv: List[str]) -> str:
    """Installer entry point: '<model> <dimensions>' for the chosen provider.

    `describe local` needs nothing. `describe remote <endpoint_url> <model>`
    reads the endpoint's bearer token from stdin (empty for an endpoint that
    takes none) and embeds one probe text to learn the vector length. Caller:
    deploy/lib/embedding_config.sh.
    """
    if argv == ["describe", "local"]:
        return f"{LOCAL_EMBEDDING_MODEL} {LOCAL_EMBEDDING_DIMENSIONS}"

    if len(argv) == 4 and argv[:2] == ["describe", "remote"]:
        endpoint_url, model = argv[2], argv[3]
        if not endpoint_url or not model:
            raise SystemExit("describe remote needs a non-empty endpoint URL and model name")
        api_key = sys.stdin.read().strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with http_client.Client(timeout=_REMOTE_TIMEOUT_SECONDS, max_retries=_REMOTE_MAX_RETRIES) as http:
            vectors = request_embeddings(http, endpoint_url, model, headers, [_PROBE_TEXT])
        dimensions = int(vectors.shape[1])
        if dimensions > MAX_EMBEDDING_DIMENSIONS:
            raise SystemExit(
                f"{model} at {endpoint_url} returns {dimensions}-dimension vectors; pgvector indexes "
                f"accept at most {MAX_EMBEDDING_DIMENSIONS}. Choose a model with smaller vectors."
            )
        return f"{model} {dimensions}"

    raise SystemExit(
        "usage: describe local | describe remote <endpoint_url> <model>  (bearer token on stdin)"
    )
