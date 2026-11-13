"""HTTP API for the store. Four routes, one bearer token."""

import hmac
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from . import __version__
from .config import Config
from .embedding import EmbeddingError, embed_documents
from .storage import MAX_TEXT_CHARS, Store, StoreError

PROBE_TEXT = "MIRA global memory store embedding probe"


class MemoryIn(BaseModel):
    text: str
    happens_at: datetime | None = None
    contributor_user_id: uuid.UUID

    @field_validator("text")
    @classmethod
    def _text_ok(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        if len(stripped) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
        return stripped

    @field_validator("happens_at")
    @classmethod
    def _happens_at_has_offset(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("happens_at must carry a UTC offset")
        return value


def create_app(config: Config, store: Store, api_key: str | None, token: str) -> FastAPI:
    app = FastAPI(title="global-memory-store", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.probe_vector = None

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        if not hmac.compare_digest(header, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="unauthorized")

    def probe_vector() -> list[float]:
        if app.state.probe_vector is None:
            app.state.probe_vector = embed_documents([PROBE_TEXT], config.embedding, api_key)[0].tolist()
        return app.state.probe_vector

    @app.get("/v1/identity", dependencies=[Depends(require_token)])
    def identity() -> dict:
        try:
            vector = probe_vector()
        except EmbeddingError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        model, dimensions, memory_count = store.identity()
        return {
            "model": model,
            "dimensions": dimensions,
            "probe_text": PROBE_TEXT,
            "probe_vector": vector,
            "memory_count": memory_count,
        }

    @app.get("/v1/memories", dependencies=[Depends(require_token)])
    def list_memories(
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> dict:
        memories, next_after = store.list_memories(after, limit)
        return {"memories": memories, "next_after": next_after}

    @app.post("/v1/memories", dependencies=[Depends(require_token)])
    def create_memory(body: MemoryIn, response: Response) -> dict:
        try:
            embedding = embed_documents([body.text], config.embedding, api_key)[0]
        except EmbeddingError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        status, memory = store.insert_memory(
            body.text, embedding, body.happens_at, str(body.contributor_user_id)
        )
        response.status_code = 201 if status == "created" else 200
        return {"status": status, "memory": memory}

    @app.delete("/v1/memories/{memory_id}", status_code=204, dependencies=[Depends(require_token)])
    def delete_memory(memory_id: uuid.UUID) -> None:
        if not store.delete_memory(str(memory_id)):
            raise HTTPException(status_code=404, detail="memory not found")

    @app.exception_handler(StoreError)
    def store_error(_request: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
