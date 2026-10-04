"""
Chat API endpoint - simple request/response over HTTP.

Provides a non-streaming JSON API to send a user message and receive
the assistant's response plus structured metadata. Authenticated via
Bearer token (header) or session cookie.
"""
import base64
import logging
from typing import Any

from cns.core.message import ContentBlock

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from config.config_manager import config as app_config
from cns.services.async_work_barrier import get_async_work_barrier
from utils.distributed_lock import UserRequestLock

from utils.document_processing import process_document, ProcessedDocument, SUPPORTED_DOCUMENT_FORMATS, MAX_DOCUMENT_SIZE_MB
from utils.image_compression import compress_image, CompressedImage
from utils.text_sanitizer import sanitize_message_content
from utils.timezone_utils import utc_now, format_utc_iso

from .base import PropagatingHandler, SuccessResponse, ValidationError, create_success_response
from cns.services.orchestrator import get_orchestrator, MAX_LOCAL_TOOL_CALLS_PER_TURN
from cns.infrastructure.continuum_pool import get_continuum_pool


logger = logging.getLogger(__name__)

router = APIRouter()


# Image validation constants (keep consistent with websocket implementation)
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB = 5

# Text message size limit - prevents context overflow in summarization
#
# Deliberately tighter than the WebSocket gate (MAX_CONTENT_LENGTH = 100_000 in
# cns/api/websocket_chat.py). A WS turn streams content continuously, so it can
# afford a more permissive bound; this HTTP endpoint must hold the connection
# open for the full synchronous request, so the tighter limit protects request
# duration, not model input. Do not "fix" the inconsistency by aligning them.
MAX_TEXT_MESSAGE_LENGTH = 20000

# Structural worst-case bound on one HTTP chat turn: the tool loop admits up
# to MAX_LOCAL_TOOL_CALLS_PER_TURN model steps, each bounded by the provider
# HTTP/stall timeouts, plus a final generation step. The lock TTL must exceed
# the worst legal turn plus margin — approximating it downward lets a long
# turn outlive the lock and admit a concurrent second turn on the same
# segment. Background renewal (start_renewal, cadence TTL/3) keeps a live
# turn's TTL refreshed, so the TTL is a crash backstop, not a turn-length
# estimate.
_HTTP_TURN_LOCK_TTL_SECONDS = (
    (MAX_LOCAL_TOOL_CALLS_PER_TURN + 1)
    * max(app_config.api.timeout, app_config.api.provider_response_timeout)
    * 2
)

# Distributed per-user request lock (coordinates across workers)
_user_request_lock = UserRequestLock(ttl=_HTTP_TURN_LOCK_TTL_SECONDS)

# The lock-contention rejection message. cns/api/mcp.py imports this constant
# to key its busy-mapping row — change it here and nowhere else.
_BUSY_REJECTION_MESSAGE = "Another chat request is already in progress for this user"


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., description="User message text")
    image: str | None = Field(None, description="Optional image as base64 string (no data: prefix)")
    image_type: str | None = Field(None, description="MIME type for image (e.g., image/jpeg)")
    document: str | None = Field(None, description="Optional document as base64 string")
    document_type: str | None = Field(None, description="MIME type for document (e.g., application/pdf)")
    include_thinking: bool = Field(False, description="Include thinking trace in response")
    show_cost: bool = Field(False, description="Include per-request token usage and USD cost in the response under `data.cost`")

    @model_validator(mode="after")
    def validate_attachment_exclusion(self) -> "ChatRequest":
        """One message carries an image or a document, never both — the same
        contract MessageFrame enforces on the WS transport
        (websocket_chat.py:MessageFrame.validate_attachment_pairs); keep the
        wording identical in both. Before this validator, the HTTP endpoint
        silently dropped the document when both arrived."""
        if self.image is not None and self.document is not None:
            raise ValueError("one message may contain an image or a document, not both")
        return self


class ChatEndpoint(PropagatingHandler):
    """Handler for HTTP chat requests (non-streaming)."""

    def process_request(
        self,
        *,
        user_id: str,
        message: str,
        image: str | None,
        image_type: str | None,
        document: str | None,
        document_type: str | None,
        include_thinking: bool = False,
        show_cost: bool = False,
    ) -> SuccessResponse:
        start_time = utc_now()

        # Set user context for RLS and utility functions
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)

        # Basic validation
        msg = (message or "").strip()
        if not msg:
            raise ValidationError("Message cannot be empty")

        # Sanitize text
        msg = sanitize_message_content(msg)

        # Concurrency control: one active request per user; the oversized
        # rejection persists a message, so it must be ordered like a real turn.
        lock_token = _user_request_lock.acquire(user_id)
        if lock_token is None:
            # Use a validation error to preserve consistent error envelope.
            # The message is the anchor cns/api/mcp.py keys its busy-mapping
            # row on (imported constant — no transcription).
            raise ValidationError(_BUSY_REJECTION_MESSAGE)

        # Background renewal: one renewal every TTL/3 — safely inside the
        # TTL — so a turn of any legal length cannot outlive its own lock
        # and admit a concurrent second turn on the same segment.
        renewal_stop = _user_request_lock.start_renewal(user_id, lock_token)

        try:
            # Check message length - reject oversized messages with friendly assistant response
            if len(msg) > MAX_TEXT_MESSAGE_LENGTH:
                rejection_msg = (
                    f"I can't process messages longer than {MAX_TEXT_MESSAGE_LENGTH:,} characters. "
                    f"Your message was {len(msg):,} characters. "
                    f"Please break it into smaller chunks or summarize the key points you'd like to discuss."
                )

                continuum_pool = get_continuum_pool()
                continuum = continuum_pool.get_or_create()

                # Add the rejection as an assistant message.
                # add_assistant_message is cache-only (same contract the
                # orchestrator relies on): the row is durable only if staged
                # into the UnitOfWork before commit — the same staging idiom as
                # orchestrator.process_message's unit_of_work.add_messages().
                # The user's oversized message is NOT staged: it was rejected,
                # not accepted (same convention as the empty-message and
                # bad-image rejections — a rejected user message persists
                # nothing), and storing the 20k+ text would feed it right
                # back into the history/summarization the limit protects.
                rejection, _ = continuum.add_assistant_message(
                    rejection_msg, {"type": "size_limit_rejection"}
                )
                unit_of_work = continuum_pool.begin_work(continuum)
                unit_of_work.add_messages(rejection)
                unit_of_work.commit()

                return create_success_response(
                    data={"response": rejection_msg, "rejected": True},
                    meta={"timestamp": utc_now().isoformat()}
                )

            # Validate and compress image if provided
            compressed: CompressedImage | None = None
            if image:
                if not image_type:
                    raise ValidationError("image_type is required when image is provided")
                if image_type not in SUPPORTED_IMAGE_FORMATS:
                    raise ValidationError(
                        f"Unsupported image format. Supported: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
                    )
                try:
                    decoded = base64.b64decode(image, validate=True)
                    if len(decoded) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                        raise ValidationError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB")

                    # Compress to both tiers: inference (1200px) and storage (512px WebP)
                    compressed = compress_image(decoded, image_type)

                except ValidationError:
                    raise
                except ValueError as e:
                    # compress_image raises ValueError on failure
                    raise ValidationError(f"Image compression failed: {e}")
                except Exception as e:
                    raise ValidationError(f"Invalid base64 image: {str(e)}")

            # Validate document if provided (decode and process after getting orchestrator)
            document_bytes: bytes | None = None
            if document:
                if not document_type:
                    raise ValidationError("document_type is required when document is provided")
                if document_type not in SUPPORTED_DOCUMENT_FORMATS:
                    raise ValidationError(
                        "Unsupported document format. Supported: PDF, DOCX, XLSX, TXT, CSV, JSON"
                    )
                try:
                    document_bytes = base64.b64decode(document, validate=True)
                    if len(document_bytes) > MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                        raise ValidationError(f"Document exceeds maximum size of {MAX_DOCUMENT_SIZE_MB}MB")
                except ValidationError:
                    raise
                except Exception as e:
                    raise ValidationError(f"Invalid base64 document: {str(e)}")

            # Resolve dependencies
            orchestrator = get_orchestrator()
            continuum_pool = get_continuum_pool()

            # Give previous-turn tool-result compaction a bounded head start before
            # loading the hot continuum. Timeout is intentionally fail-open.
            get_async_work_barrier().wait_for_user(
                user_id,
                timeout=app_config.api.async_work_barrier_timeout_seconds,
                source="tool_summarizer",
            )

            # Get the user's continuum
            continuum = continuum_pool.get_or_create()

            # Increment segment turn counter at API boundary (before any internal processing)
            # This ensures only real user messages increment the counter, not synthetic messages.
            # increment_segment_turn also sets current_segment_id contextvar and
            # (for new segments) defers sentinel persistence to commit-time.
            result = continuum_pool.repository.increment_segment_turn(
                continuum.id, user_id
            )
            segment_turn_number = result.turn_number

            # Process documents into provider-neutral text.
            processed_doc: ProcessedDocument | None = None
            if document_bytes:
                try:
                    processed_doc = process_document(
                        document_bytes,
                        document_type,
                        filename=f"document.{document_type.split('/')[-1]}",  # Extract extension from MIME
                    )
                except ValueError as e:
                    raise ValidationError(f"Document processing failed: {e}")
                except Exception as e:
                    raise ValidationError(f"Document upload failed: {e}")

            # Build content arrays (inference tier for LLM, storage tier for persistence)
            inference_content: str | list[ContentBlock]
            storage_content: str | list[ContentBlock] | None = None

            if compressed:
                # Image: Inference tier (1200px) for current LLM call
                inference_content = [
                    {"type": "text", "text": msg},
                    {
                        "type": "image",
                        "media_type": compressed.inference_media_type,
                        "data": compressed.inference_base64,
                    }
                ]
                # Storage tier (512px WebP) for persistence and multi-turn context
                storage_content = [
                    {"type": "text", "text": msg},
                    {
                        "type": "image",
                        "media_type": compressed.storage_media_type,
                        "data": compressed.storage_base64,
                    }
                ]
            elif processed_doc:
                doc_block: ContentBlock = {
                    "type": "text",
                    "text": f"[Document: {processed_doc.media_type}]\n{processed_doc.data}",
                }

                inference_content = [{"type": "text", "text": msg}, doc_block]
                # Storage uses the same provider-neutral document block.
                storage_content = [
                    {"type": "text", "text": msg},
                    doc_block
                ]
            else:
                inference_content = msg

            # Create a Unit of Work and process via orchestrator
            uow = continuum_pool.begin_work(continuum)

            # Per-request cost tracking (opt-in). Started before any LLM calls
            # fire (including the fast route's subcortical work and every other
            # model_configs route the turn touches), drained
            # after the orchestrator returns so the summary covers every call
            # the turn triggered.
            if show_cost:
                from utils import cost_accumulator
                cost_accumulator.start()

            try:
                continuum, response_text, metadata = orchestrator.process_message(
                    continuum,
                    inference_content,
                    app_config.system_prompt,
                    stream=True,           # orchestrator currently streams internally
                    stream_callback=None,   # no external streaming for HTTP endpoint
                    unit_of_work=uow,
                    storage_content=storage_content,  # 512px WebP for persistence
                    segment_turn_number=segment_turn_number,  # Turn count within segment
                )
            except Exception:
                # process_message stages the accepted user message before model
                # work begins, so a failure after acceptance must still commit
                # what the user was told was received — the HTTP analogue of the
                # websocket turn path. Commit the staged rows, then let the
                # exception propagate to the error response (the outer finally
                # still releases the request lock).
                if uow.pending_messages:
                    uow.commit()
                raise

            # Renew before committing so a long turn cannot race lock expiry
            # against its own commit; a lost renewal (expired + re-acquired)
            # is already unrecoverable and release() will report it.
            _user_request_lock.renew(user_id, lock_token)

            # Commit batched changes
            uow.commit()

            cost_summary = None
            if show_cost:
                from utils import cost_accumulator
                summary = cost_accumulator.drain()
                if summary is not None:
                    cost_summary = summary.to_dict()

            processing_time_ms = int((utc_now() - start_time).total_seconds() * 1000)

            # Build response
            data: dict[str, Any] = {
                "continuum_id": str(continuum.id),
                "response": response_text,
                "metadata": {
                    "tools_used": metadata.get("tools_used", []),
                    "referenced_memories": metadata.get("referenced_memories", []),
                    "surfaced_memories": metadata.get("surfaced_memories", []),
                    "processing_time_ms": processing_time_ms,
                },
            }
            if include_thinking and metadata.get("thinking"):
                data["thinking"] = metadata["thinking"]
            if cost_summary is not None:
                data["cost"] = cost_summary

            return create_success_response(
                data=data,
                meta={
                    "timestamp": format_utc_iso(utc_now()),
                },
            )

        finally:
            renewal_stop.set()
            _user_request_lock.release(user_id, lock_token)


@router.post("/chat")
def chat_endpoint(
    request: ChatRequest,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
):
    """Send a message and receive assistant response as JSON.

    Deliberately sync (not async def) so Starlette runs it in a threadpool
    instead of blocking the event loop during the multi-round tool execution.
    Handler errors propagate to main.py's global exception handlers, which
    assign the HTTP status and build the standard error body.
    """
    handler = ChatEndpoint()
    response = handler.handle_request(
        user_id=current_user.user_id,
        message=request.message,
        image=request.image,
        image_type=request.image_type,
        document=request.document,
        document_type=request.document_type,
        include_thinking=request.include_thinking,
        show_cost=request.show_cost,
    )
    return response.to_dict()
