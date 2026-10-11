"""First-token provider reachability probe — one sanctioned mechanism, two consumers.

Success is the first well-formed streamed token, not a completed completion:
the dialects raise `ProviderProtocolError` on malformed chunks, so any yielded
event is well-formed by construction, and a thinking model's first token may be
a reasoning delta — text, thinking, and tool-detection all count. Reachability
and generation latency are different properties; the bound must exceed
first-token latency, not completion latency (a full non-streaming probe under
a completion-shaped bound fails healthy thinking providers under load).

Consumers:
- `utils/power_on_self_test.py` boot gate (all deduped model_configs targets).
- `utils/failover_jobs.py` recovery probe (while the diverter is tripped).
"""
from __future__ import annotations

from dataclasses import dataclass

from clients.llm.resolver import ModelSelection

# First-token stall bound, not a generation bound. A thinking model under load
# can spend 25s+ before its first token while streaming normally after that
# (measured: gw.lunaroute.com glm-5.3, 2026-10-11, first streamed byte 25.2s
# solo), and concurrent probes over one API key queue behind each other —
# observed first-token latency has a tail past 60s. The bound covers the tail;
# a genuinely hung endpoint still stalls at it.
FIRST_TOKEN_PROBE_SECONDS = 120

PROBE_SYSTEM = "MIRA provider reachability probe. Reply with OK."
PROBE_MESSAGE = "Reply with OK."
PROBE_MAX_TOKENS = 32


@dataclass(frozen=True)
class FirstTokenProbeResult:
    """Which event kind established reachability, and how much content it carried."""
    event_type: str
    content_chars: int


def probe_first_token(selection: ModelSelection, timeout_seconds: float) -> FirstTokenProbeResult:
    """Stream one tiny request and succeed on the first well-formed token event.

    Raises on any provider failure: `ProviderStallError` at the bound for a
    hung endpoint, `ProviderProtocolError` for garbage, the mapped transport
    errors for HTTP failures. After the first token the generation is aborted —
    `abort_active_stream` closes the httpx response from outside (the documented
    unblock for a reader in another thread) and closing the lifecycle generator
    unwinds both generators and releases the connection; max_tokens stays capped
    at PROBE_MAX_TOKENS so the abandoned tail is bounded.
    """
    from clients.llm.dialect_registry import get_registry
    from clients.llm.events import CompleteEvent, TextEvent, ThinkingEvent, ToolDetectedEvent
    from clients.llm.lifecycle import LLMLifecycle
    from clients.llm.types import Request, RequestMetadata, ThinkingConfig
    from config import config

    dialect_class = get_registry().get(selection.dialect_name)
    dialect = dialect_class.from_selection(
        selection,
        api_key=None,
        timeout=timeout_seconds,
        artifact_sink=None,
    )
    request = Request(
        messages=[{"role": "user", "content": PROBE_MESSAGE}],
        system=PROBE_SYSTEM,
        model=selection.model,
        max_tokens=min(selection.max_tokens, PROBE_MAX_TOKENS),
        temperature=config.api.temperature,
        thinking=ThinkingConfig(),
        metadata=RequestMetadata(
            endpoint_url=selection.endpoint_url,
            model_config_name=selection.model_config_name,
        ),
    )

    events = LLMLifecycle(response_timeout_seconds=timeout_seconds).stream(request, dialect)
    try:
        for event in events:
            if isinstance(event, (TextEvent, ThinkingEvent, ToolDetectedEvent)):
                return FirstTokenProbeResult(
                    event_type=type(event).__name__,
                    content_chars=len(getattr(event, "content", "") or ""),
                )
            if isinstance(event, CompleteEvent):
                # Stream ended without a single token event: a completion
                # carrying no text, reasoning, or tool calls is not a
                # reachable provider.
                response = event.response
                reasoning_text = response.reasoning.text if response.reasoning is not None else ""
                if not response.text.strip() and not reasoning_text.strip() and not response.tool_calls:
                    raise RuntimeError("Provider returned no text, reasoning, or tool calls")
                return FirstTokenProbeResult(
                    event_type="CompleteEvent",
                    content_chars=len(response.text) + len(reasoning_text),
                )
    finally:
        abort = getattr(dialect, "abort_active_stream", None)
        if abort is not None:
            abort()
        events.close()
