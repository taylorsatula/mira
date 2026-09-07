"""
Per-request LLM cost accumulator for user-facing "what did this cost" feedback.

A request handler (typically `cns/api/chat.py`) calls `start()` when the user
has asked to see cost. Every completed provider call in
`clients/llm/lifecycle.py` then calls `record(result)` with the final
`Result`, from which the usage tokens and the `model_configs` route name are
derived. At the end of the request the handler calls `drain()` which returns
a structured summary.

Pricing is looked up from the `usage_pricing` table, price field by field, in
precedence order: the route's own row, then the reserved `__default__` row, then
`FALLBACK_PRICES` below. The lower tiers are what keeps the feature useful in OSS,
where the closed `billing` module isn't present to auto-populate from OpenRouter,
and where the greenfield schema seeds route rows with no prices at all.

The accumulator is scoped to a contextvar, so call-site code can be ignorant
of whether cost tracking is active — `record()` is a no-op when it isn't.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict

from utils.database_session_manager import get_shared_session_manager

if TYPE_CHECKING:
    from clients.llm.types import Result


# Public pricing as of 2026-04. Used only when `usage_pricing` has no price for
# a field the user exercised, for either the route or the reserved default row.
# Values are USD per million tokens.
# Updating this table is a low-risk change; Anthropic and Groq publish their
# prices openly. A hosted install with a real billing backend will have
# populated `usage_pricing` rows and never hit this fallback.
FALLBACK_PRICES: Dict[str, Dict[str, float]] = {
    # Anthropic Claude
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00, "cache_read": 1.50,  "cache_write": 18.75},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00, "cache_read": 0.30,  "cache_write":  3.75},
    "claude-haiku-4-5":          {"input":  1.00, "output":  5.00, "cache_read": 0.10,  "cache_write":  1.25},
    "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00, "cache_read": 0.10,  "cache_write":  1.25},
    # Groq (subcortical)
    "qwen/qwen3.6-27b":          {"input":  0.29, "output":  0.39, "cache_read": 0.0,   "cache_write":  0.0},
}

# Reserved `usage_pricing` key seeded by the greenfield schema as the fallback
# price pair for any route without an explicit price. It is not a model_configs
# route name, so it never collides with a pricing_key.
DEFAULT_PRICING_KEY = "__default__"

_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
# A record is only costable when both dominant dimensions resolve somewhere in
# the chain. The `__default__` row is deliberately seeded with input/output
# alone, so an unresolved cache field costs 0.0 rather than voiding the record.
_CORE_PRICE_FIELDS = ("input", "output")


class _UsageRecord(TypedDict):
    pricing_key: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class CostSummary:
    """Structured cost summary for a single request."""
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    calls: int
    unpriced_calls: int
    fallback_prices_used: bool

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "fallback_prices_used": self.fallback_prices_used,
        }


_records: ContextVar[Optional[List[_UsageRecord]]] = ContextVar(
    "cost_accumulator_records", default=None
)


def start() -> None:
    """Begin accumulating usage records for the current request."""
    _records.set([])


def is_active() -> bool:
    """True if the accumulator has been started for the current request."""
    return _records.get() is not None


def record(result: "Result") -> None:
    """Append a usage record derived from one completed provider Result.

    No-op if no accumulator is active or the result carries no usage. The
    pricing key is the result's `model_configs` route name (primary, fast,
    batch, assessment, or other); None route provenance means the call cannot
    be priced and is skipped.
    """
    records = _records.get()
    if records is None:
        return
    usage = result.usage
    if usage is None:
        return
    metadata = result.provider_metadata
    pricing_key = _resolve_pricing_key(metadata.model_config_name)
    if pricing_key is None:
        # Called outside user context (rare — startup, admin jobs). Skip silently;
        # cost tracking is only meaningful inside a user request anyway.
        return
    records.append(_UsageRecord(
        pricing_key=pricing_key,
        model=metadata.model or "unknown",
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        cache_read_tokens=int(usage.cache_read_input_tokens or 0),
        cache_write_tokens=int(usage.cache_creation_input_tokens or 0),
    ))


def drain() -> Optional[CostSummary]:
    """Consume the accumulator and return a priced summary.

    Returns None if no accumulator was active. The accumulator is always
    cleared, even on partial pricing data.
    """
    records = _records.get()
    _records.set(None)
    if not records:
        return None

    prices = _fetch_prices({r["pricing_key"] for r in records})

    total_in = sum(r["input_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    total_cr = sum(r["cache_read_tokens"] for r in records)
    total_cw = sum(r["cache_write_tokens"] for r in records)

    total_cost = 0.0
    unpriced = 0
    used_fallback = False

    for r in records:
        db_price = prices.get(r["pricing_key"])
        price, is_fallback = _price_for_record(db_price, prices.get(DEFAULT_PRICING_KEY), r["model"])
        if price is None:
            unpriced += 1
            continue
        if is_fallback:
            used_fallback = True
        total_cost += (
            (r["input_tokens"]       / 1_000_000) * price["input"] +
            (r["output_tokens"]      / 1_000_000) * price["output"] +
            (r["cache_read_tokens"]  / 1_000_000) * price["cache_read"] +
            (r["cache_write_tokens"] / 1_000_000) * price["cache_write"]
        )

    return CostSummary(
        input_tokens=total_in,
        output_tokens=total_out,
        cache_read_tokens=total_cr,
        cache_write_tokens=total_cw,
        cost_usd=total_cost,
        calls=len(records),
        unpriced_calls=unpriced,
        fallback_prices_used=used_fallback,
    )


def _resolve_pricing_key(model_config_name: Optional[str]) -> Optional[str]:
    """Compute the `usage_pricing.name` for the call being recorded.

    `usage_pricing` is keyed by `model_configs` route name (D5 re-key).
    Returns None when there is no active user context or the call carried no
    route name — cost tracking is only meaningful inside a user request.

    The `__default__` fallback is applied at pricing time in `_price_for_record`,
    not here: this function records which route made the call, so an operator can
    seed a route-specific row later without re-attributing past usage.
    """
    from utils.user_context import has_user_context

    if not has_user_context():
        return None

    return model_config_name


def _fetch_prices(keys: set[str]) -> Dict[str, Optional[Dict[str, Optional[float]]]]:
    """Return {pricing_key: price_dict_or_None} for the given keys and the default.

    Each price_dict has 'input'/'output'/'cache_read'/'cache_write' floats, any
    of which may be None if that column is NULL. Keys with no row at all map
    to None. The reserved `__default__` row is always fetched — it is the tier
    between a route's row and FALLBACK_PRICES.
    """
    lookup_keys = set(keys)
    lookup_keys.add(DEFAULT_PRICING_KEY)
    with get_shared_session_manager().get_admin_session() as session:
        rows = session.execute_query(
            """SELECT name,
                      input_price_per_mtok,
                      output_price_per_mtok,
                      cache_read_price_per_mtok,
                      cache_write_price_per_mtok
                 FROM usage_pricing
                WHERE name = ANY(%(keys)s)""",
            {"keys": list(lookup_keys)},
        )
    result: Dict[str, Optional[Dict[str, Optional[float]]]] = {k: None for k in lookup_keys}
    for row in rows:
        result[row["name"]] = {
            "input":       float(row["input_price_per_mtok"])       if row["input_price_per_mtok"]       is not None else None,
            "output":      float(row["output_price_per_mtok"])      if row["output_price_per_mtok"]      is not None else None,
            "cache_read":  float(row["cache_read_price_per_mtok"])  if row["cache_read_price_per_mtok"]  is not None else None,
            "cache_write": float(row["cache_write_price_per_mtok"]) if row["cache_write_price_per_mtok"] is not None else None,
        }
    return result


def _price_for_record(
    db_price: Optional[Dict[str, Optional[float]]],
    default_price: Optional[Dict[str, Optional[float]]],
    model: str,
) -> tuple[Optional[Dict[str, float]], bool]:
    """Resolve one record's prices through the fallback chain, field by field.

    Precedence for each of input/output/cache_read/cache_write: the route's own
    `usage_pricing` row, then the reserved `__default__` row, then
    FALLBACK_PRICES by model name, then no price at all. Resolving per field is
    what makes a partially populated row usable — the schema documents a non-NULL
    column as a manual override and seeds `__default__` with input/output only.

    Returns (price_dict, is_fallback). price_dict is None when input or output
    resolved nowhere, so the record is counted but not costed; a cache field that
    resolved nowhere contributes 0.0. When a price is returned, `price_dict` has
    all four fields as concrete floats, as before.
    """
    fallback = FALLBACK_PRICES.get(model)
    resolved: Dict[str, Optional[float]] = {}
    used_fallback = False
    for field in _PRICE_FIELDS:
        value: Optional[float] = None
        for tier, from_fallback in ((db_price, False), (default_price, False), (fallback, True)):
            if tier is None:
                continue
            candidate = tier.get(field)
            if candidate is not None:
                value = float(candidate)
                used_fallback = used_fallback or from_fallback
                break
        resolved[field] = value

    if any(resolved[field] is None for field in _CORE_PRICE_FIELDS):
        return (None, False)
    return (
        {
            field: (resolved[field] if resolved[field] is not None else 0.0)
            for field in _PRICE_FIELDS
        },
        used_fallback,
    )
