"""The dict shapes crossing the composer/core boundary.

`ComposedPrompt` is what `SystemPromptComposer.compose()` returns and what
`WorkingMemory._handle_compose_prompt` unpacks into event fields;
`TrinketState`, `TrinketStatesMeta` and `AllTrinketStates` describe the two
working-memory API read responses. These are `TypedDict`s, so nothing checks
them at runtime: a producer that renames or omits a key surfaces as a
`KeyError` in the consumer, never as an error at the point the dict is built.
"""
from typing import TypedDict


class ComposedPrompt(TypedDict):
    """Output of SystemPromptComposer.compose()."""
    cached_content: str
    non_cached_content: str
    post_history_items: list[str]
    notification_center: str


class TrinketState(TypedDict):
    """Single trinket's cached state from Valkey (returned by get_trinket_state)."""
    section_name: str
    content: str
    cache_policy: bool
    last_updated: str | None


class TrinketStatesMeta(TypedDict):
    """Metadata block within AllTrinketStates."""
    trinket_count: int
    loaded_at: str


class AllTrinketStates(TypedDict):
    """Output of WorkingMemory.get_all_trinket_states()."""
    trinkets: list[TrinketState]
    meta: TrinketStatesMeta
