"""Package marker for the provider-neutral LLM boundary, carrying a convenience re-export.

The sixteen names in `__all__` are the subset of `clients.llm.types` a
casual caller usually wants. Nothing in MIRA imports them through this
module: application code imports `clients.llm.types` (or a submodule such as
`clients.llm.lifecycle`) by full path, and `from clients.llm import
diverter` resolves the submodule rather than any name here. A name added to
the import block but not to `__all__` is therefore invisible to `from
clients.llm import *` and reachable to nobody else either.
"""

from clients.llm.types import (
    DialectName,
    EffortLevel,
    ProviderMetadata,
    ReasoningArtifact,
    ReasoningEntry,
    RequestMetadata,
    Result,
    StopReason,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    coerce_dialect_name,
    coerce_effort,
    coerce_stop_reason,
)

__all__ = [
    "DialectName",
    "EffortLevel",
    "ProviderMetadata",
    "ReasoningArtifact",
    "ReasoningEntry",
    "RequestMetadata",
    "Result",
    "StopReason",
    "ThinkingConfig",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Usage",
    "coerce_dialect_name",
    "coerce_effort",
    "coerce_stop_reason",
]
