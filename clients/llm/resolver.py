"""Turn a route name into the immutable transport selection one provider call uses.

`ModelResolver.resolve` reads one of MIRA's five DB-configured routes and
builds a frozen, self-validating `ModelSelection`: transport coordinates are
required and blank-rejected, while the credential field goes through
`optional_api_key_name`, which converts the on-disk `''` no-credential
sentinel to `None` so a `ModelSelection` can never carry it. Resolution
therefore fails the same way whether or not the failover diverter is tripped —
the requested route is resolved first and only then swapped for
`_failover_selection`, which keeps the primary's `model_config_name` so cost
pricing and route attribution survive an outage. Calling `ModelSelection`
directly with a raw `''` raises rather than deferring the failure to a Vault
lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

from clients.llm.types import DialectName, EffortLevel, coerce_dialect_name, coerce_effort


def optional_api_key_name(value: object, field_name: str = "api_key_name") -> str | None:
    """Normalise a ``model_configs.api_key_name`` value to the in-memory contract.

    The column is ``text NOT NULL``, so ``''`` is the on-disk representation of
    "this route requires no credential" — ``deploy/postgresql.sh`` clears every
    route's key name when an install is offline and points all five routes at a
    local OpenAI-compatible llama-server. ``None`` is the in-memory
    representation of the same fact, so no consumer downstream has to know the
    sentinel exists.

    A named key is returned stripped so stray padding cannot turn a Vault
    lookup into a miss. This relaxes only the credential field: transport
    parameters (model, endpoint) stay required at the call site.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    return value.strip() or None


@dataclass(frozen=True)
class ModelSelection:
    """Immutable transport selection for one fixed MIRA model route.

    ``api_key_name`` is ``None`` when the route names no credential. Dialects
    that can talk to an unauthenticated endpoint send no credential at all; a
    dialect that cannot (Anthropic) rejects the selection in ``from_selection``.
    """

    dialect_name: DialectName
    model: str
    endpoint_url: str
    api_key_name: str | None
    max_tokens: int
    effort: EffortLevel | None
    model_config_name: str

    def __post_init__(self) -> None:
        for field_name in ("model", "endpoint_url", "model_config_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ModelSelection.{field_name} must be a non-empty string")
        if self.api_key_name is not None and (
            not isinstance(self.api_key_name, str) or not self.api_key_name.strip()
        ):
            raise ValueError(
                "ModelSelection.api_key_name must be None or a non-empty string "
                "(the empty string is the on-disk sentinel; normalise it at the "
                "model_configs read)"
            )
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("ModelSelection.max_tokens must be a positive integer")
        if self.effort is not None:
            object.__setattr__(self, "effort", coerce_effort(self.effort))


class ModelResolver:
    """Resolve one of MIRA's five fixed model routes by name."""

    def resolve(
        self,
        *,
        model_config: str,
        max_tokens: int | None = None,
        effort: EffortLevel | None = None,
    ) -> ModelSelection:
        from utils.user_context import get_model_config

        name = self._required_non_empty_string(model_config, "model_config")
        model_cfg = get_model_config(name)
        selection = self._selection_for_config(name, model_cfg, max_tokens=max_tokens, effort=effort)
        # Transparent divert: while the failover diverter is tripped, every
        # route resolves to the backup target while KEEPING the original route
        # name — cost accounting keys usage_pricing off model_config_name, and
        # nothing downstream learns a new route name. The route is still
        # resolved first so an unknown route name fails the same way tripped or
        # not, and _model_config_cache is never mutated.
        from clients.llm import diverter
        if diverter.tripped():
            return ModelResolver._failover_selection(selection)
        return selection

    @staticmethod
    def _failover_selection(primary: ModelSelection) -> ModelSelection:
        """The backup-target selection for one diverted call.

        max_tokens is min(route ceiling, failover cap): the caller's own
        max_tokens (if it passed one) already resolved into primary.max_tokens,
        so both the caller's intent and the backup's smaller ceiling hold.
        effort stays None — the backup runs with its own thinking defaults;
        a caller that explicitly named an effort keeps it via the Request's
        thinking config, which is merged downstream of this selection.
        """
        from config import config

        cap = config.api.failover_max_tokens_cap
        return ModelSelection(
            dialect_name=coerce_dialect_name(config.api.failover_dialect_name),
            model=ModelResolver._required_non_empty_string(config.api.failover_model, "failover_model"),
            endpoint_url=ModelResolver._required_non_empty_string(
                config.api.failover_endpoint_url, "failover_endpoint_url"
            ),
            api_key_name=optional_api_key_name(config.api.failover_api_key_name, "failover_api_key_name"),
            max_tokens=min(primary.max_tokens, cap) if cap else primary.max_tokens,
            effort=None,
            model_config_name=primary.model_config_name,
        )

    @staticmethod
    def _selection_for_config(
        name: str,
        model_cfg,
        *,
        max_tokens: int | None,
        effort: EffortLevel | None,
    ) -> ModelSelection:
        dialect_name = coerce_dialect_name(model_cfg.dialect_name)
        db_effort = coerce_effort(model_cfg.effort) if model_cfg.effort else None
        return ModelSelection(
            dialect_name=dialect_name,
            model=ModelResolver._required_non_empty_string(model_cfg.model, f"model_config '{name}' model"),
            endpoint_url=ModelResolver._required_non_empty_string(
                model_cfg.endpoint_url,
                f"model_config '{name}' endpoint_url",
            ),
            api_key_name=optional_api_key_name(
                model_cfg.api_key_name,
                f"model_config '{name}' api_key_name",
            ),
            max_tokens=max_tokens if max_tokens is not None else model_cfg.max_tokens,
            effort=effort if effort is not None else db_effort,
            model_config_name=name,
        )

    @staticmethod
    def _required_non_empty_string(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()
