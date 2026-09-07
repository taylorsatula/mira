from __future__ import annotations

from pathlib import Path

import pytest

from clients.llm.dialects.base import ProviderProtocolError
from clients.llm.lifecycle import LLMLifecycle
from clients.llm.resolver import ModelResolver
from clients.llm.types import Request, RequestMetadata, ThinkingConfig
from config.config import ApiConfig
from utils import user_context
from utils.user_context import ModelConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixed_model_cache(monkeypatch):
    models = {
        "primary": "qwen/qwen3.7-plus",
        "fast": "qwen/qwen3.7-plus",
        "batch": "qwen/qwen3.7-plus",
    }
    cache = {
        name: ModelConfig(
            name=name,
            model=model,
            dialect_name="openrouter",
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            api_key_name="provider_key",
            effort="high",
            max_tokens=16000,
        )
        for name, model in models.items()
    }
    monkeypatch.setattr(user_context, "_model_config_cache", cache)
    return cache


def test_resolver_accepts_only_fixed_cache_names(fixed_model_cache) -> None:
    resolver = ModelResolver()
    for name in ("primary", "fast", "batch"):
        selection = resolver.resolve(model_config=name)
        assert selection.model_config_name == name
        assert selection.dialect_name == "openrouter"
        assert selection.model == fixed_model_cache[name].model
    with pytest.raises(KeyError, match="Unknown model_config"):
        resolver.resolve(model_config="tidyup")


def test_resolver_uses_configured_dialect_effort_and_token_ceiling(fixed_model_cache) -> None:
    fixed_model_cache["fast"] = ModelConfig(
        name="fast",
        model="model",
        dialect_name="anthropic",
        endpoint_url="https://example.test",
        api_key_name="provider_key",
        effort="low",
        max_tokens=2048,
    )

    selection = ModelResolver().resolve(model_config="fast")

    assert selection.dialect_name == "anthropic"
    assert selection.effort == "low"
    assert selection.max_tokens == 2048


def test_call_site_overrides_take_precedence_over_model_row(fixed_model_cache) -> None:
    selection = ModelResolver().resolve(
        model_config="primary",
        effort="high",
        max_tokens=1024,
    )

    assert selection.effort == "high"
    assert selection.max_tokens == 1024


def test_primary_output_ceiling_bounds_compaction_input_budget() -> None:
    api_config = ApiConfig(
        context_window_tokens=5000,
        compaction_trigger_tokens=4000,
    )

    api_config.validate_compaction_budget(primary_max_tokens=1000)

    with pytest.raises(ValueError, match="primary model input budget of 3999 tokens"):
        api_config.validate_compaction_budget(primary_max_tokens=1001)


def test_provider_timeouts_allow_three_minute_responses() -> None:
    api_config = ApiConfig()

    assert api_config.timeout == 180
    assert api_config.provider_response_timeout == 180


def test_lifecycle_propagates_provider_failure_without_fallback() -> None:
    class FailingDialect:
        dialect_name = "openai"
        endpoint_url = "https://provider.invalid/chat/completions"

        def complete(self, request):
            raise ProviderProtocolError(self.endpoint_url, "complete", "visible failure")

    request = Request(
        messages=({"role": "user", "content": "hello"},),
        system=None,
        tools=(),
        model="model",
        max_tokens=10,
        temperature=1.0,
        thinking=ThinkingConfig(),
        metadata=RequestMetadata(
            endpoint_url=FailingDialect.endpoint_url,
            model_config_name="primary",
        ),
    )
    with pytest.raises(ProviderProtocolError, match="visible failure"):
        LLMLifecycle(response_timeout_seconds=1).complete(request, FailingDialect())


def test_runtime_has_no_legacy_routing_identifiers() -> None:
    roots = ["agents", "clients", "cns", "config", "lt_memory", "tools", "utils", "working_memory"]
    forbidden = (
        "conversation_llm",
        "internal_llm",
        # `usage_pricing` is deliberately absent from this list. Decision D5
        # retains the table and the per-field price lookup in
        # utils/cost_accumulator.py and only re-keys it from internal_llm row
        # names to model_configs route names, so flagging the identifier here
        # would assert the opposite of the locked contract.
        "emergency_fallback",
        "ProviderSwitchEvent",
    )
    offenders: list[str] = []
    for root in roots:
        for path in (ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for term in forbidden:
                if term in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {term}")
    assert offenders == []
