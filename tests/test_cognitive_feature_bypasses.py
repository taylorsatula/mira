from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.config_manager import AppConfig
from cns.integration.event_bus import EventBus
from cns.integration.factory import CNSIntegrationFactory
from cns.services.orchestrator import ContinuumOrchestrator


def _feature_config(*, subcortical_enabled: bool, peanutgallery_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(subcortical_prefill_warmup=False),
        system=SimpleNamespace(
            subcortical_enabled=subcortical_enabled,
            peanutgallery_enabled=peanutgallery_enabled,
        ),
    )


def test_feature_flags_load_from_strict_environment_switches(monkeypatch) -> None:
    monkeypatch.setenv("MIRA_SUBCORTICAL_ENABLED", "0")
    monkeypatch.setenv("MIRA_PEANUTGALLERY_ENABLED", "0")

    loaded = AppConfig.load()

    assert loaded.system.subcortical_enabled is False
    assert loaded.system.peanutgallery_enabled is False


def test_feature_flags_reject_ambiguous_environment_values(monkeypatch) -> None:
    monkeypatch.setenv("MIRA_SUBCORTICAL_ENABLED", "false")

    with pytest.raises(ValueError, match="MIRA_SUBCORTICAL_ENABLED must be exactly 0 or 1"):
        AppConfig.load()


def test_disabled_subcortical_layer_is_not_constructed() -> None:
    factory = CNSIntegrationFactory(
        _feature_config(subcortical_enabled=False, peanutgallery_enabled=True)
    )

    assert factory._get_subcortical_layer(SimpleNamespace()) is None


def test_disabled_peanutgallery_registers_no_turn_subscriber() -> None:
    factory = CNSIntegrationFactory(
        _feature_config(subcortical_enabled=True, peanutgallery_enabled=False)
    )
    event_bus = EventBus()
    subscriber_count = event_bus.get_subscriber_count("TurnCompletedEvent")

    factory._initialize_peanutgallery_service(event_bus, SimpleNamespace())

    assert factory._peanutgallery_service is None
    assert event_bus.get_subscriber_count("TurnCompletedEvent") == subscriber_count


def test_subcortical_fast_path_returns_no_stale_or_fresh_memories() -> None:
    orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
    orchestrator.subcortical_layer = None

    result = orchestrator._surface_memories(
        continuum=SimpleNamespace(),
        text_for_context="hello",
        previous_memories=[{
            "id": "existing-memory",
            "text": "stale context",
            "importance_score": 1.0,
        }],
    )

    assert result.surfaced_memories == []
    assert result.pinned_ids == set()
    assert result.subcortical_result is None


def test_enabled_subcortical_failure_propagates() -> None:
    class FailingSubcorticalLayer:
        def generate(self, continuum, text_for_context, *, previous_memories):
            raise RuntimeError("provider unavailable")

    orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
    orchestrator.subcortical_layer = FailingSubcorticalLayer()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        orchestrator._surface_memories(
            continuum=SimpleNamespace(),
            text_for_context="hello",
            previous_memories=[],
        )
