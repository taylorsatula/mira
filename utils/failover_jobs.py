"""Failover recovery probe job: flip the diverter back when the primary recovers.

Registered only when `config.api.failover_enabled` — absent from the scheduler
(and from POST's required-job set) otherwise. The job no-ops while the diverter
is healthy; while tripped it probes every deduped primary model_configs target
with the first-token reachability probe (`clients/llm/probe.py` — the same
mechanism the boot gate uses) and flips back after
`failover_flipback_consecutive_passes` consecutive full passes. One failed
probe resets the pass counter.

Embedding health rides along as advisory visibility only (principal decision:
embeddings never divert — vectors are not cross-model comparable): while
tripped and the primary embedding provider is remote, one real encode is run
against the primary and its outcome is TOAST-logged, nothing more.
"""
from __future__ import annotations

import logging

from config import config

logger = logging.getLogger(__name__)

_EMBEDDING_PROBE_TEXT = "MIRA failover embedding health probe"


def _primary_probe_selections() -> list:
    """One ModelSelection per deduped primary target, mirroring the boot gate's
    grouping key (dialect, endpoint, credential, model): probing the target
    once covers every route it represents."""
    from clients.llm.resolver import ModelSelection, optional_api_key_name
    from clients.llm.types import coerce_dialect_name
    from utils.user_context import get_model_configs

    grouped: dict[tuple[str, str, str, str], ModelSelection] = {}
    for cfg in get_model_configs().values():
        key = (
            cfg.dialect_name,
            cfg.endpoint_url or "",
            cfg.api_key_name or "",
            cfg.model,
        )
        if key in grouped:
            continue
        grouped[key] = ModelSelection(
            dialect_name=coerce_dialect_name(cfg.dialect_name),
            model=cfg.model,
            endpoint_url=cfg.endpoint_url,
            api_key_name=optional_api_key_name(cfg.api_key_name, f"model_config '{cfg.name}' api_key_name"),
            max_tokens=cfg.max_tokens,
            effort=None,
            model_config_name=cfg.name,
        )
    return list(grouped.values())


def _embedding_health_advisory() -> None:
    """One real encode against the primary embedding provider, TOAST-logged.

    Advisory only: the outcome never changes divert state (embeddings do not
    divert — see the module docstring).
    """
    from clients.embeddings_provider import get_embeddings_provider, load_embedding_config

    try:
        cfg = load_embedding_config()
        if cfg.provider != "remote":
            return
        provider = get_embeddings_provider(cache_enabled=False)
        provider.encode_realtime(_EMBEDDING_PROBE_TEXT)
        logger.toast("Failover advisory: primary embedding endpoint answered.")
    except Exception as error:  # noqa: BLE001 - advisory path, never trips or flips
        logger.warning("Failover advisory: primary embedding endpoint failed: %s", error)


def run_failover_recovery_probe() -> dict:
    """Scheduler entry point: probe primaries while tripped, flip on passes."""
    from clients.llm import diverter
    from clients.llm.probe import FIRST_TOKEN_PROBE_SECONDS, probe_first_token

    if not diverter.tripped():
        return {"tripped": False, "probed": 0}

    selections = _primary_probe_selections()
    try:
        for selection in selections:
            probe_first_token(selection, FIRST_TOKEN_PROBE_SECONDS)
    except Exception as error:  # noqa: BLE001 - a failed probe resets the pass counter
        diverter.record_probe_fail(error)
        logger.warning("Failover recovery probe failed: %s", error)
        return {"tripped": True, "probed": len(selections), "passed": False}

    _embedding_health_advisory()
    flipped = diverter.record_probe_pass()
    return {"tripped": diverter.tripped(), "probed": len(selections),
            "passed": True, "flipped": flipped}


def register_failover_probe_job(scheduler_service) -> None:
    """Register the recovery probe when failover is enabled; skip otherwise."""
    if not config.api.failover_enabled:
        logger.info("Provider failover disabled; recovery probe not registered")
        return
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler_service.register_job(
        job_id="llm_failover_recovery_probe",
        func=run_failover_recovery_probe,
        trigger=IntervalTrigger(seconds=config.api.failover_recovery_probe_seconds),
        component="llm",
        description=(
            "While the failover diverter is tripped: first-token probe of every "
            "primary target every "
            f"{config.api.failover_recovery_probe_seconds}s; flip back after "
            f"{config.api.failover_flipback_consecutive_passes} consecutive passes"
        ),
    )
    logger.toast(
        "Provider failover enabled: backup %s @ %s; recovery probe every %ss",
        config.api.failover_model,
        config.api.failover_endpoint_url,
        config.api.failover_recovery_probe_seconds,
    )
