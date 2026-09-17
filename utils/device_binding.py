"""Device low-power binding site for the heartbeat wake cycle.

At the end of every dispatcher pass, cns/services/heartbeat_service.py knows
the earliest time any user's next heartbeat wake is due — the per-user sleep
schedule lives on segment sentinels as `heartbeat_wake_at`. This module is the
single sanctioned place where that fact crosses out of MIRA into a
deployer-supplied shim that bridges to whatever C++ actually puts the metal
into a low-power state and arms the wake.

The shim is NOT MIRA code. It is deployed by whoever owns the device, and it
must define exactly one function:

    def arm_low_power(wake_at_utc: str, wake_lead_seconds: int,
                      metadata: HeartbeatSleepMetadata) -> None

      wake_at_utc        UTC ISO timestamp of the earliest next heartbeat
                         wake obligation across all users with active segments
      wake_lead_seconds  the device should be back at full power this many
                         seconds before wake_at_utc so the dispatcher tick
                         fires on time
      metadata           the routing field — everything the far side needs to
                         decide what this ping is about and where it goes
                         (camera, low-power controller, actuator, anything
                         else the deployer wired up): a `kind` tag plus the
                         per-user wake schedule composing the aggregate.
                         MIRA produces structured facts only; freeform
                         content never crosses this link, and the field's
                         shape lives in `HeartbeatSleepMetadata` below.
      On failure: raise. The failure is logged here and the tick continues.

MIRA never suspends anything itself. APScheduler keeps running regardless of
what the shim does; a shim that suspends the machine owns the RTC alarm (or
equivalent) that brings it back, and owns the consequence that nothing else on
the box — websockets, scheduled jobs, inbound traffic — runs while it sleeps.

Failure policy: log and continue, by explicit design decision. The binding is
optional infrastructure; a failure leaves the device at full power, which is
exactly the unbound behavior, so the heartbeat cycle's correctness is never
affected. Load failures are cached per module path so a broken shim logs
once, not once per tick pass; fix the shim and restart (or point the config at
a different path) to clear the cached failure.

--------------------------------------------------------------------------------
FOR A FUTURE AGENT: touchpoints for making this binding configurable
--------------------------------------------------------------------------------

Everything is wired; configuration is the only thing left. Today there is a
single process-global binding selected by
`config.heartbeat.device_power_binding.module_path`
(`config/config.py`, DevicePowerBindingConfig — a nested Pydantic block on
HeartbeatConfig). If you're here to make the binding richer — per-user, per
device class, dynamically reloaded, exposed through the web UI — the broad
flow is:

1. Config shape: extend `DevicePowerBindingConfig` in `config/config.py`. The
   heartbeat config block is consumed at two places — job registration and
   every tick — read both before changing semantics
   (`cns/services/heartbeat_service.py`: `register_heartbeat_job`,
   `heartbeat_tick`). If the config becomes per-user, follow the established
   per-user override pattern in `config/config_manager.py` / the
   `config.<tool>_tool` merge in `utils/tool_config_store.py`, NOT a new
   mechanism.

2. The publisher: `heartbeat_tick()` in `cns/services/heartbeat_service.py`
   computes the aggregation (earliest wake across users, "stay awake" cases)
   right after the per-segment loop and hands one ISO timestamp to
   `notify_heartbeat_sleep()` below. Any configurability that changes WHEN or
   WHAT gets published — per-user bindings, publishing on confirm instead of
   per-pass, an opt-out — is decided there. Read the loop's `stay_awake` /
   `earliest_wake` tracking comments; every branch has a reason.

3. This module: `notify_heartbeat_sleep()` is the only entry point callers
   should use. The loader (`_load_arm_fn`) resolves the shim from config and
   caches it per path. If you add config the shim needs (e.g. a device
   profile), thread it through the arguments to `arm_low_power` rather than
   having the shim read MIRA's config itself — the shim shouldn't know MIRA's
   config system exists. The `metadata` argument is the routing seam: new
   far-side capabilities extend what MIRA puts in `HeartbeatSleepMetadata`
   (or add new `kind` values), never the argument list.

4. The heartbeat status surface: `heartbeat_state()` in
   `cns/services/heartbeat_service.py` feeds the control API
   (`cns/api/heartbeat_api.py`). If operators should be able to see or toggle
   the binding at runtime, that's the surface to extend — it already reports
   latch, config, and recent decisions.

5. Verification: this is a live-path module. Any behavioral change gets a
   probe that imports a REAL shim file (write one to /tmp), patches the
   config path in-process (the config object is mutable), calls
   `notify_heartbeat_sleep`, and asserts the shim observed the call — plus
   the failure path (shim raises → warning logged, no exception escapes).
   No mocks; the shim-under-test is a real file with real imports.
"""
import importlib.util
import logging
from typing import Callable, Dict, Optional, TypedDict

logger = logging.getLogger(__name__)

class HeartbeatSleepMetadata(TypedDict):
    """The single metadata field carried on the device link.

    Structured routing facts only — never freeform content. `kind` is the
    coarse routing tag for the far side (what this ping is about);
    `user_wake_times` is the per-user wake schedule that composed the
    aggregate, so the far side can attribute the sleep window without a
    second round trip."""

    kind: str  # "heartbeat_sleep" today; new kinds extend this
    user_wake_times: Dict[str, str]  # user_id -> next wake UTC ISO

# Resolved arm_low_power callables keyed by module path. A cached None means
# that path failed to load and was already warned about — retry only when the
# configured path itself changes.
_arm_fn_by_path: Dict[str, Optional[Callable[..., None]]] = {}


def _load_arm_fn(module_path: str) -> Optional[Callable[..., None]]:
    """Resolve the deployer shim's arm_low_power from its file path, cached.

    Returns None when the module cannot be loaded or does not expose a
    callable arm_low_power; the failure is warned once per path and never
    raises — a broken binding must not take the heartbeat cycle down."""
    if module_path in _arm_fn_by_path:
        return _arm_fn_by_path[module_path]
    try:
        spec = importlib.util.spec_from_file_location("mira_device_binding", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build an import spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arm_fn = getattr(module, "arm_low_power", None)
        if not callable(arm_fn):
            raise ValueError(
                f"{module_path} does not define callable "
                f"arm_low_power(wake_at_utc, wake_lead_seconds, metadata)"
            )
        _arm_fn_by_path[module_path] = arm_fn
        return arm_fn
    except Exception as exc:
        logger.warning(
            "Device power binding %s failed to load (%s); device will not "
            "enter low-power state until the shim is fixed and the path "
            "re-resolved", module_path, exc,
        )
        _arm_fn_by_path[module_path] = None
        return None


def notify_heartbeat_sleep(wake_at_utc: str, metadata: HeartbeatSleepMetadata) -> None:
    """Publish the earliest next heartbeat wake obligation to the device binding.

    No-op when no shim is configured (the default unbound state). Never
    raises: a binding failure leaves the device at full power, which is the
    unbound behavior, so the heartbeat cycle continues unaffected."""
    from config.config_manager import config

    binding = config.heartbeat.device_power_binding
    if not binding.module_path:
        return
    arm_fn = _load_arm_fn(binding.module_path)
    if arm_fn is None:
        return
    try:
        arm_fn(wake_at_utc, binding.wake_lead_seconds, metadata)
    except Exception as exc:
        logger.warning(
            "Device power binding %s failed to arm low-power state until %s "
            "(%s); device stays at full power",
            binding.module_path, wake_at_utc, exc,
        )
