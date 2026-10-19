"""
SegmentPoller — generic base for services that poll an external source while a
conversation segment is active.

A poller holds no thread of its own until a user composes a system prompt:
construction only subscribes to ComposeSystemPromptEvent (start) and
SegmentCollapsedEvent (stop); one daemon thread per user is spawned on the
first compose, runs one cycle per interval, and exits on collapse. No active
segment means no polling and no external calls.

Two mechanisms the caller must respect. `_on_compose` runs on the publishing
thread — the turn path inside working_memory's `_compose_lock` — so
`_try_load_config` must stay a cheap local read; a network call there stalls
every concurrent compose in the process. And the membership check and the
registry insert are separate lock sections, so two composes for one user can
both pass the check: only the upstream one-turn-per-user lock prevents a
double start, and the second start overwrites the state entry, orphaning the
first thread's stop handle so it polls until process exit alongside its
replacement.

Shutdown is best-effort — the collapsed handler joins the thread with a bounded
timeout and returns. A cycle still blocked inside `_poll_once` outlives the
collapse, may publish once more, and briefly co-exists with the replacement
poller the next compose starts.

Subclasses set `poll_interval_seconds`, `target_trinket`, `poller_name`, and
implement two methods: `_try_load_config` (None means the user is unconfigured,
a normal case rather than an error) and `_poll_once` (None means a failed
cycle, `[]` a genuinely empty source). Swapping those two either masks an
outage as an empty listing or blanks a good listing during one.
"""
import logging
import threading
from abc import ABC, abstractmethod
from contextvars import copy_context
from typing import TYPE_CHECKING, NamedTuple

from cns.core.events import (
    ComposeSystemPromptEvent,
    SegmentCollapsedEvent,
    UpdateTrinketEvent,
)

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)


class _PollerState(NamedTuple):
    stop_event: threading.Event
    thread: threading.Thread
    continuum_id: str


class SegmentPoller(ABC):
    """Base for external-service pollers bound to conversation segment lifecycle.

    Subclasses set three class attributes and implement two methods:
        poll_interval_seconds  — how often to poll (e.g. 180 for 3 min)
        target_trinket         — trinket class name for UpdateTrinketEvent
        poller_name            — thread name prefix and log identifier

        _try_load_config()     — return config dict or None if not configured
        _poll_once(config)     — execute one poll, return data or None on failure
    """

    poll_interval_seconds: int
    target_trinket: str
    poller_name: str

    def __init__(self, event_bus: 'EventBus') -> None:
        self._event_bus = event_bus
        self._active_pollers: dict[str, _PollerState] = {}
        self._pollers_lock = threading.Lock()

        event_bus.subscribe('ComposeSystemPromptEvent', self._on_compose)
        event_bus.subscribe('SegmentCollapsedEvent', self._on_segment_collapsed)

        logger.info(
            f"{self.poller_name} poller initialized: "
            f"interval={self.poll_interval_seconds}s, "
            f"trinket={self.target_trinket}"
        )

    def _on_compose(self, event: ComposeSystemPromptEvent) -> None:
        """Start polling for this user if not already running."""
        user_id = event.user_id
        with self._pollers_lock:
            if user_id in self._active_pollers:
                return

        config = self._try_load_config()
        if config is None:
            logger.debug(
                f"{self.poller_name}: no config for user {user_id[:8]}, "
                "skipping poller start"
            )
            return

        continuum_id = event.continuum_id
        stop_event = threading.Event()
        ctx = copy_context()

        thread = threading.Thread(
            target=ctx.run,
            args=(self._poll_loop, user_id, continuum_id, config, stop_event),
            name=f"{self.poller_name}-{user_id[:8]}",
            daemon=True,
        )

        with self._pollers_lock:
            self._active_pollers[user_id] = _PollerState(
                stop_event=stop_event,
                thread=thread,
                continuum_id=continuum_id,
            )

        thread.start()
        logger.info(
            f"{self.poller_name} poller started for user {user_id[:8]}"
        )

    def _on_segment_collapsed(self, event: SegmentCollapsedEvent) -> None:
        """Stop polling for this user."""
        user_id = event.user_id
        with self._pollers_lock:
            state = self._active_pollers.pop(user_id, None)
        if state is None:
            return

        state.stop_event.set()
        state.thread.join(timeout=5)
        if state.thread.is_alive():
            logger.warning(
                f"{self.poller_name} poller thread for user {user_id[:8]} "
                "did not exit within 5s (daemon, will die on shutdown)"
            )
        else:
            logger.info(
                f"{self.poller_name} poller stopped for user {user_id[:8]}"
            )

    def _poll_loop(
        self,
        user_id: str,
        continuum_id: str,
        config: dict,
        stop_event: threading.Event,
    ) -> None:
        """Poll at interval until stop signal. Errors skip the cycle."""
        logger.debug(
            f"{self.poller_name} poll loop starting for user {user_id[:8]}"
        )

        while not stop_event.is_set():
            try:
                data = self._poll_once(config)
                if data is not None:
                    self._event_bus.publish(UpdateTrinketEvent.create(
                        continuum_id=continuum_id,
                        target_trinket=self.target_trinket,
                        context={'data': data},
                    ))
            except Exception as e:
                logger.warning(
                    f"{self.poller_name} poll cycle failed for user "
                    f"{user_id[:8]}: {e}",
                    exc_info=True,
                )

            stop_event.wait(timeout=self.poll_interval_seconds)

        with self._pollers_lock:
            state = self._active_pollers.get(user_id)
            if state is not None and state.stop_event is stop_event:
                self._active_pollers.pop(user_id, None)

        logger.debug(
            f"{self.poller_name} poll loop exiting for user {user_id[:8]}"
        )

    @abstractmethod
    def _try_load_config(self) -> dict | None:
        """Load service config for the current user.

        Runs on the main event bus thread — user context is available
        via contextvar.

        Returns:
            Config dict with connection params, or None if the service
            is not configured for this user (missing credentials, etc.).
            Returning None is a normal case, not an error.
        """

    @abstractmethod
    def _poll_once(self, config: dict) -> list[dict] | None:
        """Execute one polling cycle against the external service.

        Runs on a background daemon thread — user context is available
        via copy_context().

        Args:
            config: The dict returned by _try_load_config().

        Returns:
            List of data dicts to publish to the trinket, or None if the
            poll failed (logged, cycle skipped, no crash).
        """
