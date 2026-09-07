"""
SegmentPoller — Generic base for services that poll external sources during
active conversation segments.

Handles all lifecycle and threading concerns:
- Starts a per-user polling thread on first ComposeSystemPromptEvent
- Polls at a configurable interval
- Stops the thread on SegmentCollapsedEvent
- No active segment = no polling = no external calls

Subclasses implement only the protocol-specific parts:
- _try_load_config() — credentials and connection params
- _poll_once(config) — one polling cycle against the external service
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

    # ------------------------------------------------------------------
    # Event handlers (run on event bus thread — synchronous)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Background polling loop (runs in per-user daemon thread)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Abstract methods — subclass implements protocol-specific logic
    # ------------------------------------------------------------------

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
