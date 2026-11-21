"""
Peanut Gallery observer service: a standing session record and rare steering
injected into the HUD.

`PeanutGalleryService` subscribes to `TurnCompletedEvent` and fires on every
`PG_TRIGGER_INTERVAL`th segment turn, running the observation on a copied-context
executor so it never blocks the reply. It is fire-and-forget: observation
failures are logged and swallowed, because a failed observation must not affect a
turn that already completed.

The contract between the two halves lives in `PeanutGalleryModel` (what an
observation is made of, how a verdict is read) and `PeanutGalleryTrinket` (how a
record is stored and rendered). This service only orchestrates: snapshot the
continuum and the standing record, run the model, publish one `observe`
update carrying the new beats and any steering. Every pass re-decides the
steering — an absent steering in the result clears the previous one.
"""
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

from cns.core.events import TurnCompletedEvent, UpdateTrinketEvent
from cns.core.message import Message
from cns.integration.event_bus import EventBus
from cns.services.peanutgallery_model import PeanutGalleryModel, PeanutGalleryResult
from working_memory.trinkets.peanutgallery_trinket import PeanutGalleryTrinket

from config import config

logger = logging.getLogger(__name__)

PG_TRIGGER_INTERVAL = 5      # Run observer every N segment turns (broad periodic refresh)


class PeanutGalleryService:
    """
    Orchestrates the Peanut Gallery standing-record observer.

    Every N segment turns, asynchronously:
    1. Captures the turn-completed continuum snapshot and the standing record
    2. Runs PeanutGalleryModel evaluation
    3. Applies the result into the standing record and (rarely) steering

    Fire-and-forget pattern ensures observation doesn't block conversation.
    """

    def __init__(
        self,
        model: PeanutGalleryModel,
        event_bus: EventBus,
        trinket: PeanutGalleryTrinket,
    ):
        self.model = model
        self.event_bus = event_bus
        self.trinket = trinket

        self._executor = ThreadPoolExecutor(
            max_workers=config.worker_pools.peanutgallery_workers,
            thread_name_prefix="peanutgallery"
        )

        event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)

        logger.info(
            "PeanutGalleryService initialized: trigger_interval=%d",
            PG_TRIGGER_INTERVAL,
        )

    def _handle_turn_completed(self, event: TurnCompletedEvent) -> None:
        """
        Handle turn completion - check if it's time for observation.

        Runs every N segment turns based on segment_turn_number.
        Fire-and-forget async execution to avoid blocking the conversation.
        """
        if event.segment_turn_number % PG_TRIGGER_INTERVAL != 0:
            return

        logger.debug(
            "Triggering Peanut Gallery observation at segment turn %d",
            event.segment_turn_number,
        )

        messages_snapshot = tuple(event.continuum.messages)
        standing_log = tuple(self.trinket.get_session_log())

        ctx = contextvars.copy_context()
        self._executor.submit(
            ctx.run,
            self._run_observation,
            event.continuum_id,
            messages_snapshot,
            standing_log,
        )

    def _run_observation(
        self,
        continuum_id: str,
        messages_snapshot: tuple[Message, ...],
        standing_log: tuple[str, ...],
    ) -> None:
        """
        Execute Peanut Gallery observation and apply results.

        Errors are logged but don't affect the main conversation flow.
        """
        try:
            if not messages_snapshot:
                logger.debug("No messages in snapshot for observation")
                return

            result = self.model.evaluate(list(messages_snapshot), list(standing_log))
            self._apply_result(continuum_id, result)

        except Exception as e:
            logger.warning("Peanut Gallery observation failed (non-critical): %s", e)

    def _apply_result(self, continuum_id: str, result: PeanutGalleryResult) -> None:
        """Apply observation results to the standing record and steering."""
        steering = None
        if result.steering:
            steering = {"text": result.steering, "critical": result.steering_critical}

        if result.session_log:
            logger.info(
                "Peanut Gallery: %d new beat(s)",
                len(result.session_log),
            )
        if result.steering:
            logger.info(
                "Peanut Gallery steering (critical=%s): %s...",
                result.steering_critical, result.steering[:80],
            )

        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket="PeanutGalleryTrinket",
            context={
                "action": "observe",
                "log_lines": list(result.session_log),
                "steering": steering,
            },
        ))
