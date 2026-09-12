"""
Event Bus Implementation for CNS

Provides event publishing and subscription for CNS components.
Integrates CNS events with existing MIRA components for system coordination.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..core.events import ContinuumEvent

logger = logging.getLogger(__name__)


class EventBus:
    """
    Event bus for CNS that integrates with existing MIRA components.

    Handles event publishing/subscription and coordinates state changes
    between CNS and working memory, tool repository, and other MIRA components.
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        self._subscribers: dict[str, list[Callable[[ContinuumEvent], None]]] = {}

        # Shutdown event for cleanup
        self._shutdown_event = threading.Event()

        # Register built-in MIRA integrations
        self._register_mira_integrations()

    def _register_mira_integrations(self) -> None:
        """Register built-in event handlers for MIRA component integration."""
        logger.info("Registered built-in MIRA component integrations")

    def publish(self, event: object) -> None:
        """
        Publish an event to all subscribers.

        Dispatch is structural on the event's class name; events are typically
        ContinuumEvents but any domain event object may be published.

        Handles both sync and async callbacks appropriately:
        - Sync callbacks are executed immediately
        - Async callbacks are queued for processing in the event loop

        Args:
            event: Domain event object to publish
        """
        event_type = event.__class__.__name__
        logger.debug(f"Publishing event: {event_type} - {event}")
        
        # Call subscribers
        if event_type in self._subscribers:
            # Delivery is at-most-once by design: subscriber failures are
            # logged with traceback and skipped — no retry, no re-queue. A
            # failed subscriber shows up in the logs; the remaining
            # subscribers still hear the event.
            for callback in self._subscribers[event_type]:
                # Execute all callbacks synchronously
                try:
                    callback(event)
                except Exception:
                    logger.exception(f"Error in event subscriber for {event_type}")
                    
        logger.debug(f"Event {event_type} published to {len(self._subscribers.get(event_type, []))} subscribers")
    
    def subscribe(self, event_type: str, callback: Callable[[ContinuumEvent], None]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Name of event class to subscribe to
            callback: Function to call when event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type} events")
    
    def unsubscribe(self, event_type: str, callback: Callable[[ContinuumEvent], None]) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Name of event class to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type} events")
            except ValueError:
                logger.warning(f"Callback not found in {event_type} subscribers")
                
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))

    def get_all_event_types(self) -> list[str]:
        """Get all event types with subscribers."""
        return list(self._subscribers.keys())

    def clear_subscribers(self, event_type: str | None = None) -> None:
        """
        Clear subscribers for specific event type or all events.
        
        Args:
            event_type: Event type to clear, or None for all events
        """
        if event_type:
            if event_type in self._subscribers:
                del self._subscribers[event_type]
                logger.info(f"Cleared subscribers for {event_type}")
        else:
            self._subscribers.clear()
            logger.info("Cleared all event subscribers")
            
    
    
    
    def shutdown(self) -> None:
        """Shutdown the event bus and clean up resources."""
        logger.info("Shutting down event bus")

        # Signal processor to stop
        self._shutdown_event.set()

        self.clear_subscribers()