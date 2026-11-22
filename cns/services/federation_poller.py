"""
Federation inbox poller: delivers Lattice→Mira messages into local pagers.

latticed cannot call back into Mira; it queues deliveries in this service's
inbox until they are acknowledged. One process-wide daemon thread (started in
`main.py`'s lifespan when `config.lattice.enabled`) fetches batches from the
local API, delivers each to the local user whose username is the delivery's
mailbox (`tmbx`), and acks only after the pager write is durable.
`deliver_federated_message` dedups on `external_message_id` (the lattice
message id, hex), so a re-fetched delivery is written once even across
restarts; the ack itself is the at-least-once re-delivery gate.

Ack policy: anything that can never succeed (unknown recipient, undecodable
body) is logged and acked so it does not clog the inbox forever; anything
unexpected (database error, pager failure) is left un-acked and retried on a
later poll. Priority rides inside the body because lattice never puts it on
the wire; pager priorities top out at 2, so higher lattice values clamp.
"""

import json
import logging
import threading

from clients.postgres_client import PostgresClient
from clients.lattice_client import encode_node_id
from utils.user_context import clear_user_context, set_current_user_id

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 5.0
_FETCH_MAX = 16
_MAX_CONTENT_LENGTH = 1000


def _safe_part(raw, fallback: str = "unknown") -> str:
    """Decode a wire byte string for display; anything unusable becomes hex/fallback."""
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = bytes(raw).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return f"0x{bytes(raw).hex()}"
    if not text or any(c in text for c in ":@") or any(c.isspace() for c in text):
        return fallback
    return text


def _from_address(delivery) -> str:
    fmbx = _safe_part(delivery.fmbx, fallback=f"0x{bytes(delivery.fmbx).hex()}")
    try:
        id_text = encode_node_id(bytes(delivery.from_))
    except ValueError:
        id_text = bytes(delivery.from_).hex() or "unknown"
    return f"{_safe_part(delivery.fsvc)}:{fmbx}@{id_text}"


class FederationInboxPoller:
    """Boot-time daemon thread that drains this service's Lattice inbox into pagers."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="federation-inbox-poller", daemon=True
        )
        self._thread.start()
        logger.info("Federation inbox poller started (interval=%.0fs)", _POLL_INTERVAL_SECONDS)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("Federation inbox poll failed; retrying next interval")
            self._stop_event.wait(_POLL_INTERVAL_SECONDS)

    def _poll_once(self) -> None:
        from clients.lattice_client import get_lattice_client

        client = get_lattice_client()
        deliveries = client.fetch(_FETCH_MAX)
        if not deliveries:
            return

        ack_ids = []
        for delivery in deliveries:
            mid_hex = bytes(delivery.mid).hex()
            try:
                self._deliver_one(delivery)
            except Exception:
                logger.exception("Failed to deliver federated message %s; it will be retried", mid_hex)
            else:
                ack_ids.append(delivery.delivery_id)

        if ack_ids:
            acked = client.ack(ack_ids)
            logger.info("Acknowledged %d federated deliveries", acked)

    def _deliver_one(self, delivery) -> None:
        """
        Deliver one fetched message; return normally when it can be acked.

        Permanently undeliverable deliveries (unknown recipient, undecodable
        body) return normally so the caller acks them; unexpected failures
        (database error, pager failure) raise, leaving the delivery in the
        inbox for a later retry.
        """
        mid_hex = bytes(delivery.mid).hex()

        try:
            username = bytes(delivery.tmbx).decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            logger.warning("Federated message %s: undecodable recipient mailbox; dropping", mid_hex)
            return

        postgres_db = PostgresClient("mira_service")
        row = postgres_db.execute_single(
            """
            SELECT user_id
            FROM global_usernames
            WHERE username = %(username)s
              AND active = true
            """,
            {"username": username},
        )
        if not row:
            logger.warning("Federated message %s: no local user for username %r; dropping", mid_hex, username)
            return

        from_address = _from_address(delivery)

        try:
            body = json.loads(delivery.body)
            if not isinstance(body, dict):
                raise ValueError("body is not an object")
            content = body["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("body.content must be a non-empty string")
            if len(content) > _MAX_CONTENT_LENGTH:
                raise ValueError(f"body.content exceeds {_MAX_CONTENT_LENGTH} characters")
            priority = body.get("priority", 0)
            if not isinstance(priority, int) or isinstance(priority, bool):
                priority = 0
            priority = max(0, min(2, priority))
            location = body.get("location")
            metadata = {"location": location} if location is not None else None
        except (ValueError, KeyError) as e:
            logger.warning("Federated message %s: undecodable body (%s); dropping", mid_hex, e)
            return

        from tools.implementations.pager_tool import PagerTool

        user_id = str(row["user_id"])
        try:
            set_current_user_id(user_id)
            pager = PagerTool()
            result = pager.deliver_federated_message(
                from_address=from_address,
                content=content,
                priority=priority,
                metadata=metadata,
                external_message_id=mid_hex,
            )
        finally:
            clear_user_context()

        if not result.get("success"):
            # deliver_federated_message reports failure only for permanent
            # per-user reasons (e.g. no active pager) — ack so it does not wedge the inbox.
            logger.warning(
                "Federated message %s for user %s not delivered: %s",
                mid_hex, user_id, result.get("error"),
            )
            return

        logger.info("Delivered federated message %s from %s to user %s", mid_hex, from_address, user_id)


_poller: FederationInboxPoller | None = None


def start_federation_inbox_poller() -> FederationInboxPoller:
    """Get or create the process-wide poller and start it."""
    global _poller
    if _poller is None:
        _poller = FederationInboxPoller()
    _poller.start()
    return _poller


def stop_federation_inbox_poller(timeout: float = 5.0) -> None:
    """Stop the process-wide poller; safe to call when it never started."""
    global _poller
    if _poller is not None:
        _poller.stop(timeout=timeout)
