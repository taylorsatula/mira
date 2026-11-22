"""
Lattice client for federation messaging over the v2 local API.

Talks to a running `latticed` over its Unix application socket using the
optional `lattice` Python package (lattice/clients/python). Lattice knows no
names: it routes on 32-byte node identifiers, and an address is
`<service>:<mailbox>@<identifier>`, e.g. `mira:taylor@k7f3…` — lowercase
base32 (RFC 4648, no padding) or 64-character hex. The mailbox is opaque to
Lattice and belongs to the service; Mira uses it for the recipient's
username. The payload is opaque bytes; the pager federation body format is
owned by `tools/implementations/pager_tool.py`.

Delivery to the application is at least once and deduped by the 16-byte
message id; the pager tool persists that id as `external_message_id` so a
re-fetched delivery writes nothing twice.

Each call opens and closes its own connection, so there is no connection
reuse and no retry: a transport failure reaches the caller as a lattice
`ConnectionError_`/`Timeout`/`LatticeError`. The service secret comes from
Vault `mira/services` field `lattice_service_secret` and must equal the
secret of this service in latticed's `[services]` table; it is read lazily
on first connect, never at import.
"""

import base64
import binascii
import re
from typing import Optional

from config.config_manager import config

_BASE32_ID = re.compile(r"[a-z2-7]{52}")
_HEX_ID = re.compile(r"[0-9a-fA-F]{64}")
_ADDRESS = re.compile(r"([^:@\s]+):([^:@\s]+)@([^:@\s]+)")
_VALID_PART = re.compile(r"[^:@\s]+")


def parse_node_id(text: str) -> bytes:
    """Parse a 52-char base32 or 64-char hex node identifier to 32 bytes."""
    if _BASE32_ID.fullmatch(text):
        padded = text.upper() + "=" * ((8 - len(text) % 8) % 8)
        try:
            node_id = base64.b32decode(padded)
        except binascii.Error as e:
            raise ValueError(f"node identifier is not valid base32: {e}") from e
    elif _HEX_ID.fullmatch(text):
        node_id = bytes.fromhex(text)
    else:
        raise ValueError("node identifier must be 52-char lowercase base32 or 64-char hex")
    if len(node_id) != 32:
        raise ValueError("node identifier must decode to 32 bytes")
    return node_id


def encode_node_id(node_id: bytes) -> str:
    """Encode a 32-byte node identifier as 52-char lowercase base32."""
    if len(node_id) != 32:
        raise ValueError("node_id must be 32 bytes")
    return base64.b32encode(node_id).decode("ascii").lower().rstrip("=")


def parse_federated_address(address: str) -> tuple[str, str, bytes]:
    """
    Parse `<service>:<mailbox>@<identifier>` into (service, mailbox, node_id).

    Raises ValueError naming the offending part otherwise.
    """
    match = _ADDRESS.fullmatch(address.strip())
    if not match:
        raise ValueError(
            "federated address must be <service>:<mailbox>@<identifier> "
            "(e.g. mira:taylor@<52-char base32 identifier>)"
        )
    service, mailbox, id_text = match.groups()
    return service, mailbox, parse_node_id(id_text)


def format_federated_address(service: str, mailbox: str, node_id: bytes) -> str:
    """Format `<service>:<mailbox>@<base32 identifier>`."""
    for name, part in (("service", service), ("mailbox", mailbox)):
        if not part or not _VALID_PART.fullmatch(part):
            raise ValueError(f"{name} must be non-empty and free of ':', '@' and whitespace")
    return f"{service}:{mailbox}@{encode_node_id(node_id)}"


class LatticeClient:
    """Application-socket client for a running latticed."""

    def __init__(self, app_socket: Optional[str] = None, service: Optional[str] = None):
        self.app_socket = app_socket or config.lattice.app_socket
        self.service = service or config.lattice.service
        self._secret: Optional[bytes] = None

    def _secret_bytes(self) -> bytes:
        if self._secret is None:
            from clients.vault_client import get_service_config
            try:
                secret = get_service_config("lattice_service_secret")
            except Exception as e:
                raise ValueError(
                    "Vault mira/services field lattice_service_secret is not set. Set it to the "
                    "same value as this service's secret in latticed's [services] table "
                    "(vault kv patch secret/mira/services lattice_service_secret=<secret>)."
                ) from e
            if not isinstance(secret, str) or not secret:
                raise ValueError("Vault mira/services field lattice_service_secret must be a non-empty string")
            self._secret = secret.encode("utf-8")
        return self._secret

    def _connect(self):
        try:
            from lattice import Client
        except ImportError as e:
            raise RuntimeError(
                "lattice package is not installed. Install it from the lattice repo: "
                "pip install -e ./lattice/clients/python"
            ) from e
        return Client.connect_app(self.app_socket, self.service, self._secret_bytes())

    def node_id(self) -> bytes:
        """This node's own 32-byte identifier (the hello handshake returns it)."""
        with self._connect() as client:
            return client.node_id

    def send_message(
        self,
        *,
        to_address: str,
        fmbx: str,
        body: bytes,
        priority: Optional[int] = None,
    ) -> bytes:
        """
        Queue a message to a federated address; returns the 16-byte message id.

        `fmbx` is the sender's mailbox (the sender's username) and is echoed to
        the recipient so replies can be addressed. `priority` orders only this
        node's outbound queue and never goes on the wire; it must be 0-3.
        """
        tsvc, tmbx, node_id = parse_federated_address(to_address)
        if not _VALID_PART.fullmatch(fmbx):
            raise ValueError("fmbx (sender mailbox) must be non-empty and free of ':', '@' and whitespace")
        if priority is not None and not 0 <= priority <= 3:
            raise ValueError("priority must be 0-3")
        with self._connect() as client:
            return client.send(
                to=node_id,
                tsvc=tsvc,
                tmbx=tmbx.encode("utf-8"),
                fmbx=fmbx.encode("utf-8"),
                body=body,
                priority=priority,
            )

    def status(self, mid: bytes) -> Optional[tuple]:
        """(status, detail) for a 16-byte message id, or None if unrecorded."""
        with self._connect() as client:
            return client.status(mid)

    def fetch(self, max: int = 16) -> list:
        """Fetch up to `max` deliveries from this service's inbox."""
        with self._connect() as client:
            return client.fetch(max)

    def ack(self, delivery_ids: list) -> int:
        """Acknowledge deliveries; returns how many were removed."""
        with self._connect() as client:
            return client.ack(delivery_ids)

    def lookup(self, node_id: bytes):
        """Whether a valid record for the identifier is held, its mode, and path presence."""
        with self._connect() as client:
            return client.lookup(node_id)


_lattice_client: Optional[LatticeClient] = None


def get_lattice_client() -> LatticeClient:
    """Get or create the global Lattice client instance."""
    global _lattice_client
    if _lattice_client is None:
        _lattice_client = LatticeClient()
    return _lattice_client
