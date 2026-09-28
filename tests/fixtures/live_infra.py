"""Live-infrastructure reachability gate: report BLOCKED and exit nonzero, never a default.

Spec: tests/fixtures/AGENTS.md. Real TCP connects; no simulation. Use before a
live probe so a dead dependency reports "cannot run", distinct from "ran and failed".
"""
from __future__ import annotations

import socket
import sys

_PORTS = {"postgres": 5432, "valkey": 6379, "vault": 8200}


def reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when something accepts a TCP connection on host:port."""
    with socket.socket() as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def require(service: str, host: str = "127.0.0.1", port: int | None = None) -> None:
    """Exit 3 with `<SERVICE>-PROBE-BLOCKED` when the service is not listening.

    A probe that cannot reach its dependency must say so, not run against a fallback:
    the distinction between a blocked probe and a failed probe is the point.
    """
    resolved = port if port is not None else _PORTS.get(service)
    if resolved is None:
        raise ValueError(f"unknown service {service!r}; pass port=")
    if not reachable(host, resolved):
        print(f"{service.upper()}-PROBE-BLOCKED: nothing listening on {host}:{resolved}")
        sys.exit(3)
