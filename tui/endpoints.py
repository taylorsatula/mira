"""EndpointStore: named MIRA endpoints and the active pointer, persisted as
0600 JSON at ~/.config/mira-tui/config.json.

Fail-fast: a corrupt file raises ValueError naming the path and the parse
error — never a silent reset. Every write re-applies 0600 so a umask change
or an editor swap cannot leave credentials world-readable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HISTORY_FETCH_MODES = ("session_plus_summary", "session_only", "all")
_DEFAULT_PATH = Path.home() / ".config" / "mira-tui" / "config.json"


@dataclass
class EndpointConfig:
    """One named deployed MIRA instance."""

    base_url: str  # e.g. "http://localhost:1993", no trailing slash
    api_key: str
    history_fetch: str = "session_plus_summary"  # one of HISTORY_FETCH_MODES
    include_thinking: bool = True


def _validate_history_fetch(mode: str) -> None:
    if mode not in HISTORY_FETCH_MODES:
        raise ValueError(
            f"Unknown history_fetch mode {mode!r}; expected one of "
            + ", ".join(HISTORY_FETCH_MODES)
        )


class EndpointStore:
    """Named endpoints with an active pointer, persisted immediately on
    every mutation."""

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else _DEFAULT_PATH
        self._endpoints: dict[str, EndpointConfig] = {}
        self._active: str | None = None

    def load(self) -> None:
        """Load from disk; missing file creates the default empty store.

        Raises ValueError on corrupt JSON or an invalid stored value —
        fail-fast, never silently reset.
        """
        if not self.path.exists():
            self._endpoints = {}
            self._active = None
            self._write()
            return
        # chmod follows symlinks: a symlinked store path protects the target file.
        try:
            os.chmod(self.path, 0o600)
        except OSError as error:
            raise OSError(f"cannot chmod 0600 {self.path}: {error}") from error
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Corrupt endpoint config {self.path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"Corrupt endpoint config {self.path}: expected a JSON object")
        active = raw.get("active")
        if active is not None and not isinstance(active, str):
            raise ValueError(f"Corrupt endpoint config {self.path}: 'active' must be a string or null")
        endpoints_raw = raw.get("endpoints")
        if not isinstance(endpoints_raw, dict):
            raise ValueError(f"Corrupt endpoint config {self.path}: 'endpoints' must be an object")
        endpoints: dict[str, EndpointConfig] = {}
        for name, fields in endpoints_raw.items():
            if not isinstance(fields, dict):
                raise ValueError(f"Corrupt endpoint config {self.path}: endpoint {name!r} must be an object")
            for field, expected_type in (
                ("base_url", str),
                ("api_key", str),
                ("include_thinking", bool),
            ):
                if field in fields and not isinstance(fields[field], expected_type):
                    raise ValueError(
                        f"Corrupt endpoint config {self.path}: endpoint {name!r} "
                        f"field {field!r} must be {expected_type.__name__}"
                    )
            try:
                cfg = EndpointConfig(
                    base_url=fields["base_url"],
                    api_key=fields["api_key"],
                    history_fetch=fields.get("history_fetch", "session_plus_summary"),
                    include_thinking=fields.get("include_thinking", True),
                )
            except KeyError as error:
                raise ValueError(
                    f"Corrupt endpoint config {self.path}: endpoint {name!r} is missing {error}"
                ) from error
            _validate_history_fetch(cfg.history_fetch)
            endpoints[name] = cfg
        if active is not None and active not in endpoints:
            raise ValueError(
                f"Corrupt endpoint config {self.path}: active endpoint {active!r} is not defined"
            )
        self._endpoints = endpoints
        self._active = active

    def _write(self) -> None:
        """Persist the full store and re-apply 0600 on every save."""
        payload = {
            "active": self._active,
            "endpoints": {
                name: {
                    "base_url": cfg.base_url,
                    "api_key": cfg.api_key,
                    "history_fetch": cfg.history_fetch,
                    "include_thinking": cfg.include_thinking,
                }
                for name, cfg in self._endpoints.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.chmod(self.path, 0o600)

    def names(self) -> list[str]:
        return list(self._endpoints)

    def get(self, name: str) -> EndpointConfig:
        try:
            return self._endpoints[name]
        except KeyError:
            valid = ", ".join(self._endpoints) or "(none configured)"
            raise KeyError(f"No endpoint named {name!r}; configured endpoints: {valid}") from None

    def upsert(self, name: str, cfg: EndpointConfig) -> None:
        _validate_history_fetch(cfg.history_fetch)
        self._endpoints[name] = cfg
        self._write()

    def delete(self, name: str) -> None:
        self.get(name)  # KeyError with a clear message if absent
        del self._endpoints[name]
        if self._active == name:
            self._active = None
        self._write()

    def active_name(self) -> str | None:
        return self._active

    def active_config(self) -> EndpointConfig | None:
        if self._active is None:
            return None
        return self._endpoints[self._active]

    def set_active(self, name: str) -> None:
        self.get(name)  # validates existence first
        self._active = name
        self._write()
