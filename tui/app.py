"""The MIRA TUI, minimal edition: a bare synchronous REPL.

One brick of the 2026-09-19 rebuild — deliberately low moving parts:
alternating cyan/green You/MIRA text blocks, a dark-grey delimiter row
between turns, and a plain ``input()`` line. No prompt_toolkit, no Rich
panels, no streaming preview, no reconnect machinery. The turn wire is
the synchronous REST endpoint ``POST /v0/api/chat`` (``cns/api/chat.py``):
one request per turn, the whole reply printed as one block when it
arrives. The retained WebSocket stack (``client.py`` + ``protocol.py``)
is intentionally unused here; it waits for the streaming brick.

Display filtering is delegated to ``tui.render.filter_system_tags`` (the
mirror of the web client's filters) so think blocks and ``<mira:...>``
tags never leak into a block.
"""

from __future__ import annotations

import shutil

import httpx

from tui.endpoints import EndpointConfig, EndpointStore
from tui.render import filter_system_tags

_CHAT_PATH = "/v0/api/chat"
# A full turn runs tools and multiple model rounds; this REST call returns
# only when the whole turn is done. Generous bound by design.
_CHAT_TIMEOUT = httpx.Timeout(300.0)
# Startup preflight: one authenticated history request. Short bound — a
# local/tunneled instance answers in well under a second.
_PREFLIGHT_TIMEOUT = httpx.Timeout(10.0)

# ANSI SGR codes — this client's entire chrome.
_CYAN = "\033[36m"   # user blocks
_GREEN = "\033[32m"  # MIRA blocks
_GREY = "\033[90m"   # turn delimiters (dark grey)
_RED = "\033[31m"    # errors
_DIM = "\033[2m"     # meta lines / thinking indicator
_RESET = "\033[0m"

_SETUP_GUIDE = """\
No endpoint is configured yet. Three ways to set up:

1. Fastest — automatic login (recommended):

  python3 -m tui --login

  Mints an API token against your MIRA instance directly — zero prompts on
  single-user instances (email magic-link on multi-user) — and stores it
  in the config file for you.

2. Manual — create the config file by hand (needs an API token, see 3):

  {path}

  Create it with this template (chmod 0600 is applied by the store):

  {{
    "active": "my-mira",
    "endpoints": {{
      "my-mira": {{
        "base_url": "http://localhost:1993",
        "api_key": "<your API token>",
        "history_fetch": "session_plus_summary",
        "include_thinking": true
      }}
    }}
  }}

3. To mint the API token however you can: MIRA web UI → Settings →
  API tokens → create; the raw token is shown once at mint time.
Then rerun: python3 -m tui"""


class ChatError(Exception):
    """One failed turn request — carries the real HTTP/server error text."""


# --- ANSI helpers -----------------------------------------------------------


def _color(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}"


def _term_width() -> int:
    return shutil.get_terminal_size().columns


def _delimiter() -> str:
    """The dark-grey row between turns, full terminal width."""
    return _color(_GREY, "─" * _term_width())


def _clear_last_line() -> None:
    """Erase the last printed/echoed line (typed input, thinking
    indicator). Single-visual-line assumption: a wrapped or pasted
    multi-line input leaves debris above the cleared row — accepted MVP
    limit."""
    print("\033[1A\033[2K", end="", flush=True)


def _block(label: str, color: str, text: str) -> None:
    """One colored block: sender label, then the message text."""
    print(_color(color, f"{label}\n{text}"), flush=True)


# --- the turn wire (sync REST) ----------------------------------------------


def _server_error_text(response: httpx.Response) -> str:
    """Best-effort server error text for a non-200 turn response."""
    try:
        error = response.json().get("error")
        if isinstance(error, dict) and error.get("message"):
            return f"[{error.get('code', '?')}] {error['message']}"
    except ValueError:
        pass
    return response.text.strip()[:300] or "(no body)"


def _post_chat(config: EndpointConfig, message: str) -> str:
    """One synchronous turn: POST /v0/api/chat, return the response text.

    Mirrors the envelope of ``cns/api/chat.py`` (BaseHandler
    SuccessResponse): ``{"success": true, "data": {"response": ...}}`` or
    ``{"success": false, "error": {"code", "message"}}``. Any drift from
    that shape fails loud with the real payload, never a default.
    """
    url = config.base_url.rstrip("/") + _CHAT_PATH
    try:
        response = httpx.post(
            url,
            json={"message": message},
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=_CHAT_TIMEOUT,
        )
    except httpx.HTTPError as error:
        raise ChatError(f"request to {url} failed: {error}") from error
    if response.status_code != 200:
        raise ChatError(f"HTTP {response.status_code}: {_server_error_text(response)}")
    try:
        payload = response.json()
    except ValueError as error:
        raise ChatError(f"non-JSON response: {response.text[:300]!r}") from error
    if payload.get("success") is not True:
        error = payload.get("error") or {}
        raise ChatError(
            f"[{error.get('code', 'UNKNOWN')}] "
            f"{error.get('message', '(no message in error envelope)')}"
        )
    reply = (payload.get("data") or {}).get("response")
    if not isinstance(reply, str):
        raise ChatError(f"envelope carried no response string: {str(payload)[:300]}")
    return reply


# --- startup preflight --------------------------------------------------------


def preflight(config: EndpointConfig, store_path: str) -> None:
    """Prove the endpoint reachable and the stored key accepted BEFORE the
    REPL opens: one authenticated history request (GET /v0/api/data,
    the same surface as client.py's pager). Raises ChatError whose message
    is the complete user-facing error + directions; never returns silently
    degraded. """
    url = config.base_url.rstrip("/") + "/v0/api/data"
    try:
        response = httpx.get(
            url,
            params={"type": "history", "limit": 1, "message_type": "regular"},
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=_PREFLIGHT_TIMEOUT,
        )
    except httpx.HTTPError as error:
        raise ChatError(
            f"cannot reach MIRA at {config.base_url} ({error})\n"
            "  Is the instance running and the address/tunnel up? Fix base_url in:\n"
            f"  {store_path}"
        ) from error
    if response.status_code in (401, 403):
        raise ChatError(
            f"the stored API key was rejected (HTTP {response.status_code}) — "
            "it is missing, stale, or from a different instance.\n"
            "  Re-mint it automatically:  python3 -m tui --login\n"
            f"  Or paste a fresh token (web UI → Settings → API tokens) into: {store_path}"
        )
    if response.status_code != 200:
        raise ChatError(
            f"MIRA at {config.base_url} answered HTTP {response.status_code}: "
            f"{_server_error_text(response)}\n"
            f"  Check the instance is healthy: {config.base_url}/health"
        )
    try:
        payload = response.json()
    except ValueError:
        raise ChatError(
            f"MIRA at {config.base_url} answered HTTP 200 with a non-JSON "
            f"body: {response.text.strip()[:200]}"
        ) from None
    if not isinstance(payload, dict):
        payload = {}  # a malformed envelope cannot claim success
    if payload.get("success") is not True:
        error = payload.get("error") or {}
        raise ChatError(
            f"MIRA at {config.base_url} rejected the request (HTTP 200 envelope, "
            f"success:false): [{error.get('code')}] {error.get('message')}\n"
            "  The stored API key may be missing, stale, or from a different instance.\n"
            "  Re-mint it automatically:  python3 -m tui --login\n"
            f"  Or paste a fresh token (web UI → Settings → API tokens) into: {store_path}"
        )


# --- the REPL ----------------------------------------------------------------


def setup_guidance(store: EndpointStore) -> str:
    """First-run setup guidance naming the exact config path."""
    return _SETUP_GUIDE.format(path=store.path)


def run(store: EndpointStore) -> int:
    """Entry point: the bare REPL. 0 on /exit, Ctrl+C at the prompt, or
    EOF; 1 when no usable endpoint is configured (or preflight fails);
    130 on Ctrl+C during preflight."""
    config = store.active_config()
    if config is None or not config.base_url or not config.api_key:
        print(setup_guidance(store), flush=True)
        return 1
    try:
        preflight(config, store.path)
    except ChatError as error:
        print(_color(_RED, f"error: {error}"), flush=True)
        return 1
    except KeyboardInterrupt:
        print("aborted during preflight", flush=True)
        return 130
    print(_color(_DIM, f"mira · {store.active_name()} · /exit to quit"), flush=True)
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()  # keep the shell prompt off the input row
            return 0
        _clear_last_line()
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            if stripped == "/exit":
                return 0
            print(_color(_DIM, f"unknown command: {stripped}"), flush=True)
            continue
        _block("You", _CYAN, filter_system_tags(stripped))
        print(_delimiter(), flush=True)
        print(_color(_DIM, "thinking…"), flush=True)
        try:
            reply = _post_chat(config, stripped)
        except ChatError as error:
            _clear_last_line()
            print(_color(_RED, f"error: {error}"), flush=True)
            print(_delimiter(), flush=True)
            continue
        except KeyboardInterrupt:
            # The request died client-side; the server turn may still run
            # to completion and the exchange lands in server history.
            _clear_last_line()
            print(
                _color(_DIM, "interrupted — the request was dropped client-side "
                "(the server turn may still complete)"),
                flush=True,
            )
            print(_delimiter(), flush=True)
            continue
        _clear_last_line()
        _block("MIRA", _GREEN, filter_system_tags(reply))
        print(_delimiter(), flush=True)
