"""The MIRA terminal chat client: entry point of the streaming chat app.

``run(store)`` resolves the active endpoint, connects the WebSocket BEFORE any
UI exists (so auth and reachability failures are plain printed errors with
guidance, not a half-drawn screen), then hands the live connection to
``ChatSession``, which owns the screen, the turn lifecycle and reconnects.

Exit codes: 0 on a user quit; 1 when no usable endpoint is configured, stdin or
stdout is not a terminal, or the connection cannot be established; 130 on
Ctrl+C while connecting. Failures print the real error and what to do about
it, keyed on the client's error code.
"""

from __future__ import annotations

import asyncio
import sys

from tui.chat import ChatSession
from tui.client import ClientError, MiraClient
from tui.endpoints import EndpointConfig, EndpointStore
from tui.screen import Screen

# Inbox capacity: keyboard intents and wire events share one queue. The client
# awaits put (backpressure); the Screen rings the bell on QueueFull.
INBOX_MAX = 1024

_RED = "\033[31m"
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

3. To mint the API token by hand, drive the same headless auth API chain
  option 1 uses (scripted end-to-end in deploy/vm/talktomira.sh):
  log in (auto local session, or emailed magic-link on multi-user), then
  POST /v0/auth/csrf, then POST /v0/auth/api-tokens — the raw token is
  shown once at mint time.
Then rerun: python3 -m tui"""


def setup_guidance(store: EndpointStore) -> str:
    """First-run setup guidance naming the exact config path."""
    return _SETUP_GUIDE.format(path=store.path)


def _connect_guidance(error: ClientError, config: EndpointConfig, store: EndpointStore) -> str:
    """What to do about one failed connect, keyed on the client's error code."""
    if error.code == "AUTH_FAILED":
        return (
            "the stored API key was rejected — it is missing, stale, or from a different instance.\n"
            "  Re-mint it automatically:  python3 -m tui --login\n"
            f"  Or paste a token into: {store.path}"
        )
    if error.code in ("AUTH_CONNECTION_FAILED", "AUTH_TIMEOUT"):
        return (
            f"cannot reach MIRA at {config.base_url}: {error.message}\n"
            "  Is the instance running and the address/tunnel up? Fix base_url in:\n"
            f"  {store.path}"
        )
    if error.code == "BAD_BASE_URL":
        return f"{error.message}\n  Fix base_url in:\n  {store.path}"
    return f"{error.message} [{error.code}]"


async def _chat(store: EndpointStore, config: EndpointConfig) -> int:
    inbox: asyncio.Queue = asyncio.Queue(maxsize=INBOX_MAX)
    client = MiraClient(config, inbox)
    try:
        # connect() is bounded by the client; the user can still Ctrl+C (SIGINT
        # is live here — the terminal is not in raw mode yet).
        await client.connect()
    except ClientError as error:
        print(f"{_RED}error: {_connect_guidance(error, config, store)}{_RESET}", flush=True)
        return 1
    try:
        session = ChatSession(
            client, Screen(inbox), inbox, store.active_name() or "default", config.base_url
        )
        return await session.run()
    finally:
        await client.close()


def run(store: EndpointStore) -> int:
    """Entry point: 0 on a user quit, 1 on setup/connect failure or no TTY,
    130 on Ctrl+C while connecting."""
    config = store.active_config()
    if config is None or not config.base_url or not config.api_key:
        print(setup_guidance(store), flush=True)
        return 1
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "error: the MIRA chat client needs an interactive terminal "
            "(stdin and stdout must both be a TTY).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    try:
        return asyncio.run(_chat(store, config))
    except KeyboardInterrupt:
        print("aborted", flush=True)
        return 130
