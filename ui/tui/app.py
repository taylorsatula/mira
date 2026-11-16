"""The MIRA terminal chat client: entry point of the streaming chat app.

``run(store)`` resolves the active endpoint, connects the WebSocket BEFORE any
UI exists (so auth and reachability failures are plain printed errors with
guidance, not a half-drawn screen), then hands the live connection to
``ChatSession``, which owns the screen, the turn lifecycle and reconnects.

With no endpoint configured anywhere in the store (the first run), ``run``
defaults to the local install: it mints a token against
``http://localhost:1993`` via the same headless chain as ``--login``, stores it
as the ``local`` endpoint, and proceeds. Remote/thinclient installs are the
edge case and opt in with ``--login --base-url``.

Exit codes: 0 on a user quit; 1 when no usable endpoint is configured (and the
first-run local setup did not succeed), stdin or stdout is not a terminal, or
the connection cannot be established; 130 on Ctrl+C while connecting or during
the first-run mint. Failures print the real error and what to do about
it, keyed on the client's error code.
"""

from __future__ import annotations

import asyncio
import sys

from .chat import ChatSession
from .client import ClientError, MiraClient
from .endpoints import EndpointConfig, EndpointStore
from .screen import Screen

# Inbox capacity: keyboard intents and wire events share one queue. The client
# awaits put (backpressure); the Screen rings the bell on QueueFull.
INBOX_MAX = 1024

_RED = "\033[31m"
_RESET = "\033[0m"

LOCAL_BASE_URL = "http://localhost:1993"
LOCAL_ENDPOINT_NAME = "local"

_SETUP_GUIDE = """\
No usable endpoint is configured. If automatic setup against the local MIRA
at http://localhost:1993 was attempted, the error above says why it did not
complete.

To set up by hand:

1. Local install — make sure MIRA is running, then just run `mira` again.
   The first run mints and stores a token automatically.

2. Remote / thinclient — mint against your instance and store it:

  python3 -m ui.tui --login --base-url https://your-mira.example.com --save-as remote

3. Manual — create the config file by hand (needs an API token, see 4):

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

4. To mint the API token by hand, drive the same headless auth API chain
  option 2 uses (scripted end-to-end in deploy/vm/talktomira.sh):
  log in (auto local session, or emailed magic-link on multi-user), then
  POST /v1/auth/csrf, then POST /v1/auth/api-tokens — the raw token is
  shown once at mint time.
Then rerun: python3 -m ui.tui"""


def setup_guidance(store: EndpointStore) -> str:
    """Setup guidance naming the exact config path (printed when the
    first-run local setup did not produce a usable endpoint)."""
    return _SETUP_GUIDE.format(path=store.path)


def _first_run_setup(store: EndpointStore) -> EndpointConfig | None:
    """First run with no endpoint configured anywhere: default to the local
    install. Mints a token against ``LOCAL_BASE_URL`` with the same headless
    chain as ``--login``, stores it as the ``local`` endpoint, and makes it
    active. Returns the usable config, or None after printing why it failed
    (the caller then prints the setup guidance)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        # The mint may prompt (multi-user magic-link); without a terminal
        # there is no way to answer, so fail before touching the network.
        print(
            "error: no endpoint configured and no interactive terminal to set one up.\n"
            f"  Run `mira` in a terminal to auto-configure the local install, or mint a\n"
            f"  token with `python3 -m ui.tui --login --base-url URL`.",
            file=sys.stderr,
            flush=True,
        )
        return None
    print(f"No endpoint configured — setting up the local MIRA at {LOCAL_BASE_URL}…", flush=True)
    from .login import LoginError, mint_api_token

    try:
        token = mint_api_token(LOCAL_BASE_URL)
    except LoginError as error:
        print(f"first-run setup failed: {error}", flush=True)
        return None
    except EOFError:
        print("first-run setup aborted — input closed (EOF)", flush=True)
        return None

    new_config = EndpointConfig(base_url=LOCAL_BASE_URL, api_key=token)
    try:
        store.upsert(LOCAL_ENDPOINT_NAME, new_config)
        store.set_active(LOCAL_ENDPOINT_NAME)
    except OSError as error:
        print(
            f"WARNING: could not write the token to {store.path}: {error}\n"
            "Store it manually, then add an endpoint with this api_key. "
            "The raw token (shown once, last resort):\n"
            f"{token}",
            flush=True,
        )
        return None
    print(f"connected: endpoint {LOCAL_ENDPOINT_NAME!r} ({LOCAL_BASE_URL}) stored in {store.path}", flush=True)
    return new_config


def _connect_guidance(error: ClientError, config: EndpointConfig, store: EndpointStore) -> str:
    """What to do about one failed connect, keyed on the client's error code."""
    if error.code == "AUTH_FAILED":
        return (
            "the stored API key was rejected — it is missing, stale, or from a different instance.\n"
            "  Re-mint it automatically:  python3 -m ui.tui --login\n"
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
            client,
            Screen(inbox),
            inbox,
            store.active_name() or "default",
            config.base_url,
            config.history_fetch,
        )
        return await session.run()
    finally:
        await client.close()


def run(store: EndpointStore) -> int:
    """Entry point: 0 on a user quit, 1 on setup/connect failure or no TTY,
    130 on Ctrl+C while connecting or during the first-run mint."""
    try:
        config = store.active_config()
        if config is not None and (not config.base_url or not config.api_key):
            config = None
        if config is None and not store.names():
            # First run: no endpoint anywhere — default to the local install
            # instead of making the user mint a token by hand.
            config = _first_run_setup(store)
            if config is None:
                print(setup_guidance(store), flush=True)
                return 1
        if config is None:
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
        return asyncio.run(_chat(store, config))
    except KeyboardInterrupt:
        print("aborted", flush=True)
        return 130
