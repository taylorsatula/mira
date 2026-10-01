"""Entry point for ``python3 -m tui`` — the MIRA terminal chat client.

CLI:
- ``--config PATH``   override the endpoint-store path (default
  ``~/.config/mira-tui/config.json``); needed for live verification runs.
- ``--login``         pre-run bootstrap: mint an API token against a MIRA
  instance over the auth HTTP surface (zero prompts on single-user
  instances; email magic-link on multi-user) and store it 0600. Targets
  the ACTIVE endpoint; ``--endpoint NAME`` or ``--base-url URL
  [--save-as NAME]`` pick the target. Prints a summary, exits 0/1.
- ``--config-debug``  print the resolved store path and, per endpoint,
  name / base_url / history_fetch / include_thinking — never the api_key —
  then exit 0.
- default: load the store and run the streaming chat app (WebSocket turns,
  prompt_toolkit input bar pinned under native scrollback, Rich-rendered
  transcript). First run with no usable endpoint prints setup guidance
  and exits 1.

Exit codes: 0 on a user quit (``/exit``, Ctrl+D on an empty box, Ctrl+C on
an idle empty box); 1 on fatal startup errors (unreadable or corrupt store,
no usable endpoint, not a terminal, connect or auth failure) with the real
message — fail loud, no silent defaults; 130 on Ctrl+C while connecting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tui.app import run as run_app
from tui.endpoints import EndpointStore


def _print_config_debug(store: EndpointStore) -> None:
    print(f"store path: {store.path}")
    names = store.names()
    if not names:
        print("endpoints: (none configured)")
        return
    for name in names:
        config = store.get(name)
        print(
            f"endpoint: {name}\n"
            f"  base_url: {config.base_url}\n"
            f"  history_fetch: {config.history_fetch}\n"
            f"  include_thinking: {config.include_thinking}"
        )
    print(f"active: {store.active_name() or '(none)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tui",
        description="MIRA terminal chat client (streaming WebSocket chat with a bottom-pinned input bar over native scrollback).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the endpoint store (default: ~/.config/mira-tui/config.json).",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Mint an API token headlessly (single-mode: auto-provisioned local session; "
        "multi-mode: email magic-link) and store it in the endpoint store, then exit.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="With --login: mint against this stored endpoint (default: the active one).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="With --login: mint against this URL instead of a stored endpoint.",
    )
    parser.add_argument(
        "--save-as",
        default=None,
        help="With --login --base-url: name for the new endpoint (default: 'default').",
    )
    parser.add_argument(
        "--config-debug",
        action="store_true",
        help="Print the resolved store path and per-endpoint config (never api_key), then exit.",
    )
    args = parser.parse_args(argv)

    store = EndpointStore(args.config)
    try:
        store.load()
    except (ValueError, OSError) as error:
        print(f"fatal: cannot load endpoint store at {store.path}: {error}", file=sys.stderr)
        return 1

    if args.login:
        if args.config_debug:
            print("fatal: --login and --config-debug are mutually exclusive", file=sys.stderr)
            return 1
        from tui.login import run_login

        return run_login(store, args.endpoint, args.base_url, args.save_as)

    if args.config_debug:
        _print_config_debug(store)
        return 0

    return run_app(store)


if __name__ == "__main__":
    sys.exit(main())
