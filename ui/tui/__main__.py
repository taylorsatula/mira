"""Entry point for ``python3 -m ui.tui`` (the ``mira`` command on installs) — the MIRA terminal chat client.

CLI:
- ``update``            subcommand: update the deployed install in place
  (resolve the newest published release, rebuild
  changed venvs, swap code, restart; exit 0/1/3). ``UPDATE_FLAGS``:
  ``--nightly`` updates to the repo's main HEAD instead;
  ``--rebuild-venvs`` forces both venvs to rebuild; ``--adopt-drifted``
  lets the one-time database handover proceed on a database whose
  structure matches no known release. Machine half:
  ``deploy/update.sh``.
- ``status``            subcommand: the install's last official install or
  update and every change since, plus where its database stands
  (``ui/tui/migrate.py``).
- ``migrate [...]`` / ``restore <STAMP>`` / ``backup [--output DIR]``  the
  database step runner (``scripts/migrate.py``) on the install's server venv,
  arguments and exit code passed through unchanged; ``backup`` writes a full
  backup of the install (database, Vault credentials, settings, service
  config, program files, per-user tool data) into a directory you choose —
  nothing on the install is touched.
- These subcommands run before the chat client and need no endpoint store.
- ``--config PATH``   override the endpoint-store path (default
  ``~/.config/mira-tui/config.json``); needed for live verification runs.
- ``--login``         pre-run bootstrap: mint an API token against a MIRA
  instance over the auth HTTP surface (zero prompts on single-user
  instances; email magic-link on multi-user) and store it 0600. Targets
  the ACTIVE endpoint; ``--endpoint NAME`` or ``--base-url URL
  [--save-as NAME]`` pick the target. Prints a summary, exits 0/1.
- ``--chat``           with ``--login``: continue into the chat app after
  a successful mint (mint failures still exit 1) — the mint-and-chat
  one-liner.
- ``--config-debug``  print the resolved store path and, per endpoint,
  name / base_url / history_fetch / include_thinking — never the api_key —
  then exit 0.
- default: load the store and run the streaming chat app (WebSocket turns,
  prompt_toolkit input bar pinned under native scrollback, Rich-rendered
  transcript). First run with no usable endpoint prints setup guidance
  and exits 1.

Exit codes: 0 on a user quit (``/exit``, Ctrl+D on an empty box, a second
Ctrl+C within the quit window on an idle empty box); 1 on fatal startup
errors (unreadable or corrupt store,
no usable endpoint, not a terminal, connect or auth failure) with the real
message — fail loud, no silent defaults; 130 on Ctrl+C while connecting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import run as run_app
from .endpoints import EndpointStore

UPDATE_FLAGS = ("--nightly", "--rebuild-venvs", "--adopt-drifted")


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
    args_list = list(sys.argv[1:] if argv is None else argv)

    # Install-maintenance subcommands dispatch before argparse so the flag
    # surface of the chat client stays untouched. They run standalone: no
    # endpoint store is loaded, no chat client starts.
    if args_list and args_list[0] == "update":
        from .update import run_update

        flags = args_list[1:]
        unknown = [flag for flag in flags if flag not in UPDATE_FLAGS]
        if unknown:
            print(
                f"mira update: unknown option {unknown[0]} (supported: {', '.join(UPDATE_FLAGS)})",
                file=sys.stderr,
            )
            return 1
        return run_update(nightly="--nightly" in flags, rebuild_venvs="--rebuild-venvs" in flags,
                          adopt_drifted="--adopt-drifted" in flags)
    if args_list and args_list[0] in ("migrate", "restore", "backup"):
        from .migrate import run_runner

        return run_runner(args_list[0], args_list[1:])
    if args_list and args_list[0] == "status":
        from .migrate import run_status

        if len(args_list) > 1:
            print("mira status: takes no options", file=sys.stderr)
            return 1
        return run_status()

    parser = argparse.ArgumentParser(
        prog="mira",
        description="MIRA terminal chat client (streaming WebSocket chat with a bottom-pinned input bar over native scrollback). "
        "Subcommands: `mira update` updates a deployed install in place; "
        "`mira status` reports how the install differs from its last official update.",
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
        "--chat",
        action="store_true",
        help="With --login: continue into the chat app after a successful mint "
        "(mint failures still exit 1). Requires --login.",
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
    args = parser.parse_args(args_list)

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
        from .login import run_login

        rc = run_login(store, args.endpoint, args.base_url, args.save_as)
        if rc != 0 or not args.chat:
            return rc
        return run_app(store)

    if args.chat:
        print("fatal: --chat requires --login", file=sys.stderr)
        return 1

    if args.config_debug:
        _print_config_debug(store)
        return 0

    return run_app(store)


if __name__ == "__main__":
    sys.exit(main())
