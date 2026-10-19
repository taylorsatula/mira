"""Operator entry point for the post-server power-on self-test checks.

A shim that holds no logic of its own. ``utils.power_on_self_test.post_server_cli``
owns the arguments, the checks, and the verdict; this module exists so the
post-server phase has a stable command name (``python -m
scripts.post_server_post``) that does not move when that module is reshaped.
The process exits with the CLI's status: 0 when every required check passed,
1 otherwise.
"""

from utils.power_on_self_test import post_server_cli


if __name__ == "__main__":
    raise SystemExit(post_server_cli())
