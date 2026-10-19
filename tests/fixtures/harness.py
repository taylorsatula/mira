"""Claim-free probe harness: run named checks, print a terminal marker, return the exit code.

`finish()` returns 0 only when every recorded check passed and at least one ran;
any FAIL, or a probe that recorded nothing at all, yields 1 — a vacuous probe
fails. The module holds no behavior assertion and touches no infrastructure: the
callable passed to `check` is the thing that touches reality. `check` catches
`Exception` only, so a `SystemExit` raised inside a check ends the process
instead of being recorded as a FAIL.
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable

_results: list[tuple[str, str, Any]] = []


def check(label: str, fn: Callable[[], Any]) -> Any:
    """Run `fn`; record and print PASS/FAIL with the raw outcome; return fn's value (None on failure).

    A raised exception is recorded as FAIL, never swallowed. A check that signals
    failure with `raise SystemExit` escapes this harness and ends the probe directly.
    """
    try:
        out = fn()
    except Exception as error:
        _results.append((label, "FAIL", f"{type(error).__name__}: {error}"))
        print(f"FAIL {label}: {type(error).__name__}: {error}")
        traceback.print_exc()
        return None
    _results.append((label, "PASS", out))
    print(f"PASS {label}: {_short(out)}")
    return out


def _short(value: Any, limit: int = 400) -> str:
    try:
        return json.dumps(value, default=str)[:limit]
    except Exception:
        return repr(value)[:limit]


def finish() -> int:
    """Print SUMMARY + terminal marker; return the process exit code (nonzero on any FAIL or zero checks)."""
    passed = sum(1 for _, status, _ in _results if status == "PASS")
    failed = sum(1 for _, status, _ in _results if status == "FAIL")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    ok = failed == 0 and bool(_results)
    print("PROBE PASSED" if ok else "PROBE FAILED")
    return 0 if ok else 1
