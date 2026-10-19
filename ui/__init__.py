"""MIRA's two chat clients under one package: the terminal client (``ui.tui``, run as ``python3 -m ui.tui``) and the browser client (``ui/browser/``, static files the server serves).

This module stays import-free. ``python3 -m ui.tui`` loads it before any of the
terminal client's own modules, inside the TUI venv built from
``ui/tui/requirements.txt`` — a pin set that overlaps the server's
``requirements.txt`` on ``httpx`` and ``pydantic`` only, and carries neither the
server's runtime stack nor its optional dependencies. An import added here
therefore executes on every launch of the ``mira`` command in an environment
that cannot satisfy it, and it fails during package resolution — before any
client module loads, so the traceback names this file rather than the feature
that wanted the import.
"""
