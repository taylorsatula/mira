"""MIRA terminal chat client: a streaming WebSocket chat UI over the deployed instance.

A pure client of the instance's HTTP and WebSocket surface — it imports
nothing from the server tree except ``utils/release_identity.py``, a
stdlib-only module ``update.py`` reaches so ``mira update`` compares releases
the same way the server's daily check does. The import rule is load-bearing
beyond tidiness: ``python3 -m ui.tui`` (the ``mira`` command) resolves this
package inside a venv built only from ``requirements.txt``, so a server import
here breaks the command on every install instead of failing at one call site.
"""
