"""
Package marker for the API routing layer.

Nothing is defined or re-exported here: every endpoint module builds its own
``router`` and ``main.py`` imports those modules directly, so an import that
looks like it belongs in this file does not exist.
"""