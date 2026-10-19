"""Package marker for ``utils``; deliberately empty.

Nothing is re-exported here. Every module in this directory is imported by its
full path (``from utils.user_context import ...``), so an import added to this
file would load that module — and, for several of them, its configuration or
Vault-backed singletons — for every importer of anything here.
"""
