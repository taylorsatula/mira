"""
CNS - the package boundary over the five conversation-orchestration subsystems.

This file holds no imports and re-exports nothing, deliberately. Python
executes it before any `cns` submodule loads, so an import added here runs
for every consumer of every part of `cns`, including directories that want
one leaf type and nothing else: `lt_memory/processing/orchestrator.py`
imports `cns.core.message` and `cns.infrastructure.continuum_repository` on
its boot path, and `agents/spawn.py` imports `cns.core.events`. A transport
or orchestrator import in this file puts the request boundary on a
memory-layer or sidebar-agent import path that never asked for it, which is
the layering contract stated in this directory's AGENTS.md broken at its
root.

Consumers name a submodule by full path; there is no package-level
surface to extend.
"""
