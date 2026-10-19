"""
CNS Services - Application layer orchestration.

This package contains application services that coordinate domain objects
and infrastructure components. Services contain workflow logic but delegate
to domain objects for business rules.

`ContinuumOrchestrator` is re-exported for convenience only; no in-repo caller
uses the package-level import (the integration factory imports the submodule),
so a new service needs no entry here.
"""

from .orchestrator import ContinuumOrchestrator

__all__ = [
    'ContinuumOrchestrator',
]