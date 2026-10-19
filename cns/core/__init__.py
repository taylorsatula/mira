"""
The CNS domain package root: re-exports `Message`, `Continuum`,
`ContinuumState`, and `ContinuumEvent`.

Importing the root pulls the whole chain in eagerly and that chain reaches
outside the package — `message` imports `utils.timezone_utils`,
`utils.user_context`, and `clients.llm.types` — so importing this package is
not dependency-free. Nothing in the repo imports the root regardless; every
caller names a submodule (`from cns.core.message import Message`).

What the root does control is import order. A module-level import of
`cns.services` anywhere on the chain (`message`, `continuum`, `state`,
`events`) runs `cns/services/__init__.py`, which imports the orchestrator,
whose own `from cns.core.continuum import Continuum` re-enters this package
half-built and raises ImportError at boot. Keep such imports function-local.
"""

from .message import Message
from .continuum import Continuum
from .state import ContinuumState
from .events import ContinuumEvent

__all__ = [
    'Message',
    'Continuum',
    'ContinuumState',
    'ContinuumEvent'
]