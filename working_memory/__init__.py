"""Package root re-exporting `WorkingMemory` for the integration factory.

`cns/integration/factory.py:_get_working_memory` imports `from working_memory
import WorkingMemory`, while every other caller imports `working_memory.core`
by path. This re-export is the factory's only route to the class: dropping it
or renaming the class fails the factory import at boot, before the server
binds.
"""
from .core import WorkingMemory

__all__ = ['WorkingMemory']