"""Package entry point for the `auth` tree; re-exports the credential dependency.

Every consumer imports a submodule directly — `from auth.api import
get_current_user`, `from auth.mode import auth_mode` — and nothing imports a name
from the package root, so this file's only job is the convenience re-export.

That has a cost worth knowing before adding an import here: Python runs a
package's `__init__` before any of its submodules, so importing `auth.mode`
transitively loads `auth.api`, `auth.service`, `auth.database` and their clients.
A module that wants one cheap constant pays for the whole stack.
"""

from auth.api import get_current_user

__all__ = ["get_current_user"]
