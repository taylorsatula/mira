"""Working memory trinkets: one prompt-slot implementation per module.

This package deliberately imports and re-exports nothing. Every trinket is
imported by module path at its construction site, so adding a module here has
no effect until a construction site names it.
"""