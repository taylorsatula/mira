"""Prompt template package: the template corpus, and the one sanctioned reader for it.

`load_prompt` is imported from `config.prompts.loader` and re-exported here, so
`from config.prompts import load_prompt` and
`from config.prompts.loader import load_prompt` reach the same function.

Importing this package reads no template file. Each consumer's own
`load_prompt()` call is what resolves its template, so a missing or emptied
template raises at that consumer's load site (construction, first call, or that
consumer module's own import) and never from importing this package. That is why
this module holds no template constants: a template read here would happen once
at import and its failure would arrive during application wiring rather than at
the consumer that needs it.
"""
from config.prompts.loader import load_prompt

__all__ = ["load_prompt"]
