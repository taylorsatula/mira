"""The tool framework package: no re-exports and no import-time side effects.

Import ``tools.repo`` and ``tools.registry`` directly. Importing ``tools``
alone registers no config class, constructs no repository, and discovers no
tool.
"""
