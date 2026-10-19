"""
MIRA's TOAST log level: 60, one step above CRITICAL, for messages that must
survive any filter a deployment configures.

Imports nothing but `logging`, so it is safe to load before any application
import. deploy/python.sh and deploy/update.sh copy this file into the venv's
site-packages beside a `mira-log-levels.pth` hook, which imports it at
interpreter startup — that ordering is the point: `Logger.toast` has to exist
before the first module logs. The `hasattr(logging.Logger, 'toast')` guard
makes registration idempotent, so the `.pth` hook followed by an ordinary
`import _mira_log_levels` cannot redefine the level; utils/logging_config.py
carries its own copy of this registration for a checkout where the hook was
never installed. A process that never loaded either raises AttributeError on
`.toast()`.
"""

import logging

TOAST = 60

if not hasattr(logging.Logger, 'toast'):
    logging.addLevelName(TOAST, "TOAST")

    def _toast(self, message, *args, **kwargs):
        if self.isEnabledFor(TOAST):
            self._log(TOAST, message, args, **kwargs)

    logging.Logger.toast = _toast
