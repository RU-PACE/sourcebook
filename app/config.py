"""Backward-compatible exports for application configuration.

The canonical environment configuration now lives in :mod:`app.core.config`.
Keep this import path stable for existing Sourcebook modules and integrations.
"""
from app.core.config import *  # noqa: F401,F403
from app.core.config import __all__
