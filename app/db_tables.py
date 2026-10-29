"""Backward-compatible shim for database schema creation and migrations.

Canonical implementation: :mod:`app.database.schema`.
"""
from app.database.schema import create_tables

__all__ = ["create_tables"]
