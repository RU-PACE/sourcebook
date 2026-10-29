"""Backward-compatible shim for the PostgreSQL connection pool.

Canonical implementation: :mod:`app.database.connection`.
"""
from app.database.connection import close_db_pool, get_db, get_pool, open_db_pool

__all__ = ["close_db_pool", "get_db", "get_pool", "open_db_pool"]
