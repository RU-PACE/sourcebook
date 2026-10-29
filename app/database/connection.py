"""PostgreSQL client and request-scoped connection dependency (Psycopg 3)."""

from collections.abc import Generator

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import DATABASE_URL, DB_POOL_MAX_SIZE, DB_POOL_MIN_SIZE

_pool: ConnectionPool | None = None


def open_db_pool() -> None:
    """Create the shared PostgreSQL connection pool once at application startup."""
    global _pool
    if _pool is not None:
        return

    _pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=DB_POOL_MIN_SIZE,
        max_size=DB_POOL_MAX_SIZE,
        kwargs={"row_factory": dict_row},
        open=False,
        name="stage1-postgres-pool",
    )
    _pool.open(wait=True)


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("PostgreSQL connection pool has not been opened")
    return _pool


def get_db() -> Generator:
    """Yield one pooled connection for a request; return it to the pool afterward."""
    with get_pool().connection() as connection:
        yield connection


def close_db_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
