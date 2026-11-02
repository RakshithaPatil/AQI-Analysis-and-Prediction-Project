"""
database/connection.py
=======================
Provides a reusable SQLAlchemy engine and session factory.

Usage:
    from database.connection import get_engine, get_session, test_connection

The engine is created once (singleton pattern) and reused across the app.
All credentials come from the environment / .env file – never hard-coded.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from config.logger import get_logger
from config.settings import get_db_url

logger = get_logger(__name__, log_file="database.log")

# ── Engine singleton ──────────────────────────────────────────────────────────
_engine = None


def get_engine():
    """
    Return the SQLAlchemy engine, creating it on first call.

    Connection pool settings:
      pool_size=5     – keep 5 connections open at all times
      max_overflow=10 – allow up to 10 extra connections under load
      pool_timeout=30 – wait up to 30s before raising TimeoutError
      pool_recycle=3600 – recycle connections after 1 hour (avoids stale connections)
    """
    global _engine
    if _engine is None:
        db_url = get_db_url()
        logger.info("Creating database engine (host masked for security)")
        _engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False,  # Set True to log all SQL statements (verbose)
        )
    return _engine


# ── Session factory ───────────────────────────────────────────────────────────
def get_session_factory() -> sessionmaker:
    """Return a configured sessionmaker bound to the engine."""
    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy Session and handles
    commit / rollback automatically.

    Usage:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Database session error – rolling back: %s", exc)
        raise
    finally:
        session.close()


# ── Health check ──────────────────────────────────────────────────────────────
def test_connection() -> bool:
    """
    Test the database connection by running a simple SELECT 1.

    Returns
    -------
    True if the connection succeeds, False otherwise.
    """
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        if result == 1:
            logger.info("Database connection test: OK")
            return True
        logger.error("Database connection test: unexpected result %s", result)
        return False
    except Exception as exc:
        logger.error("Database connection test FAILED: %s", exc)
        return False


def create_schema(schema_file: str | None = None) -> None:
    """
    Execute the schema.sql DDL file against the database.

    Parameters
    ----------
    schema_file : Path to the SQL file. Defaults to database/schema.sql
                  relative to the project root.
    """
    from pathlib import Path
    from config.settings import get_project_root

    if schema_file is None:
        schema_file = get_project_root() / "database" / "schema.sql"

    sql_text = Path(schema_file).read_text(encoding="utf-8")

    with get_engine().connect() as conn:
        # Execute the entire DDL as a single transaction
        with conn.begin():
            conn.execute(text(sql_text))

    logger.info("Schema created/verified from %s", schema_file)
