"""Engine / session management and schema creation.

Everything that touches the database goes through `session_scope()` so commits
and rollbacks are handled in exactly one place.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _ensure_sqlite_parent_dir(url: str) -> None:
    """SQLite will not create missing directories for the database file."""
    if not url.startswith("sqlite"):
        return
    path_part = url.split("///", 1)[-1]
    if path_part and path_part != ":memory:":
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine, _SessionFactory
    if _engine is not None and url is None:
        return _engine

    db_url = url or settings.database_url
    _ensure_sqlite_parent_dir(db_url)

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, echo=echo, future=True, connect_args=connect_args)

    if db_url.startswith("sqlite"):
        # SQLite ignores FK constraints unless they are switched on per connection.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    if url is None:
        _engine = engine
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


def sync_columns(engine: Engine) -> list[str]:
    """Add columns that models declare but existing tables lack.

    `create_all` creates missing *tables* but never alters existing ones, so a
    new column on a model silently breaks inserts against a database created
    before it. Dropping the database instead would mean re-spending API quota to
    re-ingest, so missing columns are added in place.

    Deliberately additive only: no drops, renames or type changes. Anything
    beyond adding a column needs a real migration tool, and this is a
    development convenience, not Alembic.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            column_type = column.type.compile(dialect=engine.dialect)
            statement = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'
            try:
                with engine.begin() as connection:
                    connection.execute(text(statement))
                added.append(f"{table.name}.{column.name}")
            except SQLAlchemyError as exc:  # pragma: no cover - dialect dependent
                logger.warning("Could not add column %s.%s: %s", table.name, column.name, exc)

    if added:
        logger.info("Added missing columns: %s", ", ".join(added))
    return added


def init_db(url: str | None = None, echo: bool = False) -> Engine:
    """Create every table declared in models.py if it does not already exist."""
    engine = get_engine(url=url, echo=echo)
    Base.metadata.create_all(engine)
    sync_columns(engine)
    logger.info("Database schema ready at %s", engine.url.render_as_string(hide_password=True))
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop cached engine/session factory — used by tests to swap databases."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
