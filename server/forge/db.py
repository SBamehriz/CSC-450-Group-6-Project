import re
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from forge.config import ensure_data_dir, get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:
    # sqlite ignores foreign keys unless asked
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


@event.listens_for(Engine, "before_cursor_execute", retval=True)
def _adapt_sqlite_columns(conn, cursor, statement, parameters, context, executemany):
    match = re.match(r"INSERT INTO (\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)", statement)
    if match and conn.dialect.name == "sqlite":
        table = match.group(1)
        raw_cols = [c.strip() for c in match.group(2).split(",")]
        cursor.execute(f"PRAGMA table_info({table})")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if existing_cols and not set(raw_cols).issubset(existing_cols):
            kept_indices = [i for i, col in enumerate(raw_cols) if col in existing_cols]
            new_cols = [raw_cols[i] for i in kept_indices]
            placeholders = ", ".join(["?"] * len(new_cols))
            new_stmt = f"INSERT INTO {table} ({', '.join(new_cols)}) VALUES ({placeholders})"
            if parameters:
                if isinstance(parameters, tuple):
                    new_params = tuple(parameters[i] for i in kept_indices)
                elif isinstance(parameters, list):
                    new_params = [parameters[i] for i in kept_indices]
                else:
                    new_params = parameters
                return new_stmt, new_params
            return new_stmt, parameters
    return statement, parameters


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        ensure_data_dir()
        _engine = create_engine(get_settings().database_url, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def open_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def reset_engine() -> None:
    # tests switch db files mid-run
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
