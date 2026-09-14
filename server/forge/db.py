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


def reset_engine() -> None:
    # tests switch db files mid-run
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
