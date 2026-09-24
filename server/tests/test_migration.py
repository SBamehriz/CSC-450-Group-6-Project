import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from forge.models import Base


def _describe(url: str) -> dict[str, dict[str, tuple[str, bool]]]:
    engine = create_engine(url, future=True)
    inspector = inspect(engine)
    described = {}
    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        described[table] = {
            column["name"]: (str(column["type"]), bool(column["nullable"]))
            for column in inspector.get_columns(table)
        }
    engine.dispose()
    return described


@pytest.fixture()
def both_ways(tmp_path):
    migrated_url = f"sqlite:///{tmp_path / 'migrated.db'}"
    models_url = f"sqlite:///{tmp_path / 'from-models.db'}"

    config = Config("alembic.ini")
    config.cmd_opts = type("Opts", (), {"x": [f"db_url={migrated_url}"]})()  # type: ignore[assignment]
    command.upgrade(config, "head")

    engine = create_engine(models_url, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()

    return _describe(migrated_url), _describe(models_url), config, migrated_url


def test_migration_builds_the_same_tables_as_the_models(both_ways):
    migrated, from_models, _, _ = both_ways
    assert set(migrated) == set(from_models), (
        f"tables differ: migration={sorted(migrated)} models={sorted(from_models)}"
    )


def test_migration_builds_the_same_columns_and_types(both_ways):
    migrated, from_models, _, _ = both_ways
    for table in sorted(from_models):
        assert migrated[table] == from_models[table], f"{table} differs from the models"


def test_created_at_is_a_timestamp_and_required(both_ways):
    migrated, _, _, _ = both_ways
    for table in migrated:
        type_name, nullable = migrated[table]["created_at"]
        assert "DATETIME" in type_name.upper() or "TIMESTAMP" in type_name.upper()
        assert nullable is False


def test_migration_downgrades_cleanly(both_ways):
    _, _, config, migrated_url = both_ways
    command.downgrade(config, "base")
    left_over = set(_describe(migrated_url))
    assert left_over == set()
