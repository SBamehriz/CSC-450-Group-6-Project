import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from alembic import command
from forge.models import Base, Bucket, Dataset, Document


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


def test_upgrade_from_previous_migration_preserves_existing_data(tmp_path: Path):
    """Verify upgrading from revision 0001 to head preserves pre-existing records."""
    db_file = tmp_path / "upgrade_preserve.db"
    db_url = f"sqlite:///{db_file}"

    config = Config("alembic.ini")
    config.cmd_opts = type("Opts", (), {"x": [f"db_url={db_url}"]})()  # type: ignore[assignment]

    # 1. Upgrade only to revision 0001
    command.upgrade(config, "0001")

    engine1 = create_engine(db_url, future=True)
    bucket_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(UTC)

    with Session(engine1) as s1:
        bucket = Bucket(
            id=bucket_id,
            name="legacy-bucket",
            description="Bucket created in revision 0001",
            created_at=now,
        )
        doc = Document(
            id=doc_id,
            bucket_id=bucket_id,
            filename="legacy_doc.txt",
            source_format="txt",
            source_note="Legacy document",
            raw_uri="raw/legacy.txt",
            status="parsed",
            char_count=120,
            word_count=25,
            quality={},
            created_at=now,
        )
        s1.add(bucket)
        s1.add(doc)
        s1.commit()
    engine1.dispose()

    # 2. Upgrade from revision 0001 to head (revision 0002)
    command.upgrade(config, "head")

    # 3. Open with a fresh engine and verify pre-existing records are intact
    engine2 = create_engine(db_url, future=True)
    with Session(engine2) as session:
        loaded_bucket = session.get(Bucket, bucket_id)
        assert loaded_bucket is not None
        assert loaded_bucket.name == "legacy-bucket"
        assert loaded_bucket.description == "Bucket created in revision 0001"

        loaded_doc = session.get(Document, doc_id)
        assert loaded_doc is not None
        assert loaded_doc.filename == "legacy_doc.txt"
        assert loaded_doc.bucket_id == bucket_id
        assert loaded_doc.char_count == 120

        # 4. Save and reload new dataset linked to the existing bucket
        ds_id = uuid.uuid4()
        dataset = Dataset(
            id=ds_id,
            name="migrated-dataset-linked",
            description="Dataset linked to pre-existing bucket",
            bucket_id=bucket_id,
            status="ready",
            doc_count=1,
            char_count=120,
            token_count=30,
            config={"migrated_from": "0001"},
            artifact_uri="datasets/migrated_linked.jsonl",
        )
        session.add(dataset)
        session.commit()

        reloaded_ds = session.get(Dataset, ds_id)
        assert reloaded_ds is not None
        assert reloaded_ds.bucket_id == bucket_id
        assert reloaded_ds.artifact_uri == "datasets/migrated_linked.jsonl"
        assert reloaded_ds.config["migrated_from"] == "0001"

    engine2.dispose()
