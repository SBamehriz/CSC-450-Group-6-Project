import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from forge.models import Base, Bucket, Dataset, Document


def test_create_dataset_without_bucket(client):
    res = client.post(
        "/api/datasets",
        json={"name": "raw-corpus-v1", "description": "initial uncurated dataset"},
    )
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["name"] == "raw-corpus-v1"
    assert data["description"] == "initial uncurated dataset"
    assert data["status"] == "draft"
    assert data["bucket_id"] is None
    assert data["doc_count"] == 0
    assert data["char_count"] == 0
    assert data["token_count"] == 0

    dataset_id = data["id"]
    get_res = client.get(f"/api/datasets/{dataset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "raw-corpus-v1"


def test_create_dataset_with_bucket_documents(client, db, make_document):
    bkt_res = client.post("/api/buckets", json={"name": f"bkt-{uuid.uuid4().hex[:6]}"})
    assert bkt_res.status_code == 201
    bucket_id = uuid.UUID(bkt_res.json()["id"])

    # Add 3 parsed documents (each 500 chars)
    make_document(bucket_id=bucket_id, chars=500, status="parsed")
    make_document(bucket_id=bucket_id, chars=500, status="parsed")
    make_document(bucket_id=bucket_id, chars=500, status="parsed")

    ds_res = client.post(
        "/api/datasets",
        json={
            "name": f"ds-{uuid.uuid4().hex[:6]}",
            "bucket_id": str(bucket_id),
            "description": "curated pretrain dataset",
            "config": {"val_split": 0.05, "tokenizer": "gpt2"},
        },
    )
    assert ds_res.status_code == 201, ds_res.text
    ds_data = ds_res.json()
    assert ds_data["status"] == "ready"
    assert ds_data["bucket_id"] == str(bucket_id)
    assert ds_data["doc_count"] == 3
    assert ds_data["char_count"] == 1500
    assert ds_data["token_count"] == 375
    assert ds_data["config"]["val_split"] == 0.05


def test_dataset_name_uniqueness(client):
    name = f"unique-ds-{uuid.uuid4().hex[:6]}"
    res1 = client.post("/api/datasets", json={"name": name})
    assert res1.status_code == 201

    res2 = client.post("/api/datasets", json={"name": name})
    assert res2.status_code == 409
    assert res2.json()["error"]["code"] == "name_taken"


def test_dataset_nonexistent_bucket(client):
    fake_id = str(uuid.uuid4())
    res = client.post("/api/datasets", json={"name": "orphan-ds", "bucket_id": fake_id})
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_list_datasets_and_filter(client):
    name_a = f"alpha-{uuid.uuid4().hex[:6]}"
    name_b = f"beta-{uuid.uuid4().hex[:6]}"
    client.post("/api/datasets", json={"name": name_a})
    client.post("/api/datasets", json={"name": name_b})

    res = client.get("/api/datasets?limit=100")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] >= 2
    names = [d["name"] for d in body["items"]]
    assert name_a in names
    assert name_b in names


def test_delete_dataset(client):
    res = client.post("/api/datasets", json={"name": f"temp-ds-{uuid.uuid4().hex[:6]}"})
    ds_id = res.json()["id"]

    del_res = client.delete(f"/api/datasets/{ds_id}")
    assert del_res.status_code == 204

    get_res = client.get(f"/api/datasets/{ds_id}")
    assert get_res.status_code == 404


def test_dataset_persistence_across_app_restart(tmp_path: Path):
    """Verify that dataset records persist on disk and survive engine restarts."""
    db_file = tmp_path / "persistent_test.db"
    db_url = f"sqlite:///{db_file}"

    # Step 1: Initial startup — create tables and insert dataset
    engine1 = create_engine(db_url, future=True)
    Base.metadata.create_all(engine1)

    ds_id = uuid.uuid4()
    with Session(engine1) as s1:
        dataset = Dataset(
            id=ds_id,
            name="persisted-dataset-v1",
            description="Stored for restart test",
            status="ready",
            doc_count=42,
            char_count=12000,
            token_count=3000,
            config={"split": "train"},
        )
        s1.add(dataset)
        s1.commit()

    # Dispose engine (simulate application termination / restart)
    engine1.dispose()

    # Step 2: New startup — connect with a fresh engine and retrieve dataset
    engine2 = create_engine(db_url, future=True)
    with Session(engine2) as s2:
        loaded = s2.get(Dataset, ds_id)
        assert loaded is not None
        assert loaded.name == "persisted-dataset-v1"
        assert loaded.description == "Stored for restart test"
        assert loaded.status == "ready"
        assert loaded.doc_count == 42
        assert loaded.char_count == 12000
        assert loaded.token_count == 3000
        assert loaded.config["split"] == "train"
        assert loaded.created_at is not None

    engine2.dispose()


def test_bucket_deletion_sets_dataset_bucket_to_null(db):
    """Verify foreign key constraint: deleting a bucket sets dataset.bucket_id to null."""
    bucket = Bucket(name=f"temp-bkt-{uuid.uuid4().hex[:6]}", description="temporary bucket")
    db.add(bucket)
    db.commit()

    dataset = Dataset(
        name=f"bkt-rel-ds-{uuid.uuid4().hex[:6]}",
        bucket_id=bucket.id,
        status="draft",
    )
    db.add(dataset)
    db.commit()

    assert dataset.bucket_id == bucket.id

    # Delete the bucket
    db.delete(bucket)
    db.commit()

    # Reload dataset
    db.refresh(dataset)
    assert dataset.bucket_id is None
    assert dataset.status == "draft"


def test_dataset_migration_schema_and_persistence(tmp_path: Path):
    """Verify that Alembic migration builds the dataset table and supports full persistence."""
    migrated_url = f"sqlite:///{tmp_path / 'migrated_ds.db'}"

    config = Config("alembic.ini")
    config.cmd_opts = type("Opts", (), {"x": [f"db_url={migrated_url}"]})()  # type: ignore[assignment]
    command.upgrade(config, "head")

    engine = create_engine(migrated_url, future=True)
    with Session(engine) as session:
        ds = Dataset(
            name="alembic-migrated-dataset",
            description="Verified via alembic upgrade head",
            status="ready",
            doc_count=100,
            char_count=50000,
            token_count=12500,
            config={"verified": True},
        )
        session.add(ds)
        session.commit()

        reloaded = session.execute(
            select(Dataset).where(Dataset.name == "alembic-migrated-dataset")
        ).scalar_one()
        assert reloaded.status == "ready"
        assert reloaded.char_count == 50000
        assert reloaded.token_count == 12500
        assert reloaded.config["verified"] is True

    engine.dispose()
