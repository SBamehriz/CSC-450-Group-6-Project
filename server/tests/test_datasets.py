import hashlib
import io
import json
import uuid
from pathlib import Path
from zipfile import ZipFile

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from alembic import command
from forge.models import Base, Bucket, Dataset, Document


def add_source(client, name="source"):
    bucket = client.post("/api/buckets", json={"name": name}).json()
    response = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            (
                "files",
                (
                    f"note-{index}.txt",
                    (f"Lesson {index}. Text for a real snapshot. " * 20).encode(),
                    "text/plain",
                ),
            )
            for index in range(4)
        ],
    )
    assert response.status_code == 201
    assert response.json()["parsed"] == 4
    return bucket


def create_snapshot(client, bucket, name="weekly", **settings):
    response = client.post(
        "/api/datasets", json={"name": name, "bucket_id": bucket["id"], **settings}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_requires_real_source(client):
    response = client.post("/api/datasets", json={"name": "no-source"})
    assert response.status_code == 422
    response = client.post(
        "/api/datasets", json={"name": "missing", "bucket_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404


def test_snapshot_archive_counts_and_hashes(client):
    bucket = add_source(client)
    dataset = create_snapshot(client, bucket, validation_fraction=0.25)
    assert dataset["status"] == "ready"
    assert dataset["download_ready"] is True
    assert dataset["doc_count"] == 4
    assert dataset["token_count"] == 0
    assert dataset["config"]["train_documents"] == 3
    assert dataset["config"]["validation_documents"] == 1
    response = client.get(f"/api/datasets/{dataset['id']}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert hashlib.sha256(response.content).hexdigest() == dataset["config"]["sha256"]
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"train.jsonl", "validation.jsonl", "manifest.json"}
        manifest = json.loads(archive.read("manifest.json"))
        for filename, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == digest
        train = [json.loads(line) for line in archive.read("train.jsonl").splitlines()]
        validation = [json.loads(line) for line in archive.read("validation.jsonl").splitlines()]
        assert not {row["id"] for row in train} & {row["id"] for row in validation}
        assert sum(len(row["text"]) for row in train + validation) == dataset["char_count"]
    check = client.get(f"/api/datasets/{dataset['id']}/verify").json()
    assert check["valid"] and check["train_documents"] == 3


def test_same_seed_same_split_and_frozen_after_source_delete(client):
    bucket = add_source(client)
    first = create_snapshot(client, bucket, "first", seed=31)
    second = create_snapshot(client, bucket, "second", seed=31)

    def download(dataset):
        return client.get(f"/api/datasets/{dataset['id']}/download").content

    original = download(first)
    with ZipFile(io.BytesIO(original)) as left, ZipFile(io.BytesIO(download(second))) as right:
        assert left.read("train.jsonl") == right.read("train.jsonl")
        assert left.read("validation.jsonl") == right.read("validation.jsonl")
    documents = client.get(f"/api/documents?bucket_id={bucket['id']}").json()["items"]
    for document in documents:
        assert client.delete(f"/api/documents/{document['id']}").status_code == 204
    assert client.delete(f"/api/buckets/{bucket['id']}").status_code == 204
    assert download(first) == original
    assert client.get(f"/api/datasets/{first['id']}").json()["bucket_id"] is None


def test_missing_or_changed_text_does_not_create_record(client, db, data_dir):
    bucket = add_source(client)
    document = db.scalar(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"])))
    path = data_dir / "files" / document.text_uri
    path.write_text("changed", encoding="utf-8")
    response = client.post("/api/datasets", json={"name": "bad", "bucket_id": bucket["id"]})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "changed_text"
    path.unlink()
    response = client.post("/api/datasets", json={"name": "missing", "bucket_id": bucket["id"]})
    assert response.status_code == 409
    assert client.get("/api/datasets").json()["total"] == 0


def test_archive_tamper_blocks_download_and_delete_removes_file(client, data_dir):
    dataset = create_snapshot(client, add_source(client))
    path = data_dir / dataset["artifact_uri"]
    path.write_bytes(path.read_bytes() + b"changed")
    assert not client.get(f"/api/datasets/{dataset['id']}").json()["download_ready"]
    assert client.get(f"/api/datasets/{dataset['id']}/verify").json()["valid"] is False
    assert client.get(f"/api/datasets/{dataset['id']}/download").status_code == 409
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 204
    assert not path.exists()
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 404


def test_old_metadata_only_record_is_not_downloadable(client, make_dataset):
    old = make_dataset(status="ready")
    response = client.get(f"/api/datasets/{old.id}").json()
    assert not response["download_ready"]
    assert response["readiness_note"]
    assert client.get(f"/api/datasets/{old.id}/download").status_code == 409


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": -1},
        {"validation_fraction": 0},
        {"validation_fraction": 0.8},
        {"config": {"val_split": 0.1}},
    ],
)
def test_invalid_or_old_settings_are_rejected(client, settings):
    bucket = add_source(client)
    response = client.post(
        "/api/datasets", json={"name": "invalid", "bucket_id": bucket["id"], **settings}
    )
    assert response.status_code == 422


def test_dataset_name_uniqueness_and_list_filter(client):
    bucket = add_source(client)
    create_snapshot(client, bucket)
    response = client.post("/api/datasets", json={"name": "weekly", "bucket_id": bucket["id"]})
    assert response.status_code == 409
    page = client.get(
        f"/api/datasets?bucket_id={bucket['id']}&status=ready&limit=1&offset=0"
    ).json()
    assert page["total"] == 1 and page["items"][0]["name"] == "weekly"


def test_small_or_pending_source_is_blocked(client, make_document):
    bucket = client.post("/api/buckets", json={"name": "empty"}).json()
    assert (
        client.post("/api/datasets", json={"name": "empty", "bucket_id": bucket["id"]}).status_code
        == 422
    )
    make_document(bucket_id=uuid.UUID(bucket["id"]), status="pending")
    assert (
        client.post(
            "/api/datasets", json={"name": "pending", "bucket_id": bucket["id"]}
        ).status_code
        == 409
    )


def test_dataset_persistence_across_app_restart(tmp_path: Path):
    """Verify that dataset records persist on disk and survive engine restarts."""
    db_file = tmp_path / "persistent_test.db"
    db_url = f"sqlite:///{db_file}"

    # Step 1: Initial startup  -  create tables and insert dataset
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
            artifact_uri="datasets/persisted-v1.jsonl",
        )
        s1.add(dataset)
        s1.commit()

    # Dispose engine (simulate application termination / restart)
    engine1.dispose()

    # Step 2: New startup  -  connect with a fresh engine and retrieve dataset
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
        assert loaded.artifact_uri == "datasets/persisted-v1.jsonl"
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


def test_dataset_deletion_preserves_bucket_and_documents(db, make_document):
    """Verify deleting a dataset preserves the underlying bucket and its documents."""
    bucket = Bucket(name=f"persist-bkt-{uuid.uuid4().hex[:6]}", description="preserved bucket")
    db.add(bucket)
    db.commit()

    doc = make_document(bucket_id=bucket.id, chars=500, status="parsed")

    dataset = Dataset(
        name=f"del-ds-{uuid.uuid4().hex[:6]}",
        bucket_id=bucket.id,
        status="ready",
        doc_count=1,
        char_count=500,
        token_count=125,
    )
    db.add(dataset)
    db.commit()
    ds_id = dataset.id

    # Delete the dataset
    db.delete(dataset)
    db.commit()

    # Verify dataset is deleted
    assert db.get(Dataset, ds_id) is None

    # Verify bucket and documents still exist intact
    loaded_bucket = db.get(Bucket, bucket.id)
    assert loaded_bucket is not None
    assert loaded_bucket.name == bucket.name

    loaded_doc = db.get(Document, doc.id)
    assert loaded_doc is not None
    assert loaded_doc.bucket_id == bucket.id


def test_dataset_deletion_sets_run_dataset_to_null(db):
    """Verify deleting a dataset sets run.dataset_id to null via foreign key rule."""
    from forge.models import Model, Run

    model = Model(
        name=f"test-model-{uuid.uuid4().hex[:6]}",
        n_layer=2,
        d_model=64,
        n_head=2,
        ctx_len=128,
        vocab_size=1000,
    )
    dataset = Dataset(
        name=f"run-ds-{uuid.uuid4().hex[:6]}",
        status="ready",
        doc_count=10,
        char_count=500,
        token_count=125,
    )
    db.add(model)
    db.add(dataset)
    db.commit()

    run = Run(
        name=f"test-run-{uuid.uuid4().hex[:6]}",
        model_id=model.id,
        dataset_id=dataset.id,
        status="pending",
    )
    db.add(run)
    db.commit()
    assert run.dataset_id == dataset.id

    # Delete dataset
    db.delete(dataset)
    db.commit()

    # Refresh run
    db.refresh(run)
    assert run.dataset_id is None
    assert run.status == "pending"
    assert db.get(Model, model.id) is not None


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
            artifact_uri="datasets/migrated_ds.jsonl",
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
        assert reloaded.artifact_uri == "datasets/migrated_ds.jsonl"

    engine.dispose()


def test_snapshot_missing_cleaned_text_path(client, db, make_document):
    """Verify HTTP 409 when doc.text_uri points to a nonexistent path."""
    bkt_res = client.post("/api/buckets", json={"name": f"bkt-{uuid.uuid4().hex[:6]}"})
    bucket_id = uuid.UUID(bkt_res.json()["id"])
    doc1 = make_document(bucket_id=bucket_id, chars=500, status="parsed")
    make_document(bucket_id=bucket_id, chars=500, status="parsed")
    doc1.text_uri = "text/nonexistent_path_doc.txt"
    db.commit()
    response = client.post(
        "/api/datasets", json={"name": "missing-path", "bucket_id": str(bucket_id)}
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert "missing" in error["message"].lower() or error["code"] == "missing_text"


def test_dataset_stores_training_validation_split(client):
    bucket = client.post("/api/buckets", json={"name": "split-source"}).json()
    response = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            (
                "files",
                (
                    f"lesson-{index}.txt",
                    (f"Lesson {index}. This is text for a saved training split. " * 20).encode(),
                    "text/plain",
                ),
            )
            for index in range(10)
        ],
    )
    assert response.status_code == 201
    assert response.json()["parsed"] == 10
    dataset = create_snapshot(client, bucket, validation_fraction=0.2, seed=42)
    assert dataset["doc_count"] == 10
    assert dataset["config"]["train_documents"] == 8
    assert dataset["config"]["validation_documents"] == 2
    assert dataset["config"]["seed"] == 42
    response = client.get(f"/api/datasets/{dataset['id']}/download")
    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        train = [json.loads(line) for line in archive.read("train.jsonl").splitlines()]
        validation = [json.loads(line) for line in archive.read("validation.jsonl").splitlines()]
        assert len(train) == 8
        assert len(validation) == 2
        assert not {row["id"] for row in train} & {row["id"] for row in validation}
