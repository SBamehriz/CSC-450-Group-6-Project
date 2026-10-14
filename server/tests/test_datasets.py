import hashlib
import io
import json
import uuid
from pathlib import Path
from zipfile import ZipFile

from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from alembic import command
from forge.models import Base, Bucket, Dataset, Document


def _bucket(client, name="source"):
    response = client.post("/api/buckets", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload(client, bucket_id, name, text):
    response = client.post(
        f"/api/buckets/{bucket_id}/documents",
        files={"files": (name, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["parsed"] == 1, response.text
    return response.json()["outcomes"][0]["document_id"]


def _create(client, bucket_id, name="frozen", **extra):
    return client.post(
        "/api/datasets",
        json={"name": name, "bucket_id": bucket_id, **extra},
    )



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
    assert ds_data["token_count"] == 0
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
            artifact_uri="datasets/persisted-v1.jsonl",
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


def test_snapshot_freezes_text_and_split_after_source_is_deleted(client):
    bucket = _bucket(client)
    first = _upload(client, bucket, "one.txt", "First source text. " * 30)
    second = _upload(client, bucket, "two.txt", "Second source text. " * 30)
    created = _create(client, bucket, validation_fraction=0.5, seed=17)
    assert created.status_code == 201, created.text
    dataset = created.json()
    assert dataset["status"] == "ready"
    assert dataset["doc_count"] == 2
    assert dataset["token_count"] == 0
    assert dataset["config"]["train_documents"] == 1
    assert dataset["config"]["validation_documents"] == 1

    response = client.get(f"/api/datasets/{dataset['id']}/download")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"train.jsonl", "validation.jsonl", "manifest.json"}
        train = [json.loads(line) for line in archive.read("train.jsonl").splitlines()]
        validation = [json.loads(line) for line in archive.read("validation.jsonl").splitlines()]
        manifest = json.loads(archive.read("manifest.json"))
    assert {train[0]["id"], validation[0]["id"]} == {first, second}
    assert manifest["format"] == "forge-text-snapshot-v1"
    assert manifest["documents"] == {"train": 1, "validation": 1}
    assert {source["id"] for source in manifest["sources"]} == {first, second}
    assert all(len(source["hash"]) == 64 for source in manifest["sources"])

    assert client.delete(f"/api/documents/{first}").status_code == 204
    assert client.get(f"/api/datasets/{dataset['id']}/download").content == response.content
    listed = client.get("/api/datasets?limit=1&offset=0").json()
    assert listed["total"] == 1 and listed["items"][0]["id"] == dataset["id"]
    assert client.get(f"/api/datasets/{dataset['id']}").json()["id"] == dataset["id"]


def test_same_seed_repeats_the_document_split(client):
    bucket = _bucket(client)
    for index in range(10):
        _upload(client, bucket, f"{index}.txt", f"Distinct text number {index}. " * 30)

    snapshots = []
    for name in ("first", "second"):
        response = _create(client, bucket, name=name, validation_fraction=0.3, seed=123)
        assert response.status_code == 201, response.text
        archive_response = client.get(f"/api/datasets/{response.json()['id']}/download")
        with ZipFile(io.BytesIO(archive_response.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            snapshots.append(
                (
                    {
                        (source["id"], source["hash"], source["split"])
                        for source in manifest["sources"]
                    },
                    archive.read("train.jsonl"),
                    archive.read("validation.jsonl"),
                )
            )
    assert snapshots[0] == snapshots[1]


def test_snapshot_validates_inputs_and_stored_source(client, data_dir):
    bucket = _bucket(client)
    assert _create(client, bucket).status_code == 422
    _upload(client, bucket, "one.txt", "One " * 130)
    assert _create(client, bucket).status_code == 422
    _upload(client, bucket, "two.txt", "Two " * 130)
    assert _create(client, bucket, validation_fraction=0).status_code == 422
    assert _create(client, bucket, name="   ").status_code == 422
    assert _create(client, "00000000-0000-0000-0000-000000000001").status_code == 404

    document = client.get(f"/api/documents?bucket_id={bucket}").json()["items"][0]
    before = set((data_dir / "datasets").glob("*.zip"))
    source = data_dir / "files" / "text" / f"{document['id']}.txt"
    source.write_text("changed", encoding="utf-8")
    response = _create(client, bucket)
    assert response.status_code == 409 and response.json()["error"]["code"] == "source_changed"
    assert client.get("/api/datasets").json()["total"] == 0
    assert set((data_dir / "datasets").glob("*.zip")) == before


def test_archive_check_duplicate_and_delete(client, data_dir, make_run):
    bucket = _bucket(client)
    _upload(client, bucket, "one.txt", "First " * 130)
    _upload(client, bucket, "two.txt", "Second " * 130)
    created = _create(client, bucket)
    assert created.status_code == 201, created.text
    dataset = created.json()
    assert _create(client, bucket).status_code == 409
    path = data_dir / dataset["artifact_uri"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == dataset["config"]["sha256"]

    run = make_run(dataset_id=uuid.UUID(dataset["id"]))
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 409
    from sqlalchemy.orm import Session

    from forge.db import get_engine
    from forge.models import Run

    with Session(get_engine()) as session:
        session.delete(session.get(Run, run.id))
        session.commit()
    path.write_bytes(path.read_bytes() + b"changed")
    response = client.get(f"/api/datasets/{dataset['id']}/download")
    assert response.status_code == 409 and response.json()["error"]["code"] == "artifact_changed"
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 204
    assert not path.exists()
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 404


def test_snapshot_deleted_cleaned_text_file(client, data_dir):
    """Verify HTTP 409 source_changed response when a cleaned text file is deleted."""
    bucket = _bucket(client)
    _upload(client, bucket, "doc1.txt", "Content for doc1. " * 30)
    _upload(client, bucket, "doc2.txt", "Content for doc2. " * 30)
    document = client.get(f"/api/documents?bucket_id={bucket}").json()["items"][0]
    source = data_dir / "files" / "text" / f"{document['id']}.txt"
    assert source.is_file()
    source.unlink()
    response = _create(client, bucket)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "source_changed"
    assert "is missing." in error["message"]


def test_snapshot_missing_cleaned_text_path(client, db, make_document):
    """Verify HTTP 409 source_changed response when doc.text_uri points to a nonexistent path."""
    bkt_res = client.post("/api/buckets", json={"name": f"bkt-{uuid.uuid4().hex[:6]}"})
    bucket_id = uuid.UUID(bkt_res.json()["id"])
    doc1 = make_document(bucket_id=bucket_id, chars=500, status="parsed")
    make_document(bucket_id=bucket_id, chars=500, status="parsed")
    doc1.text_uri = "text/nonexistent_path_doc.txt"
    db.commit()
    response = _create(client, str(bucket_id))
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "source_changed"
    assert "is missing." in error["message"]


