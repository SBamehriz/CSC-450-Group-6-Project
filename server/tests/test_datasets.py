import hashlib
import io
import json
import uuid
from zipfile import ZipFile


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
