import hashlib
import io
import json
import sys
import uuid
from array import array
from zipfile import ZipFile

from sqlalchemy import select
from test_datasets import add_source, create_snapshot

from forge.models import Dataset, Document, Job


def prepare(client, dataset, vocabulary=512):
    response = client.post(
        f"/api/datasets/{dataset['id']}/prepare", json={"vocab_size": vocabulary}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_prepared_archive_and_real_token_ids(client, db):
    dataset = create_snapshot(client, add_source(client))
    result = prepare(client, dataset)
    assert result["ready"] and 257 <= result["vocab_size"] <= 512
    assert result["train_tokens"] > 0 and result["validation_tokens"] > 0
    assert result["trained_on_documents"] == dataset["config"]["train_documents"]
    assert client.get(f"/api/datasets/{dataset['id']}/prepared").json()["ready"]
    response = client.get(f"/api/datasets/{dataset['id']}/prepared/download")
    assert response.status_code == 200
    assert hashlib.sha256(response.content).hexdigest() == result["sha256"]
    from tokenizers import Tokenizer

    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "train.bin",
            "validation.bin",
            "tokenizer.json",
            "manifest.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        tokenizer = Tokenizer.from_str(archive.read("tokenizer.json").decode())
        assert tokenizer.token_to_id("<|eos|>") == 0
        assert manifest["trained_on"] == "train"
        assert manifest["snapshot_sha256"] == dataset["config"]["sha256"]
        assert not {row["id"] for row in manifest["documents"]["train"]} & {
            row["id"] for row in manifest["documents"]["validation"]
        }
        snapshot = client.get(f"/api/datasets/{dataset['id']}/download")
        with ZipFile(io.BytesIO(snapshot.content)) as source:
            for split in ("train", "validation"):
                ids = array("H")
                ids.frombytes(archive.read(f"{split}.bin"))
                if sys.byteorder != "little":
                    ids.byteswap()
                assert len(ids) == result[f"{split}_tokens"]
                originals = [
                    json.loads(line) for line in source.read(f"{split}.jsonl").splitlines()
                ]
                for row, original in zip(manifest["documents"][split], originals, strict=True):
                    encoded = list(ids[row["offset"] : row["offset"] + row["tokens"]])
                    assert encoded[-1] == 0
                    assert (
                        encoded[:-1]
                        == tokenizer.encode(original["text"], add_special_tokens=False).ids
                    )
                assert all(0 <= value < result["vocab_size"] for value in ids)
    db.expire_all()
    record = db.get(Dataset, uuid.UUID(dataset["id"]))
    assert record.token_count == result["train_tokens"] + result["validation_tokens"]
    job = db.scalar(select(Job))
    assert job.status == "completed" and job.completed_at is not None


def test_preparation_uses_frozen_files_after_source_removed(client, db):
    dataset = create_snapshot(client, add_source(client))
    for document in db.scalars(select(Document)).all():
        assert client.delete(f"/api/documents/{document.id}").status_code == 204
    assert prepare(client, dataset)["ready"]
    text = "English and Arabic مرحبا 123"
    response = client.post(f"/api/datasets/{dataset['id']}/prepared/preview", json={"text": text})
    assert response.status_code == 200
    assert response.json()["token_count"] == len(response.json()["ids"])
    assert response.json()["pieces"]


def test_tampered_prepared_files_and_remove_keep_snapshot(client, data_dir):
    dataset = create_snapshot(client, add_source(client))
    prepare(client, dataset)
    path = data_dir / "prepared" / f"{dataset['id']}.zip"
    path.write_bytes(path.read_bytes() + b"changed")
    assert not client.get(f"/api/datasets/{dataset['id']}/prepared").json()["ready"]
    assert client.get(f"/api/datasets/{dataset['id']}/prepared/download").status_code == 409
    assert client.delete(f"/api/datasets/{dataset['id']}/prepared").status_code == 204
    assert not path.exists()
    assert client.get(f"/api/datasets/{dataset['id']}/download").status_code == 200
    assert prepare(client, dataset)["ready"]


def test_duplicate_prepare_and_invalid_vocabulary(client):
    dataset = create_snapshot(client, add_source(client))
    for vocabulary in (0, 256, 20000):
        assert (
            client.post(
                f"/api/datasets/{dataset['id']}/prepare", json={"vocab_size": vocabulary}
            ).status_code
            == 422
        )
    prepare(client, dataset)
    assert client.post(f"/api/datasets/{dataset['id']}/prepare", json={}).status_code == 409


def test_missing_tools_return_useful_error(client, monkeypatch):
    monkeypatch.setattr("forge.prepared_api.tools_available", lambda: False)
    assert not client.get("/api/preparation/tools").json()["available"]
    response = client.post(f"/api/datasets/{uuid.uuid4()}/prepare", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "tools_missing"


def test_failed_preparation_records_failure_and_can_retry(client, db, monkeypatch):
    dataset = create_snapshot(client, add_source(client))
    monkeypatch.setattr("forge.preparation.MAX_PREPARE_BYTES", 1)
    response = client.post(f"/api/datasets/{dataset['id']}/prepare", json={})
    assert response.status_code == 413
    assert db.scalar(select(Job)).status == "failed"
    monkeypatch.setattr("forge.preparation.MAX_PREPARE_BYTES", 10 * 1024 * 1024)
    assert prepare(client, dataset)["ready"]


def test_delete_snapshot_also_removes_prepared_archive(client, data_dir):
    dataset = create_snapshot(client, add_source(client))
    prepare(client, dataset)
    path = data_dir / "prepared" / f"{dataset['id']}.zip"
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 204
    assert not path.exists()
