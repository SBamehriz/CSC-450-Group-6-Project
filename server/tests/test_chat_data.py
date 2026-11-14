import io
import json
import uuid
from array import array
from zipfile import ZipFile

import pytest

from forge import chatformat, dataset_import, jobs
from forge.config import get_settings
from forge.models import Bucket, Document

pa = pytest.importorskip("pyarrow")
pytest.importorskip("huggingface_hub")
BpeTokenizer = pytest.importorskip("tokenizers").Tokenizer

from standin_hub import install, start  # noqa: E402


def _talk(n, good=True):
    turns = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"What is {n} plus one? Say it in words please."},
    ]
    if good:
        turns.append({"role": "assistant", "content": f"It is {n + 1}, one more than {n}."})
    return turns


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    hub = install(monkeypatch, tmp_path / "hub")
    talks = [_talk(n, good=n % 50 != 0) for n in range(500)]
    table = pa.table({"messages": talks, "source": ["made up"] * 500})
    hub.add("someone/talks", {"data/train-0.parquet": table}, sha="d" * 40)
    hub.add("someone/plain", {"data/train.parquet": pa.table({"text": ["Plain words."] * 3})})
    monkeypatch.setattr(dataset_import, "TARGET_FILES", 5)
    return hub


def _bucket(client, name):
    return client.post("/api/buckets", json={"name": name}).json()["id"]


def _import_talks(client, name="talks"):
    bucket_id = _bucket(client, name)
    response = start(client, bucket_id, dataset="someone/talks", text_column="messages")
    assert response.status_code == 201, response.text
    assert response.json()["payload"]["use"] == "chat"
    assert jobs.run_pending() == 1
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed", job
    return bucket_id, job


def test_a_messages_column_makes_a_chat_collection(client, db, hub):
    info = client.post("/api/buckets/huggingface/inspect", json={"dataset": "someone/talks"})
    config = info.json()["configs"][0]
    assert config["chat_columns"] == ["messages"] and config["text_columns"] == ["source"]

    bucket_id, job = _import_talks(client)
    assert job["progress"]["failed"] == 10 and job["progress"]["imported"] == 490
    assert client.get(f"/api/buckets/{bucket_id}").json()["kind"] == "chat"
    assert db.get(Bucket, uuid.UUID(bucket_id)).kind == "chat"
    documents = db.query(Document).filter(Document.bucket_id == uuid.UUID(bucket_id)).all()
    text = "".join(
        (get_settings().data_dir / "files" / d.text_uri).read_text(encoding="utf-8")
        for d in documents
    )
    assert text.count("<|assistant|>It is 2, one more than 1.<|end|>") == 1
    for mark in chatformat.SPECIAL_TOKENS:
        assert mark in text
    # each conversation in a file ends with the end-of-text mark
    first = (get_settings().data_dir / "files" / documents[0].text_uri).read_text(encoding="utf-8")
    assert first.count(f"<|end|>{chatformat.EOS}\n<|system|>") == first.count("<|system|>") - 1


def test_text_and_conversations_do_not_mix(client, hub):
    plain = _bucket(client, "plain")
    response = start(client, plain, dataset="someone/plain")
    assert response.status_code == 201
    jobs.run_pending()
    refused = start(client, plain, dataset="someone/talks", text_column="messages")
    assert refused.status_code == 409
    assert "already holds plain text" in refused.json()["error"]["message"]

    talks, _job = _import_talks(client)
    refused = start(client, talks, dataset="someone/plain")
    assert refused.status_code == 409
    assert "holds conversations" in refused.json()["error"]["message"]


def _upload_talks(client):
    bucket_id = _bucket(client, "uploaded")
    lines = "\n".join(json.dumps({"messages": _talk(n)}) for n in range(200))
    files = [("files", ("talks.jsonl", io.BytesIO(lines.encode()), "application/octet-stream"))]
    body = client.post(f"/api/buckets/{bucket_id}/documents", files=files).json()
    assert body["parsed"] == 200, body
    return bucket_id


@pytest.mark.parametrize("source", ["import", "upload"])
def test_a_chat_collection_goes_through_snapshot_vocabulary_and_build(
    client, data_dir, hub, source
):
    bucket_id = _import_talks(client)[0] if source == "import" else _upload_talks(client)
    snapshot = client.post("/api/datasets", json={"name": "talks", "bucket_id": bucket_id})
    assert snapshot.status_code == 201, snapshot.text
    snapshot = snapshot.json()
    assert snapshot["download_ready"]
    # the snapshot is its folder; no zip is written until someone downloads it
    folder = data_dir / "datasets"
    assert not (folder / f"{snapshot['id']}.zip").exists()
    check = client.get(f"/api/datasets/{snapshot['id']}/verify").json()
    assert check["valid"], check
    response = client.get(f"/api/datasets/{snapshot['id']}/download")
    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.infolist()[0].compress_type != 0
        rows = [json.loads(line) for line in archive.read("train.jsonl").splitlines()]
        assert "<|user|>" in rows[0]["text"]
    assert not list(folder.glob(f"{snapshot['id']}*.zip"))

    vocabulary = client.post(
        "/api/tokenizers", json={"name": "talk", "dataset_id": snapshot["id"], "vocab_size": 512}
    ).json()
    assert jobs.run_pending() == 1
    vocabulary = client.get(f"/api/tokenizers/{vocabulary['id']}").json()
    assert vocabulary["status"] == "ready", vocabulary
    words = BpeTokenizer.from_file(
        str(data_dir / "tokenizers" / vocabulary["id"] / "tokenizer.json")
    )
    for index, mark in enumerate(chatformat.SPECIAL_TOKENS):
        assert words.token_to_id(mark) == index
    assert words.encode("<|user|>Hi<|end|>").ids[0] == 2

    build = client.post(
        "/api/dataset-builds",
        json={"name": "talk", "tokenizer_id": vocabulary["id"], "dataset_id": snapshot["id"]},
    ).json()
    assert jobs.run_pending() == 1
    build = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert build["status"] == "ready", build
    ids = array("H")
    ids.frombytes((data_dir / "builds" / build["id"] / "train.bin").read_bytes())
    assert {0, 1, 2, 3, 4} <= set(ids)
