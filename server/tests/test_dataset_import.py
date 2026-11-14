import json
import uuid

import pytest
from sqlalchemy import select

from forge import dataset_import, jobs
from forge.config import get_settings
from forge.models import Document, Job

pa = pytest.importorskip("pyarrow")
pytest.importorskip("huggingface_hub")

from standin_hub import SHA, install, start  # noqa: E402

CARD = {
    "configs": [
        {"config_name": "cosmo", "data_files": [{"split": "train", "path": "cosmo/train-*"}]},
        {"config_name": "other", "data_files": "other/*.parquet"},
    ]
}
FILES = [f"cosmo/train-0000{part}-of-00003.parquet" for part in range(3)]


def _stories(first, count):
    # the short title line repeats in every row: imports must keep it
    return [
        f"Title line\nStory {n} is about a small red boat." for n in range(first, first + count)
    ]


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    hub = install(monkeypatch, tmp_path / "hub")
    files = {
        path: pa.table({"text": _stories(part * 1500, 1500), "label": list(range(1500))})
        for part, path in enumerate(FILES)
    }
    files["other/a.parquet"] = pa.table({"text": ["Something else entirely."] * 3})
    files["README.md"] = "card"
    hub.add("someone/stories", files, card=CARD)
    return hub


def _bucket(client, name="stories"):
    return client.post("/api/buckets", json={"name": name}).json()["id"]


def _job(db, job_id):
    db.expire_all()
    return db.get(Job, uuid.UUID(str(job_id)))


def _documents(db, bucket_id):
    db.expire_all()
    return db.scalars(
        select(Document)
        .where(Document.bucket_id == uuid.UUID(str(bucket_id)))
        .order_by(Document.filename)
    ).all()


def _texts(db, bucket_id):
    root = get_settings().data_dir / "files"
    found = []
    for document in _documents(db, bucket_id):
        found += (root / document.text_uri).read_text(encoding="utf-8").split("\n\n")
    return found


def test_every_file_of_a_split_is_read_to_the_end(client, db, hub, monkeypatch):
    monkeypatch.setattr(dataset_import, "TARGET_FILES", 6)
    bucket_id = _bucket(client)
    link = "https://huggingface.co/datasets/someone/stories"
    response = start(client, bucket_id, dataset=link, config="cosmo")
    assert response.status_code == 201, response.text
    assert response.json()["payload"]["limit"] is None
    assert jobs.run_pending() == 1
    job = _job(db, response.json()["id"])
    assert job.status == "completed", job.error
    assert hub.downloads == FILES
    assert job.progress["processed"] == job.progress["imported"] == 4500
    assert job.result["revision"] == SHA
    assert _texts(db, bucket_id) == _stories(0, 4500)
    assert not (get_settings().data_dir / "imports" / str(job.id)).exists()


def test_a_limit_stops_early(client, db, hub):
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/stories", config="cosmo", limit=1600)
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "completed", job.error
    assert job.progress["processed"] == 1600
    assert _texts(db, bucket_id) == _stories(0, 1600)
    assert hub.downloads == FILES[:2]


def test_a_retry_carries_on_without_repeats(client, db, hub, monkeypatch):
    monkeypatch.setattr(dataset_import, "TARGET_FILES", 12)
    real = dataset_import.report
    calls = {"n": 0}

    def flaky(*args, **values):
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("The network went away.")
        real(*args, **values)

    monkeypatch.setattr(dataset_import, "report", flaky)
    hub.flaky[FILES[2]] = 2
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/stories", config="cosmo")
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "failed" and "The network went away." in job.error
    cursor = job.result["cursor"]
    assert cursor["file"] >= 1 and job.result["counts"]["imported"] > 0

    assert client.post(f"/api/jobs/{job.id}/retry").status_code == 200
    jobs.run_pending()
    job = _job(db, job.id)
    assert job.status == "completed", job.error
    assert _texts(db, bucket_id) == _stories(0, 4500)
    assert job.progress["processed"] == 4500 and job.progress["skipped"] == 0
    # the third file failed twice, then came on the third try
    assert hub.downloads.count(FILES[2]) == 1


def test_a_download_that_keeps_failing_says_why(client, db, hub):
    hub.flaky[FILES[0]] = dataset_import.TRIES
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/stories", config="cosmo")
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "failed"
    assert "after 4 tries" in job.error and "internet connection" in job.error


def test_a_column_without_text_is_refused(client, hub):
    bucket_id = _bucket(client)
    info = client.post("/api/buckets/huggingface/inspect", json={"dataset": "someone/stories"})
    configs = {one["name"]: one for one in info.json()["configs"]}
    assert configs["cosmo"]["text_columns"] == ["text"]
    assert configs["cosmo"]["chat_columns"] == []
    response = start(
        client, bucket_id, dataset="someone/stories", config="cosmo", text_column="label"
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Column 'label' holds neither text nor conversations. Pick one of: 'text'."
    )


def test_a_mostly_empty_column_fails_early(client, db, hub):
    values = [None if n % 10 else f"Row {n}" for n in range(5000)]
    hub.add("someone/sparse", {"data/train.parquet": pa.table({"text": values})}, sha="b" * 40)
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/sparse")
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "failed" and "Most of the first 2,000 values" in job.error
    assert job.progress["processed"] <= 2000


def test_json_lines_files_are_read(client, db, hub):
    lines = "\n".join(json.dumps({"text": f"Line {n}", "n": n}) for n in range(30))
    hub.add("someone/lines", {"data/train.jsonl": lines}, sha="a" * 40)
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/lines")
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "completed", job.error
    assert _texts(db, bucket_id) == [f"Line {n}" for n in range(30)]


def test_a_dataset_without_readable_files_uses_the_partial_copy(client, db, hub):
    hub.add("someone/csv", {"data/train.csv": "text\nhello\n"}, sha="e" * 40)
    hub.add_converted(
        "someone/csv", {"default/partial-train/0000.parquet": pa.table({"text": _stories(0, 50)})}
    )
    bucket_id = _bucket(client)
    response = start(client, bucket_id, dataset="someone/csv")
    assert response.status_code == 201, response.text
    jobs.run_pending()
    job = _job(db, response.json()["id"])
    assert job.status == "completed", job.error
    assert "partial copy" in job.progress["message"]
    assert hub.downloads == ["default/partial-train/0000.parquet"]


def test_importing_the_same_rows_twice_adds_nothing(client, db, hub):
    bucket_id = _bucket(client)
    for _ in range(2):
        start(client, bucket_id, dataset="someone/stories", config="other")
        jobs.run_pending()
    second = db.scalars(select(Job).order_by(Job.created_at.desc())).first()
    assert second.status == "failed" and "already in this collection" in second.error


def test_one_copy_per_imported_file_on_disk(client, db, hub, monkeypatch):
    monkeypatch.setattr(dataset_import, "TARGET_FILES", 6)
    root = get_settings().data_dir / "files"
    before = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    bucket_id = _bucket(client)
    start(client, bucket_id, dataset="someone/stories", config="cosmo")
    jobs.run_pending()
    documents = _documents(db, bucket_id)
    assert len(documents) == 6
    assert all(d.raw_uri == d.text_uri for d in documents)
    after = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert after - before == {d.text_uri for d in documents}
    # deleting one removes its one file once, without an error
    assert client.delete(f"/api/documents/{documents[0].id}").status_code == 204
    assert not (root / documents[0].text_uri).exists()
    assert len(_documents(db, bucket_id)) == 5


def test_column_use_reads_every_way_of_writing_a_column():
    turn = {"role": {"dtype": "string"}, "content": {"dtype": "string"}}
    assert dataset_import.column_use({"dtype": "large_string", "_type": "Value"}) == "text"
    assert dataset_import.column_use("string") == "text"
    assert dataset_import.column_use([turn]) == "chat"
    assert dataset_import.column_use({"_type": "List", "feature": turn}) == "chat"
    assert dataset_import.column_use({"_type": "Sequence", "feature": turn}) == "chat"
    assert dataset_import.column_use({"dtype": "int64", "_type": "Value"}) is None
    assert dataset_import.column_use({"_type": "List", "feature": {"dtype": "string"}}) is None
