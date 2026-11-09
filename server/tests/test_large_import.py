import uuid
from email.message import Message

import pytest
from sqlalchemy import select, update

from forge import buckets, dataset_import, jobs
from forge.models import Document, Job

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROWS = 120_000


@pytest.fixture(autouse=True)
def offline_hub(monkeypatch, tmp_path):
    """Two Parquet files served as if they were the hub's copy of a split."""
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    monkeypatch.setattr(buckets, "validate_import_params", lambda *args: None)
    monkeypatch.setattr(dataset_import, "resolve_dataset", lambda name: (name, "c0ffee" * 7))
    monkeypatch.setattr(dataset_import, "inspect_dataset", _no_schema)
    monkeypatch.setattr(dataset_import, "TARGET_FILES", 6)
    stories: list[str | None] = [
        f"Once upon a time story {index} had a small red boat." for index in range(ROWS)
    ]
    stories[10], stories[11], stories[12] = "   ", None, stories[13]
    files = {}
    for part, rows in enumerate((stories[: ROWS // 2], stories[ROWS // 2 :])):
        path = tmp_path / f"hub-{part}.parquet"
        pq.write_table(pa.table({"text": rows, "label": [0] * len(rows)}), path)
        files[f"https://hub/{part}.parquet"] = path
    monkeypatch.setattr(dataset_import, "parquet_urls", lambda *args: list(files))
    monkeypatch.setattr(
        dataset_import, "download", lambda url, path: path.write_bytes(files[url].read_bytes())
    )


def _no_schema(name):
    raise RuntimeError("offline")


def start(client, limit):
    bucket = client.post("/api/buckets", json={"name": "stories"}).json()
    response = client.post(
        f"/api/buckets/{bucket['id']}/import",
        json={
            "dataset": "someone/stories",
            "config": "default",
            "split": "train",
            "text_column": "text",
            "limit": limit,
        },
    )
    assert response.status_code == 201, response.text
    assert jobs.run_pending() == 1
    return bucket, client.get(f"/api/jobs/{response.json()['id']}").json()


def test_big_import_reads_parquet_and_packs_rows_into_files(client, db, data_dir):
    bucket, job = start(client, ROWS)
    assert job["status"] == "completed", job
    progress = job["progress"]
    assert progress["processed"] == ROWS
    assert progress["skipped"] == 3 and progress["failed"] == 0
    assert progress["imported"] == ROWS - 3

    documents = list(
        db.scalars(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"])))
    )
    assert len(documents) == 6
    assert "rows 0 to" in sorted(d.source_note for d in documents)[0]
    text = "".join(
        (data_dir / "files" / document.text_uri).read_text(encoding="utf-8")
        for document in documents
    )
    assert "story 119999 had" in text and text.count("story 13 had") == 1
    assert not (data_dir / "imports" / job["id"]).exists()


def test_big_import_stops_at_its_limit(client, db):
    bucket, job = start(client, 70_000)
    assert job["status"] == "completed", job
    assert job["progress"]["processed"] == 70_000


def test_a_small_import_keeps_one_row_per_file(client, db):
    bucket, job = start(client, 5)
    assert job["status"] == "completed", job
    documents = list(
        db.scalars(select(Document).where(Document.bucket_id == uuid.UUID(bucket["id"])))
    )
    assert len(documents) == 5


def test_no_parquet_copy_falls_back_to_rows_and_waits_when_told_to(client, monkeypatch):
    def no_copy(*args):
        raise dataset_import.NoParquetCopy("someone/stories")

    monkeypatch.setattr(dataset_import, "parquet_urls", no_copy)
    monkeypatch.setattr(dataset_import.time, "sleep", lambda seconds: None)
    asked = []

    class Rows:
        def __init__(self, request, timeout):
            asked.append(request.full_url)
            if len(asked) == 1:
                raise dataset_import.urllib.error.HTTPError(
                    request.full_url, 429, "Too Many Requests", Message(), None
                )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"rows": [{"row_idx": 0, "row": {"text": "A short story."}}]}'

    monkeypatch.setattr(dataset_import.urllib.request, "urlopen", Rows)
    _, job = start(client, 1)
    assert job["status"] == "completed", job
    assert len(asked) == 2 and all("/rows?" in url for url in asked)


def test_cancelled_big_import_leaves_no_download_behind(client, db, data_dir, monkeypatch):
    real_report = dataset_import.report

    def stop(job_id, **kwargs):
        if kwargs.get("completed", 0) >= 10_000:
            db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
            db.commit()
        real_report(job_id, **kwargs)

    monkeypatch.setattr(dataset_import, "report", stop)
    _, job = start(client, ROWS)
    assert job["status"] == "cancelled", job
    assert not (data_dir / "imports" / job["id"]).exists()
