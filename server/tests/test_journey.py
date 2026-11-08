import io
import time
from zipfile import ZipFile

import pytest
from sqlalchemy import update
from test_jobs import queue_build, queue_tokenizer

from forge import jobs
from forge.models import Job

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from forge.train import trainer  # noqa: E402

LESSONS = [
    f"Lesson {index}. The river runs past the old mill and the children count the boats. " * 12
    for index in range(6)
]


@pytest.fixture(autouse=True)
def no_automatic_process(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)


def test_raw_text_to_a_downloaded_model_on_cpu(client, db, monkeypatch):
    began = time.perf_counter()
    # upload into a collection
    bucket = client.post("/api/buckets", json={"name": "river notes"}).json()
    uploaded = client.post(
        f"/api/buckets/{bucket['id']}/documents",
        files=[
            ("files", (f"lesson-{index}.txt", text.encode(), "text/plain"))
            for index, text in enumerate(LESSONS)
        ],
    )
    assert uploaded.status_code == 201 and uploaded.json()["parsed"] == len(LESSONS)

    # snapshot, vocabulary, training set
    snapshot = client.post(
        "/api/datasets",
        json={"name": "river", "bucket_id": bucket["id"], "validation_fraction": 0.34},
    ).json()
    assert snapshot["status"] == "ready"
    vocabulary = queue_tokenizer(client, snapshot, name="river pieces")
    assert jobs.run_pending() == 1
    build = queue_build(client, snapshot, vocabulary, name="river tokens")
    assert jobs.run_pending() == 1
    build = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert build["status"] == "ready"

    # a model that reads that vocabulary, and a run on that training set
    model = client.post(
        "/api/models",
        json={
            "name": "river model",
            "tokenizer_id": vocabulary["id"],
            "n_layer": 2,
            "d_model": 64,
            "n_head": 4,
            "ctx_len": 256,
        },
    ).json()
    run = client.post(
        "/api/runs",
        json={
            "name": "river run",
            "model_id": model["id"],
            "build_id": build["id"],
            "device": "cpu",
            "settings": {"batch_size": 4, "max_steps": 20, "eval_every": 10, "save_every": 10},
        },
    ).json()

    # train, stop part way, resume to the end
    real_report = jobs.report

    def stop_at_ten(job_id, completed, total, message):
        if completed == 10:
            db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
            db.commit()
        real_report(job_id, completed, total, message)

    monkeypatch.setattr(trainer, "report", stop_at_ten)
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    assert jobs.run_pending() == 1
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "stopped"
    monkeypatch.setattr(trainer, "report", real_report)
    assert client.post(f"/api/runs/{run['id']}/resume").status_code == 200
    assert jobs.run_pending() == 1
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "completed" and done["current_step"] == 20

    # download the best save point with its vocabulary
    save = next(s for s in client.get(f"/api/runs/{run['id']}/checkpoints").json() if s["is_best"])
    archive = client.get(f"/api/checkpoints/{save['id']}/download")
    assert archive.status_code == 200
    with ZipFile(io.BytesIO(archive.content)) as opened:
        assert {"save_point.pt", "config.json", "tokenizer.json"} <= set(opened.namelist())

    assert time.perf_counter() - began < 300
