import errno
import io
import json
import uuid
from zipfile import ZipFile

import pytest
from sqlalchemy import select, update
from test_datasets import add_source, create_snapshot
from test_jobs import queue_build, queue_tokenizer

from forge import builds, jobs
from forge.models import Checkpoint, DatasetBuild, Job, MetricPoint, Run, Tokenizer

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from forge.train import trainer  # noqa: E402

TINY = {"n_layer": 1, "d_model": 32, "n_head": 2, "ctx_len": 256}


@pytest.fixture(autouse=True)
def no_automatic_process(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)


@pytest.fixture()
def training_set(client):
    snapshot = create_snapshot(client, add_source(client), validation_fraction=0.25)
    vocabulary = queue_tokenizer(client, snapshot)
    assert jobs.run_pending() == 1
    build = queue_build(client, snapshot, vocabulary)
    assert jobs.run_pending() == 1
    build = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert build["status"] == "ready", build
    return build


def make_model(client, training_set, name="tiny"):
    response = client.post(
        "/api/models",
        json={"name": name, "tokenizer_id": training_set["tokenizer_id"], **TINY},
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_run(client, model, training_set, name="first", **settings):
    chosen = {
        "batch_size": 4,
        "max_steps": 20,
        "learning_rate": 3e-3,
        "eval_every": 10,
        "save_every": 10,
        "eval_tokens": 2000,
        **settings,
    }
    response = client.post(
        "/api/runs",
        json={
            "name": name,
            "model_id": model["id"],
            "build_id": training_set["id"],
            "device": "cpu",
            "seed": 7,
            "settings": chosen,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def train(client, run, action="start"):
    response = client.post(f"/api/runs/{run['id']}/{action}")
    assert response.status_code == 200, response.text
    assert jobs.run_pending() == 1
    return client.get(f"/api/runs/{run['id']}").json()


def final_weights(db, run_id):
    db.expire_all()
    row = db.scalar(
        select(Checkpoint)
        .where(Checkpoint.run_id == uuid.UUID(run_id))
        .order_by(Checkpoint.step.desc())
    )
    return trainer.load_checkpoint(row)["model"]


def test_a_run_trains_on_cpu_and_keeps_best_and_latest(client, db, training_set, data_dir):
    model = make_model(client, training_set)
    run = make_run(client, model, training_set, max_steps=30)
    assert run["status"] == "pending"
    assert run["build_name"] == training_set["name"]

    done = train(client, run)
    assert done["status"] == "completed", done
    assert done["current_step"] == 30
    assert done["tokens_seen"] > 0
    assert done["job_status"] == "completed"

    points = client.get(f"/api/runs/{run['id']}/metrics").json()["items"]
    assert [p["step"] for p in points] == list(range(1, 31))
    assert points[-1]["loss"] < points[0]["loss"]
    assert sum(p["val_loss"] is not None for p in points) == 3

    saves = client.get(f"/api/runs/{run['id']}/checkpoints").json()
    assert 1 <= len(saves) <= 2
    assert saves[0]["step"] == 30
    assert sum(save["is_best"] for save in saves) == 1
    assert all(save["sha256"] for save in saves)
    folder = data_dir / "runs" / run["id"]
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"step-{save['step']:08d}.pt" for save in saves
    )

    job = db.scalar(select(Job).where(Job.job_type == "train_run"))
    assert job is not None
    assert job.result["steps"] == 30


def test_resume_matches_an_unbroken_run(client, db, training_set, monkeypatch):
    model = make_model(client, training_set)
    unbroken = make_run(client, model, training_set, name="unbroken", save_every=100)
    assert train(client, unbroken)["status"] == "completed"

    broken = make_run(client, model, training_set, name="broken", save_every=100)
    real_report = jobs.report

    def stop_at_ten(job_id, completed, total, message):
        if completed == 10:
            db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
            db.commit()
        real_report(job_id, completed, total, message)

    monkeypatch.setattr(trainer, "report", stop_at_ten)
    stopped = train(client, broken)
    assert stopped["status"] == "stopped", stopped
    assert stopped["current_step"] == 10
    monkeypatch.setattr(trainer, "report", real_report)

    resumed = train(client, broken, "resume")
    assert resumed["status"] == "completed", resumed
    assert resumed["tokens_seen"] == client.get(f"/api/runs/{unbroken['id']}").json()["tokens_seen"]

    first, second = final_weights(db, unbroken["id"]), final_weights(db, broken["id"])
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name
    steps = [p["step"] for p in client.get(f"/api/runs/{broken['id']}/metrics").json()["items"]]
    assert steps == list(range(1, 21))


def _fails_with(client, run, words):
    done = train(client, run)
    assert done["status"] == "failed", done
    for word in words:
        assert word in done["error"], done["error"]
    return done


def test_nan_loss_stops_with_its_own_reason(client, training_set, monkeypatch):
    real_forward = trainer.GPT.forward

    def poisoned(self, idx, targets=None):
        logits, loss = real_forward(self, idx, targets)
        return logits, None if loss is None else loss * float("nan")

    monkeypatch.setattr(trainer.GPT, "forward", poisoned)
    run = make_run(client, make_model(client, training_set), training_set)
    _fails_with(client, run, ["NaN", "step 1", "lower learning rate"])


def test_out_of_memory_says_what_to_change(client, training_set, monkeypatch):
    def too_big(self, idx, targets=None):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(trainer.GPT, "forward", too_big)
    run = make_run(client, make_model(client, training_set), training_set)
    _fails_with(client, run, ["ran out of memory", "Sequences per pass to 2"])


def test_full_disk_keeps_no_half_written_save_point(client, training_set, monkeypatch, data_dir):
    real_save = torch.save

    def half_then_full(state, path, *args, **kwargs):
        with open(path, "wb") as handle:
            handle.write(b"half a file")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(trainer.torch, "save", half_then_full)
    run = make_run(client, make_model(client, training_set), training_set)
    _fails_with(client, run, ["disk filled up", "step 10"])
    monkeypatch.setattr(trainer.torch, "save", real_save)
    folder = data_dir / "runs" / run["id"]
    assert not folder.exists() or list(folder.iterdir()) == []
    assert client.get(f"/api/runs/{run['id']}/checkpoints").json() == []


def test_app_stopping_mid_run_says_where_to_resume(client, db, training_set):
    run = make_run(client, make_model(client, training_set), training_set)
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    db.execute(update(Job).where(Job.job_type == "train_run").values(status="running", pid=99999))
    db.execute(update(Run).where(Run.id == uuid.UUID(run["id"])).values(current_step=12))
    db.commit()
    jobs.recover_jobs()
    after = client.get(f"/api/runs/{run['id']}").json()
    assert after["status"] == "failed"
    assert "app stopped" in after["error"]
    assert "step 12" in after["error"]
    assert after["job_status"] == "failed"


def test_runs_refuse_the_wrong_vocabulary_and_wrong_order(client, training_set):
    other = client.post("/api/models", json={"name": "wide", "vocab_size": 16384, **TINY}).json()
    response = client.post(
        "/api/runs",
        json={"name": "mismatch", "model_id": other["id"], "build_id": training_set["id"]},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "vocabulary_mismatch"

    run = make_run(client, make_model(client, training_set), training_set, max_steps=5)
    assert client.post(f"/api/runs/{run['id']}/resume").json()["error"]["code"] == "no_save_point"
    assert client.post(f"/api/runs/{run['id']}/stop").json()["error"]["code"] == "not_running"
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    assert client.post(f"/api/runs/{run['id']}/start").json()["error"]["code"] == "already_running"
    assert jobs.run_pending() == 1
    assert client.post(f"/api/runs/{run['id']}/start").json()["error"]["code"] == "already_started"


def test_train_tokens_set_the_number_of_steps(client, training_set):
    model = make_model(client, training_set)
    response = client.post(
        "/api/runs",
        json={
            "name": "by tokens",
            "model_id": model["id"],
            "build_id": training_set["id"],
            "train_tokens": 4 * 256 * 25,
            "settings": {"batch_size": 4},
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["max_steps"] == 25


def test_stopping_a_queued_run_never_starts_it(client, training_set):
    run = make_run(client, make_model(client, training_set), training_set)
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    stopped = client.post(f"/api/runs/{run['id']}/stop").json()
    assert stopped["status"] == "stopped"
    assert jobs.run_pending() == 0
    assert client.get(f"/api/runs/{run['id']}").json()["current_step"] == 0
    # with no save point to resume from, it can start again
    assert client.post(f"/api/runs/{run['id']}/resume").json()["error"]["code"] == "no_save_point"
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    assert jobs.run_pending() == 1
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "completed"


def test_download_has_weights_shape_and_vocabulary(client, db, training_set, data_dir):
    run = make_run(client, make_model(client, training_set), training_set, max_steps=10)
    train(client, run)
    save = client.get(f"/api/runs/{run['id']}/checkpoints").json()[0]
    response = client.get(f"/api/checkpoints/{save['id']}/download")
    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"save_point.pt", "config.json", "tokenizer.json"}
        shape = json.loads(archive.read("config.json"))
        assert shape["n_layer"] == 1 and shape["step"] == 10
        assert shape["checkpoint_sha256"] == save["sha256"]

    (data_dir / save["uri"]).write_bytes(b"changed")
    response = client.get(f"/api/checkpoints/{save['id']}/download")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "changed_on_disk"


@pytest.mark.parametrize("kind", ["tokenizer", "build"])
def test_cancel_mid_way_leaves_no_half_files_or_rows(client, db, data_dir, monkeypatch, kind):
    snapshot = create_snapshot(client, add_source(client), validation_fraction=0.25)
    vocabulary = None
    if kind == "build":
        vocabulary = queue_tokenizer(client, snapshot)
        assert jobs.run_pending() == 1
    real_report = builds.report

    def cancel_now(job_id, completed, total, message):
        db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
        db.commit()
        real_report(job_id, completed, total, message)

    monkeypatch.setattr(builds, "report", cancel_now)
    if kind == "tokenizer":
        target = queue_tokenizer(client, snapshot, name="halfway")
        table = Tokenizer
    else:
        target = queue_build(client, snapshot, vocabulary)
        table = DatasetBuild
    assert jobs.run_pending() == 1

    db.expire_all()
    row = db.get(table, uuid.UUID(target["id"]))
    assert row is not None and row.status == "cancelled"
    assert row.artifact_uri is None if kind == "tokenizer" else row.manifest_uri is None
    leftovers = [p for p in data_dir.rglob("*") if target["id"] in p.name]
    assert leftovers == []


def test_cancelled_run_keeps_only_whole_save_points(
    client, db, training_set, data_dir, monkeypatch
):
    run = make_run(client, make_model(client, training_set), training_set, save_every=5)
    real_report = jobs.report

    def stop_at_fifteen(job_id, completed, total, message):
        if completed == 15:
            db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
            db.commit()
        real_report(job_id, completed, total, message)

    monkeypatch.setattr(trainer, "report", stop_at_fifteen)
    assert train(client, run)["status"] == "stopped"
    folder = data_dir / "runs" / run["id"]
    assert not [p for p in folder.iterdir() if p.suffix == ".tmp"]
    rows = list(db.scalars(select(Checkpoint).where(Checkpoint.run_id == uuid.UUID(run["id"]))))
    assert {p.name for p in folder.iterdir()} == {row.uri.rsplit("/", 1)[-1] for row in rows}
    assert db.scalar(select(MetricPoint.step).order_by(MetricPoint.step.desc())) == 15
