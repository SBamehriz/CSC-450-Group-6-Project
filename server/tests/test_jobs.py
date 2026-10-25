import hashlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from array import array
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from test_datasets import add_source, create_snapshot

from forge import builds, jobs
from forge.models import DatasetBuild, Job

BpeTokenizer = pytest.importorskip("tokenizers").Tokenizer

REAL_START_RUNNER = jobs.start_runner


@pytest.fixture(autouse=True)
def no_automatic_process(monkeypatch):
    # Most tests drain the real queue in-process. The concurrency test below
    # launches two real child processes against the same database.
    monkeypatch.setattr(jobs, "start_runner", lambda: None)


@pytest.fixture()
def snapshot(client):
    return create_snapshot(client, add_source(client), validation_fraction=0.25)


def queue_tokenizer(client, snapshot, name="pieces", vocab_size=512):
    response = client.post(
        "/api/tokenizers",
        json={"name": name, "dataset_id": snapshot["id"], "vocab_size": vocab_size},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def ready_tokenizer(client, snapshot):
    target = queue_tokenizer(client, snapshot)
    assert jobs.run_pending() == 1
    target = client.get(f"/api/tokenizers/{target['id']}").json()
    assert target["status"] == "ready", target
    return target


def queue_build(client, snapshot, tokenizer, name="tokens"):
    response = client.post(
        "/api/dataset-builds",
        json={"name": name, "tokenizer_id": tokenizer["id"], "dataset_id": snapshot["id"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def job_for(client, target):
    return next(
        job
        for job in client.get("/api/jobs").json()["items"]
        if job["payload"]["target_id"] == target["id"]
    )


def snapshot_rows(client, snapshot):
    raw = client.get(f"/api/datasets/{snapshot['id']}/download").content
    with ZipFile(io.BytesIO(raw)) as archive:
        return {
            split: [json.loads(line) for line in archive.read(f"{split}.jsonl").splitlines()]
            for split in ("train", "validation")
        }


def test_vocabulary_round_trip_learns_from_training_files_only(client, snapshot, ready_tokenizer):
    artifact = client.get(f"/api/tokenizers/{ready_tokenizer['id']}/artifact")
    assert artifact.status_code == 200
    tokenizer = BpeTokenizer.from_str(artifact.text)
    text = "A completely unseen word: 🦊 العربية café"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert tokenizer.token_to_id("<|eos|>") == 0
    assert ready_tokenizer["vocab_size"] == tokenizer.get_vocab_size()
    assert 257 <= ready_tokenizer["vocab_size"] <= 512
    rows = snapshot_rows(client, snapshot)
    trained_on = ready_tokenizer["trained_on"]
    assert trained_on["split"] == "train"
    assert trained_on["hashes"] == [row["content_hash"] for row in rows["train"]]
    assert trained_on["snapshot_sha256"] == snapshot["config"]["sha256"]
    job = job_for(client, ready_tokenizer)
    assert job["status"] == "completed" and job["attempts"] == 1
    assert job["progress"]["message"] == "Saving vocabulary"


def test_same_settings_give_the_same_vocabulary(client, snapshot, ready_tokenizer, data_dir):
    again = queue_tokenizer(client, snapshot, "same-input")
    jobs.run_pending()
    first = client.get(f"/api/tokenizers/{ready_tokenizer['id']}/artifact").content
    second = client.get(f"/api/tokenizers/{again['id']}/artifact").content
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_training_set_is_byte_identical_and_keeps_the_snapshot_split(
    client, snapshot, ready_tokenizer, data_dir
):
    one = queue_build(client, snapshot, ready_tokenizer)
    two = queue_build(client, snapshot, ready_tokenizer, "repeat")
    assert jobs.run_pending() == 2
    folders = [data_dir / "builds" / build["id"] for build in (one, two)]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folders[0].iterdir()} == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folders[1].iterdir()
    }
    manifest = client.get(f"/api/dataset-builds/{one['id']}/manifest").json()
    assert manifest["format"] == "forge-uint16-le-v1" and manifest["eos_id"] == 0
    rows = snapshot_rows(client, snapshot)
    for split in ("train", "validation"):
        listed = [entry["id"] for entry in manifest["splits"][split]["documents_in_order"]]
        assert listed == [row["id"] for row in rows[split]]
    build = client.get(f"/api/dataset-builds/{one['id']}").json()
    assert build["status"] == "ready"
    assert build["stats"]["train"]["documents"] == 3
    assert build["stats"]["validation"]["documents"] == 1
    assert manifest["warnings"]
    tokenizer = BpeTokenizer.from_str(
        client.get(f"/api/tokenizers/{ready_tokenizer['id']}/artifact").text
    )
    for split in ("train", "validation"):
        entry = manifest["splits"][split]
        raw = (folders[0] / entry["file"]).read_bytes()
        assert len(raw) == entry["tokens"] * 2
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        tokens = array("H")
        tokens.frombytes(raw)
        if sys.byteorder != "little":
            tokens.byteswap()
        assert tokens[-1] == 0
        assert tokens.count(0) == entry["documents"]
        assert len(tokenizer.decode(list(tokens)).encode("utf-8")) == entry["bytes"]


@pytest.mark.parametrize("change", [{"vocab_size": 65536}, {"vocab_size": 256}, {"name": "   "}])
def test_rejects_invalid_vocabulary_requests(client, snapshot, change):
    response = client.post(
        "/api/tokenizers", json={"name": "bad", "dataset_id": snapshot["id"], **change}
    )
    assert response.status_code == 422
    assert "error" in response.json()


def test_rejects_missing_snapshot_unready_vocabulary_and_duplicate_names(client, snapshot):
    missing = client.post(
        "/api/tokenizers", json={"name": "missing", "dataset_id": str(uuid.uuid4())}
    )
    assert missing.status_code == 404
    target = queue_tokenizer(client, snapshot)
    response = client.post(
        "/api/dataset-builds",
        json={"name": "bad", "tokenizer_id": target["id"], "dataset_id": snapshot["id"]},
    )
    assert response.status_code == 409
    response = client.post("/api/tokenizers", json={"name": "pieces", "dataset_id": snapshot["id"]})
    assert response.status_code == 409
    assert client.get("/api/jobs").json()["total"] == 1


def test_changed_snapshot_fails_cleanly_and_retry_works(
    client, snapshot, ready_tokenizer, data_dir
):
    path = data_dir / "datasets" / f"{snapshot['id']}.zip"
    original = path.read_bytes()
    build = queue_build(client, snapshot, ready_tokenizer)
    path.write_bytes(original + b"changed")
    jobs.run_pending()
    result = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert result["status"] == "failed" and "checksum" in result["error"]
    assert not (data_dir / "builds" / build["id"]).exists()
    assert not (data_dir / "builds" / (build["id"] + ".tmp")).exists()
    assert client.get("/api/health").json()["database"] == "ok"
    path.write_bytes(original)
    job = job_for(client, build)
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 200
    jobs.run_pending()
    assert client.get(f"/api/dataset-builds/{build['id']}").json()["status"] == "ready"
    assert job_for(client, build)["attempts"] == 2


def test_queued_cancel_and_ready_build_cannot_be_rewritten(
    client, snapshot, ready_tokenizer, data_dir
):
    build = queue_build(client, snapshot, ready_tokenizer)
    job = job_for(client, build)
    response = client.post(f"/api/jobs/{job['id']}/cancel")
    assert response.json()["status"] == "cancelled"
    assert jobs.run_pending() == 0
    client.post(f"/api/jobs/{job['id']}/retry")
    jobs.run_pending()
    manifest_path = data_dir / "builds" / build["id"] / "manifest.json"
    before = manifest_path.read_bytes()
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "completed"
    assert manifest_path.read_bytes() == before


def test_crash_recovery_marks_the_job_failed_and_cleans_up(
    client, snapshot, ready_tokenizer, db, data_dir
):
    build = queue_build(client, snapshot, ready_tokenizer)
    job = job_for(client, build)
    row = db.get(Job, uuid.UUID(job["id"]))
    row.status = "running"
    target = db.get(DatasetBuild, uuid.UUID(build["id"]))
    target.status = "building"
    db.commit()
    temporary = data_dir / "builds" / (build["id"] + ".tmp")
    temporary.mkdir(parents=True)
    (temporary / "half.bin").write_bytes(b"partial")
    recovered = client.get(f"/api/jobs/{job['id']}").json()
    assert recovered["status"] == "failed"
    assert "stopped unexpectedly" in recovered["error"]
    assert not temporary.exists()


@pytest.mark.parametrize("limit", ["MAX_BUILD_TOKENS", "MAX_SPLIT_BYTES"])
def test_size_guardrails_leave_no_partial_build(
    client, snapshot, ready_tokenizer, monkeypatch, data_dir, limit
):
    monkeypatch.setattr(builds, limit, 1)
    build = queue_build(client, snapshot, ready_tokenizer)
    jobs.run_pending()
    result = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert result["status"] == "failed"
    assert not list((data_dir / "builds").glob(build["id"] + "*"))


def test_two_real_runner_processes_take_turns(client, snapshot):
    one = queue_tokenizer(client, snapshot, "one")
    two = queue_tokenizer(client, snapshot, "two")
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "forge.jobs"],
            env=os.environ.copy(),
            cwd=Path(__file__).parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            out, err = process.communicate(timeout=60)
            assert process.returncode == 0, (out, err)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    for target in (one, two):
        assert client.get(f"/api/tokenizers/{target['id']}").json()["status"] == "ready"
        assert job_for(client, target)["attempts"] == 1


def test_failed_vocabulary_does_not_publish_partial_files(client, snapshot, data_dir):
    path = data_dir / "datasets" / f"{snapshot['id']}.zip"
    target = queue_tokenizer(client, snapshot)
    path.write_bytes(b"not a zip")
    jobs.run_pending()
    result = client.get(f"/api/tokenizers/{target['id']}").json()
    assert result["status"] == "failed" and result["vocab_size"] is None
    assert not list((data_dir / "tokenizers").glob(target["id"] + "*"))


def test_runner_start_failure_is_visible_and_retryable(client, snapshot, monkeypatch):
    def fail():
        raise OSError("No process slots available.")

    monkeypatch.setattr(jobs, "start_runner", fail)
    response = client.post(
        "/api/tokenizers", json={"name": "cannot-start", "dataset_id": snapshot["id"]}
    )
    assert response.status_code == 503
    job = client.get("/api/jobs").json()["items"][0]
    assert job["status"] == "failed"
    assert "No process slots available." in job["error"]
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    client.post(f"/api/jobs/{job['id']}/retry")
    jobs.run_pending()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "completed"


def test_jobs_keep_running_after_the_page_closes(client, snapshot, monkeypatch, tmp_path):
    # Use the actual launch path, even when the caller has another working
    # directory. Keep handles only so a failed assertion cannot leave a child.
    original_popen = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(jobs, "start_runner", REAL_START_RUNNER)
    monkeypatch.chdir(tmp_path)
    try:
        with TestClient(client.app) as running_app:
            target = queue_tokenizer(running_app, snapshot)
        # Closing the API lifecycle does not stop the already launched work.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = client.get(f"/api/tokenizers/{target['id']}").json()
            if result["status"] not in ("pending", "building"):
                break
            time.sleep(0.05)
        assert result["status"] == "ready", result
        assert job_for(client, target)["attempts"] == 1
        for process in processes:
            assert process.wait(timeout=10) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_runner_launch_failure_does_not_prevent_app_start(client, monkeypatch):
    def fail():
        raise OSError("No process slots available.")

    monkeypatch.setattr(jobs, "start_runner", fail)
    with TestClient(client.app) as running_app:
        assert running_app.get("/api/health").json()["database"] == "ok"


def test_launch_failure_does_not_fail_a_job_another_runner_finished(client, snapshot, monkeypatch):
    def finish_then_fail():
        jobs.run_pending()
        raise OSError("No process slots available.")

    monkeypatch.setattr(jobs, "start_runner", finish_then_fail)
    target = queue_tokenizer(client, snapshot)
    assert client.get(f"/api/tokenizers/{target['id']}").json()["status"] == "ready"
    assert job_for(client, target)["status"] == "completed"


def test_build_without_git_still_records_its_inputs(client, snapshot, ready_tokenizer, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(builds.subprocess, "run", no_git)
    build = queue_build(client, snapshot, ready_tokenizer)
    jobs.run_pending()
    manifest = client.get(f"/api/dataset-builds/{build['id']}/manifest").json()
    assert manifest["forge_git_sha"] == "unknown"
    assert manifest["tokenizer_sha256"] and manifest["splits"]


def test_vocabulary_in_use_cannot_be_deleted(client, snapshot, ready_tokenizer, data_dir):
    build = queue_build(client, snapshot, ready_tokenizer)
    jobs.run_pending()
    response = client.delete(f"/api/tokenizers/{ready_tokenizer['id']}")
    assert response.status_code == 409
    assert "training set" in response.json()["error"]["message"]
    assert client.delete(f"/api/dataset-builds/{build['id']}").status_code == 204
    assert not (data_dir / "builds" / build["id"]).exists()
    assert client.delete(f"/api/tokenizers/{ready_tokenizer['id']}").status_code == 204
    assert not (data_dir / "tokenizers" / ready_tokenizer["id"]).exists()
    assert client.get("/api/jobs").json()["total"] == 0


def test_a_queued_item_cannot_be_deleted_until_its_job_stops(client, snapshot):
    target = queue_tokenizer(client, snapshot)
    assert client.delete(f"/api/tokenizers/{target['id']}").status_code == 409
    client.post(f"/api/jobs/{job_for(client, target)['id']}/cancel")
    assert client.delete(f"/api/tokenizers/{target['id']}").status_code == 204


def test_try_a_vocabulary_on_your_own_text(client, ready_tokenizer):
    response = client.post(
        f"/api/tokenizers/{ready_tokenizer['id']}/preview", json={"text": "Lesson 2. Text"}
    )
    assert response.status_code == 200
    preview = response.json()
    assert preview["token_count"] == len(preview["ids"]) == len(preview["pieces"]) > 0


def test_jobs_list_leaves_out_prepare_data_jobs(client, db, ready_tokenizer):
    db.add(Job(job_type="tokenization", status="completed", payload={"dataset_id": "x"}))
    db.commit()
    listed = client.get("/api/jobs").json()
    assert listed["total"] == 1
    assert listed["items"][0]["job_type"] == "tokenizer_train"


def test_removing_the_snapshot_keeps_a_ready_training_set(
    client, snapshot, ready_tokenizer, data_dir
):
    build = queue_build(client, snapshot, ready_tokenizer)
    jobs.run_pending()
    directory = data_dir / "builds" / build["id"]
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert client.delete(f"/api/datasets/{snapshot['id']}").status_code == 204
    result = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert result["status"] == "ready" and result["dataset_id"] is None
    assert result["dataset_name"] == snapshot["name"]
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
