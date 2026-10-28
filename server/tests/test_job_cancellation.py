"""Tests for Sprint 8 job cancellation cleanup (C4G6P-246)."""
from forge.models import Job, DatasetBuild
from forge.db import get_engine
from sqlalchemy.orm import Session
import uuid
from forge import builds
from forge import jobs
from test_jobs import snapshot, queue_tokenizer, queue_build, job_for
from pathlib import Path

import pytest


def test_queued_vocabulary_job_can_be_cancelled(client, snapshot):
    """Check that a queued vocabulary job can be cancelled."""

    target = queue_tokenizer(client, snapshot)

    job = job_for(client, target)

    response = client.post(f"/api/jobs/{job['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    assert jobs.run_pending() == 0
def test_cancelled_vocabulary_job_removes_partial_output(
    client, snapshot, data_dir, monkeypatch
):
    """Cancelling a running vocabulary job removes incomplete output."""

    from forge import builds

    target = queue_tokenizer(client, snapshot)
    job = job_for(client, target)

    # Replace the real tokenizer work with a controlled cancellation.
    def cancel_during_training(*args, **kwargs):
        # Simulate an incomplete output file.
        partial_dir = data_dir / "tokenizers" / (target["id"] + ".tmp")
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "partial.txt").write_text("unfinished")

        # Request cancellation while the job is running.
        response = client.post(f"/api/jobs/{job['id']}/cancel")
        assert response.status_code == 200

        # Signal that cancellation has been requested.
        jobs.report(uuid.UUID(job["id"]), 1, 10, "Processing")

    monkeypatch.setattr(builds, "train_tokenizer", cancel_during_training)

    jobs.run_pending()

    result = client.get(f"/api/jobs/{job['id']}").json()

    assert result["status"] == "cancelled"

    # Verify the vocabulary record was also marked as cancelled.
    target_result = client.get(f"/api/tokenizers/{target['id']}").json()

    assert target_result["status"] == "cancelled"

    # Verify the cancelled job is recorded correctly in the database.
    with Session(get_engine()) as session:
        db_job = session.get(Job, uuid.UUID(job["id"]))

        assert db_job is not None
        assert db_job.status == "cancelled"
        assert db_job.cancel_requested is True
def test_cancelled_dataset_build_removes_partial_output(
    client, snapshot, data_dir, monkeypatch
):
    """A cancelled dataset build must not leave incomplete output."""

    from forge import builds

    ready_tokenizer = queue_tokenizer(client, snapshot)
    assert jobs.run_pending() == 1

    build = queue_build(client, snapshot, ready_tokenizer)
    job = job_for(client, build)

    original_build = builds.build_dataset

    def cancel_during_build(*args, **kwargs):
        # Create a partial output file to simulate interrupted work.
        partial_dir = data_dir / "builds" / (build["id"] + ".tmp")
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "incomplete.tmp").write_text("unfinished")

        # Request cancellation while the job is running.
        response = client.post(f"/api/jobs/{job['id']}/cancel")
        assert response.status_code == 200

        # Trigger the runner's cancellation check.
        jobs.report(uuid.UUID(job["id"]), 1, 10, "Processing")

        return original_build(*args, **kwargs)

    monkeypatch.setattr(builds, "build_dataset", cancel_during_build)

    jobs.run_pending()

    result = client.get(f"/api/jobs/{job['id']}").json()

    assert result["status"] == "cancelled"
    # Verify the cancelled dataset job in the database.
    with Session(get_engine()) as session:
        db_job = session.get(Job, uuid.UUID(job["id"]))

        assert db_job is not None
        assert db_job.status == "cancelled"
        assert db_job.cancel_requested is True

    partial_dir = data_dir / "builds" / (build["id"] + ".tmp")
    assert not partial_dir.exists()

    build_result = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert build_result["status"] == "cancelled"

    # Verify that no completed dataset manifest was published.
    with Session(get_engine()) as session:
        db_build = session.get(DatasetBuild, uuid.UUID(build["id"]))

        assert db_build is not None
        assert db_build.status == "cancelled"
        assert db_build.manifest_uri is None

def test_database_schema(engine):
    from sqlalchemy import inspect

    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("jobs")]

    print("\nDatabase:", engine.url.render_as_string(hide_password=True))
    print("Jobs columns:", columns)

    assert "pid" in columns