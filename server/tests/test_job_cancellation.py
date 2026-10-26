"""Tests for Sprint 8 job cancellation cleanup (C4G6P-246)."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "job_type",
    ["vocabulary", "dataset", "training"],
)
def test_cancelled_job_leaves_no_partial_files(tmp_path, job_type):
    """Verify the expected cleanup of incomplete job output."""

    output_dir = tmp_path / job_type
    output_dir.mkdir()

    partial_file = output_dir / "incomplete.tmp"
    partial_file.write_text("unfinished job output")

    # TODO: Start the actual job, cancel it midway,
    # and wait for cancellation to finish.
    #
    # The cancellation implementation must remove
    # incomplete output before this assertion.
    pytest.skip("Waiting for job cancellation implementation")

    assert not partial_file.exists()
def test_queued_vocabulary_job_can_be_cancelled(client, snapshot):
    """Check that a queued vocabulary job can be cancelled."""

    target = queue_tokenizer(client, snapshot)

    job = job_for(client, target)

    response = client.post(f"/api/jobs/{job['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    assert jobs.run_pending() == 0