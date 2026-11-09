# ruff: noqa: F811
import uuid

import pytest
from test_jobs import queue_build, queue_tokenizer, snapshot  # noqa: F401

from forge import builds, jobs

pytest.importorskip("tokenizers")


@pytest.fixture(autouse=True)
def no_automatic_process(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)


def test_pieces_keep_every_character_and_cut_at_line_ends():
    text = "".join(f"line {index} of a long file\n" for index in range(500)) + "no end"
    parts = list(builds.pieces(text, size=100))
    assert "".join(parts) == text
    assert all(part.endswith("\n") for part in parts[:-1])
    assert max(len(part) for part in parts) <= 100
    assert "".join(builds.pieces("x" * 250, size=100)) == "x" * 250


def test_vocabulary_and_training_set_take_big_files_in_pieces(client, snapshot, monkeypatch):
    seen = []
    real = builds.pieces

    def spy(text, size=builds.PIECE_CHARS):
        for part in real(text, size=64):
            seen.append(len(part))
            yield part

    monkeypatch.setattr(builds, "pieces", spy)
    vocabulary = queue_tokenizer(client, snapshot)
    assert jobs.run_pending() == 1
    vocabulary = client.get(f"/api/tokenizers/{vocabulary['id']}").json()
    assert vocabulary["status"] == "ready", vocabulary
    assert seen and max(seen) <= 64

    build = queue_build(client, snapshot, vocabulary)
    assert jobs.run_pending() == 1
    build = client.get(f"/api/dataset-builds/{build['id']}").json()
    assert build["status"] == "ready", build
    assert build["stats"]["train"]["tokens"] > 0


def test_snapshot_is_checked_without_reading_it_whole(client, snapshot, data_dir):
    reader = builds.Snapshot(uuid.UUID(snapshot["id"]))
    rows = list(reader.rows("train"))
    assert len(rows) == reader.counts["train"] > 0

    path = data_dir / "datasets" / snapshot["id"] / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("a", "b"), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        builds.Snapshot(uuid.UUID(snapshot["id"]))
    response = client.post(
        "/api/tokenizers", json={"name": "late", "dataset_id": snapshot["id"], "vocab_size": 512}
    )
    assert response.status_code == 409
    assert "changed" in response.json()["error"]["message"]
