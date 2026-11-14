import threading
import uuid

import numpy as np
import pytest
from sqlalchemy import select
from test_datasets import add_source

from forge import builds, chat_api, chatformat, jobs
from forge.models import Bucket, Checkpoint, Job, Model, Run

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from forge.train import trainer  # noqa: E402
from forge.train.model import GPT  # noqa: E402
from forge.train.presets import ModelConfig  # noqa: E402

SHAPE = {"n_layer": 1, "d_model": 32, "n_head": 2, "ctx_len": 64, "vocab_size": 320}
SETTINGS = {
    "batch_size": 4,
    "max_steps": 20,
    "learning_rate": 3e-3,
    "eval_every": 10,
    "save_every": 10,
    "eval_tokens": 2000,
}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    unloaded = []
    monkeypatch.setattr(chat_api, "unload_models", lambda *a: unloaded.append(a))
    return unloaded


def create(client, bucket, name="first", **body):
    payload = {
        "name": name,
        "bucket_id": bucket["id"],
        "settings": SETTINGS,
        "device": "cpu",
        **body,
    }
    if "base_model_id" not in body:
        payload.setdefault("shape", SHAPE)
    return client.post("/api/train", json=payload)


def kinds(db) -> list[str]:
    return [job.job_type for job in db.scalars(select(Job).order_by(Job.created_at, Job.id))]


def chat_source(client, db, name="talks"):
    bucket = client.post("/api/buckets", json={"name": name}).json()

    def talk(index: int) -> bytes:
        turns = [
            {"role": "user", "content": f"What is lesson {index} about?"},
            {"role": "assistant", "content": f"Lesson {index} is about real snapshots."},
        ]
        return (chatformat.render(turns) * 15).encode()

    files = [("files", (f"talk-{index}.txt", talk(index), "text/plain")) for index in range(3)]
    assert client.post(f"/api/buckets/{bucket['id']}/documents", files=files).status_code == 201
    db.get(Bucket, uuid.UUID(bucket["id"])).kind = "chat"
    db.commit()
    return bucket


def test_one_call_lines_up_the_original_jobs_and_trains(client, db, quiet):
    bucket = add_source(client)
    response = create(client, bucket)
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["status"] == "pending" and run["stage"] == "waiting"
    assert not run["fine_tune"] and run["model_name"] == "first"
    assert kinds(db) == ["snapshot_create", "tokenizer_train", "dataset_build", "train_run"]
    assert quiet, "chat models give the card back"
    model = db.get(Model, uuid.UUID(run["model_id"]))
    assert model is not None and model.preset is None and model.parent_id is None

    assert jobs.run_pending() == 4
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "completed", done
    assert done["stage"] == "finished" and done["current_step"] == 20
    points = client.get(f"/api/runs/{run['id']}/metrics").json()["items"]
    assert points[-1]["loss"] < points[0]["loss"]
    db.expire_all()
    assert db.get(Model, model.id).tokenizer_id is not None


def test_a_ready_snapshot_vocabulary_and_training_set_are_reused(client, db):
    bucket = add_source(client)
    assert create(client, bucket).status_code == 201
    assert jobs.run_pending() == 4
    assert create(client, bucket, name="second").status_code == 201
    assert kinds(db)[4:] == ["train_run"]
    # a changed collection gets a new snapshot, vocabulary and training set
    files = [("files", ("more.txt", b"Another lesson with new words. " * 30, "text/plain"))]
    client.post(f"/api/buckets/{bucket['id']}/documents", files=files)
    assert create(client, bucket, name="third").status_code == 201
    assert kinds(db)[5:] == ["snapshot_create", "tokenizer_train", "dataset_build", "train_run"]


def test_a_run_needs_text_a_free_name_and_a_good_shape(client, db):
    empty = client.post("/api/buckets", json={"name": "empty"}).json()
    assert create(client, empty).json()["error"]["code"] == "no_text"
    bucket = add_source(client)
    assert create(client, bucket).status_code == 201
    taken = create(client, bucket)
    assert taken.status_code == 409 and taken.json()["error"]["code"] == "name_taken"
    odd = create(client, bucket, name="odd", shape={**SHAPE, "d_model": 30, "n_head": 4})
    assert odd.status_code == 422 and "split evenly" in odd.json()["error"]["message"]
    tiny = create(client, bucket, name="tiny", shape={**SHAPE, "vocab_size": 50})
    assert tiny.json()["error"]["code"] == "invalid_shape"
    body = {"name": "no shape", "bucket_id": bucket["id"]}
    assert client.post("/api/train", json=body).json()["error"]["code"] == "shape_missing"
    assert db.scalar(select(Model).where(Model.name == "odd")) is None


def test_when_a_step_fails_the_later_ones_fail_with_a_plain_reason(client, db, monkeypatch):
    def broken(*args):
        raise ValueError("The vocabulary could not be learned.")

    monkeypatch.setattr(builds, "train_tokenizer", broken)
    run = create(client, add_source(client)).json()
    assert jobs.run_pending() == 4
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "failed" and done["stage"] == "finished"
    assert done["error"].startswith("The text for this run could not be made ready.")
    assert "The vocabulary could not be learned." in done["error"]
    assert "Traceback" not in done["error"]
    build = db.scalar(select(Job).where(Job.job_type == "dataset_build"))
    assert build is not None and build.status == "failed"
    assert "ready vocabulary" in (build.error or "")


def test_the_stage_says_what_the_text_is_waiting_on(client, db):
    run = create(client, add_source(client)).json()
    snapshot = db.scalar(select(Job).where(Job.job_type == "snapshot_create"))
    assert snapshot is not None
    snapshot.status = "running"
    snapshot.progress = {"message": "Writing training text"}
    db.commit()
    out = client.get(f"/api/runs/{run['id']}").json()
    assert out["stage"] == "preparing"
    assert out["stage_message"] == "Getting the text ready: Writing training text."


def _trained(client, db):
    run = create(client, add_source(client), name="base").json()
    assert jobs.run_pending() == 4
    return db.get(Model, uuid.UUID(run["model_id"]))


def test_a_chat_fine_tune_is_a_new_model_from_the_base_best_save_point(client, db, monkeypatch):
    base = _trained(client, db)
    masked = []
    real = trainer.learned_only
    monkeypatch.setattr(trainer, "learned_only", lambda x, y: masked.append(1) or real(x, y))
    response = create(client, chat_source(client, db), name="tuned", base_model_id=str(base.id))
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["fine_tune"] and run["base_model_id"] == str(base.id)
    assert run["base_model_name"] == "base"
    # the base's vocabulary: no new one is learned
    assert kinds(db)[4:] == ["snapshot_create", "dataset_build", "train_run"]
    model = db.get(Model, uuid.UUID(run["model_id"]))
    assert model is not None and model.parent_id == base.id and model.chat
    assert (model.n_layer, model.d_model, model.vocab_size) == (1, 32, 320)
    assert model.tokenizer_id == base.tokenizer_id
    row = db.get(Run, uuid.UUID(run["id"]))
    best = db.scalar(
        select(Checkpoint).join(Run).where(Run.model_id == base.id, Checkpoint.is_best)
    )
    assert row is not None and best is not None and row.init_checkpoint_id == best.id
    assert row.hyperparams["chat"] is True

    assert jobs.run_pending() == 3
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "completed", done
    assert masked, "a chat fine-tune masks its loss"


def test_a_fine_tune_starts_from_the_base_weights(client, db, monkeypatch):
    base = _trained(client, db)
    # nothing to learn in any step: no update, so the weights stay the base's
    monkeypatch.setattr(trainer, "learned_only", lambda x, y: np.full_like(y, -1))
    run = create(client, chat_source(client, db), name="still", base_model_id=str(base.id)).json()
    assert jobs.run_pending() == 3
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "completed", done
    assert done["loss"] is None
    start = db.get(Checkpoint, db.get(Run, uuid.UUID(run["id"])).init_checkpoint_id)
    after = db.scalar(select(Checkpoint).where(Checkpoint.run_id == uuid.UUID(run["id"])))
    first, second = trainer.load_checkpoint(start)["model"], trainer.load_checkpoint(after)["model"]
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_hub_and_untrained_models_are_refused_as_bases(client, db, make_model):
    bucket = add_source(client)
    hub = make_model()
    hub.source = "hub"
    db.commit()
    response = create(client, bucket, name="a", base_model_id=str(hub.id))
    assert response.json()["error"]["code"] == "hub_base"
    fresh = make_model()
    response = create(client, bucket, name="b", base_model_id=str(fresh.id))
    assert response.json()["error"]["code"] == "base_untrained"


def test_only_the_assistant_words_are_learned():
    ids = np.array([[2, 10, 11, 4, 3, 12, 13, 4, 2, 14]])
    x, y = ids[:, :-1], ids[:, 1:]
    kept = trainer.learned_only(x, y)
    # after the assistant mark (3): its words and the end mark after them
    assert kept.tolist() == [[-1, -1, -1, -1, 12, 13, 4, -1, -1]]


def test_two_cards_wrap_the_model_and_save_plain_weights(client, db, monkeypatch):
    # one process, a pass split across cards; here torch reports no card, so it runs as one
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: f"Card {index}")
    monkeypatch.setattr(trainer, "card_count", lambda device: 2)
    wrapped = []
    real = torch.nn.DataParallel
    monkeypatch.setattr(torch.nn, "DataParallel", lambda net: wrapped.append(net) or real(net))
    run = create(client, add_source(client), name="pair").json()
    assert jobs.run_pending() == 4
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "completed", done
    assert wrapped and done["metrics"]["cards"] == 2
    assert done["metrics"]["card_names"] == ["Card 0", "Card 1"]
    save = db.scalar(select(Checkpoint).where(Checkpoint.run_id == uuid.UUID(run["id"])))
    weights = trainer.load_checkpoint(save)["model"]
    assert not [name for name in weights if name.startswith("module.")]
    GPT(ModelConfig(**SHAPE)).load_state_dict(weights)


def test_the_model_enters_autocast_itself_in_any_thread():
    model = GPT(ModelConfig(**SHAPE))
    tokens = torch.randint(0, SHAPE["vocab_size"], (2, 16))
    assert model(tokens, tokens)[0].dtype == torch.float32
    model.autocast_dtype = torch.bfloat16
    seen = {}

    def replica():
        seen["dtype"] = model(tokens, tokens)[0].dtype

    worker = threading.Thread(target=replica)
    worker.start()
    worker.join()
    assert seen["dtype"] == torch.bfloat16


def test_a_run_whose_training_job_is_running_cannot_be_deleted(client, db, make_run):
    run = make_run(status="pending")
    job = jobs.enqueue_task(db, "train_run", {"run_id": str(run.id)})
    job.status = "running"
    db.commit()
    assert client.delete(f"/api/runs/{run.id}").json()["error"]["code"] == "job_running"


def test_emulated_bf16_does_not_count(monkeypatch):
    asked = []

    def supported(including_emulation=True):
        asked.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    assert trainer.real_bf16() is False and asked == [False]

    def older():
        return True

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", older)
    assert trainer.real_bf16() is True


def test_a_failed_snapshot_ends_the_chain_with_its_reason(client, monkeypatch):
    from forge import snapshots
    from forge.errors import ApiError

    def broken(*args, **kwargs):
        raise ApiError(409, "missing_text", "The cleaned text for 'note-1.txt' is missing.")

    monkeypatch.setattr(snapshots, "write_snapshot_streaming", broken)
    run = create(client, add_source(client)).json()
    # the runner carries on past the failed snapshot, so the rest fail at once
    assert jobs.run_pending() == 4
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "failed"
    assert done["error"] == (
        "The text for this run could not be made ready. "
        "The cleaned text for 'note-1.txt' is missing."
    )


def test_a_speed_test_with_no_shape_times_a_standard_small_model(client, db):
    from forge.train.speed import STANDARD

    response = client.post("/api/speed-tests", json={"device": "cpu"})
    assert response.status_code == 202, response.text
    job = db.get(Job, uuid.UUID(response.json()["id"]))
    assert job is not None and job.payload["shape"] == STANDARD
    assert "preset" not in job.payload
