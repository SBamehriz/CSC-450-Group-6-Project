# ruff: noqa: F811
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from test_training_api import stand_in
from test_training_loop import make_model, make_run, train, training_set  # noqa: F401

from forge import harness, hub, jobs
from forge.models import Checkpoint, Model

torch = pytest.importorskip("torch")
try:
    import transformers
except ImportError:
    transformers = None
needs_hf = pytest.mark.skipif(transformers is None, reason="Install the hub extra to test this")

from forge.train import speed  # noqa: E402

SHAPE = {"n_layer": 1, "d_model": 32, "n_head": 2, "ctx_len": 256, "vocab_size": 512}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    monkeypatch.setattr(speed, "MOST_STEPS", 4)
    # no card unless a test says so
    monkeypatch.setattr("forge.gpu.TOOL", ("forge-no-such-tool",))


def finished(client, job):
    assert jobs.run_pending() == 1
    done = client.get(f"/api/jobs/{job['id']}").json()
    return done


def speed_test(client, **body):
    response = client.post("/api/speed-tests", json={"device": "cpu", **body})
    assert response.status_code == 202, response.text
    done = finished(client, response.json())
    assert done["status"] == "completed", done
    return done


def test_speed_test_measures_this_machine(client):
    done = speed_test(client, shape=SHAPE, sequences_per_step=2)
    assert done["result"]["tokens_per_sec"] > 0
    rows = client.get("/api/speed-tests").json()
    assert len(rows) == 1
    assert rows[0]["device"] == "cpu" and rows[0]["device_name"] == "The CPU"
    assert rows[0]["steps"] >= 3 and rows[0]["shape"] == SHAPE


def test_estimate_says_where_its_speed_came_from(client):
    ask = {"shape": SHAPE, "train_tokens": 1_000_000, "sequences_per_step": 2}
    none = client.post("/api/estimate", json=ask).json()
    assert none["speed_source"] == "none"
    assert none["hours"] is None and none["fits"] is None and none["free_bytes"] is None

    speed_test(client, shape=SHAPE, sequences_per_step=2)
    measured = client.post("/api/estimate", json=ask).json()
    assert measured["speed_source"] == "measured"
    assert measured["hours"] == pytest.approx(1_000_000 / measured["tokens_per_sec"] / 3600)

    wider = client.post("/api/estimate", json={**ask, "shape": {**SHAPE, "d_model": 64}}).json()
    assert wider["speed_source"] == "scaled"
    assert wider["tokens_per_sec"] < measured["tokens_per_sec"]


def test_estimate_judges_memory_against_what_is_free(client, monkeypatch, tmp_path):
    ask = {"preset": "135m", "train_tokens": 10**9, "sequences_per_step": 16}
    stand_in(monkeypatch, tmp_path, "0, GPU-1111aaaa, Card, 8192, 4000, 595.79, 8.6\n")
    tight = client.post("/api/estimate", json=ask).json()
    assert tight["free_bytes"] == 4000 * 2**20
    assert tight["fits"] is False
    assert "sequences per pass" in tight["suggestion"]
    stand_in(monkeypatch, tmp_path, "0, GPU-1111aaaa, Card, 81920, 80000, 595.79, 8.6\n")
    roomy = client.post("/api/estimate", json=ask).json()
    assert roomy["fits"] is True and roomy["suggestion"] is None


def trained(client, training_set, name="ours", steps=10):  # noqa: F811
    model = make_model(client, training_set, name=name)
    run = make_run(client, model, training_set, name=f"{name} run", max_steps=steps)
    assert train(client, run)["status"] == "completed"
    save = client.get(f"/api/runs/{run['id']}/checkpoints").json()[0]
    return model, run, save


def exported(client, save, data_dir) -> Path:
    response = client.post(f"/api/checkpoints/{save['id']}/export")
    assert response.status_code == 202, response.text
    done = finished(client, response.json())
    assert done["status"] == "completed", done
    folder = data_dir / done["result"]["folder"]
    assert {"config.json", "model.safetensors", "tokenizer.json"} <= set(done["result"]["files"])
    return folder


@needs_hf
def test_export_gives_the_same_logits_in_transformers(client, db, training_set, data_dir):  # noqa: F811
    from forge.train.model import GPT
    from forge.train.presets import ModelConfig
    from forge.train.trainer import load_checkpoint

    _, _, save = trained(client, training_set)
    folder = exported(client, save, data_dir)
    assert transformers is not None
    theirs = transformers.GPT2LMHeadModel.from_pretrained(folder).eval()
    state = load_checkpoint(db.get(Checkpoint, uuid.UUID(save["id"])))
    ours = GPT(ModelConfig(**state["config"])).eval()
    ours.load_state_dict(state["model"])
    tokens = torch.randint(0, state["config"]["vocab_size"], (2, 40))
    with torch.no_grad():
        mine, _ = ours(tokens, tokens)
        other = theirs(tokens).logits
    assert mine.shape == other.shape
    assert torch.allclose(mine, other, atol=1e-5)

    archive = client.get(f"/api/checkpoints/{save['id']}/export")
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"


def fake_hub(monkeypatch, folder: Path, files: dict | None = None, gated=False):
    """Serve a folder as if it were a repo on the hub."""
    listed = files or {
        path.name: {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix == ".safetensors"
            else None,
        }
        for path in folder.iterdir()
    }

    def listing(repo_id, revision):
        if gated:
            raise hub.Refused(f"{repo_id} is gated. Accept its terms on huggingface.co first.")
        return listed, {"commit": "c0ffee" * 6 + "abcd", "license": "mit", "parameters": None}

    def read_small(repo_id, commit, name):
        return (folder / name).read_bytes()

    def fetch(repo_id, commit, name, destination):
        target = destination / name
        shutil.copyfile(folder / name, target)
        return target

    monkeypatch.setattr(hub, "listing", listing)
    monkeypatch.setattr(hub, "read_small", read_small)
    monkeypatch.setattr(hub, "fetch", fetch)


def imported(client, name="theirs", **body):
    response = client.post(
        "/api/models/import", json={"repo_id": "someone/tiny-gpt", "name": name, **body}
    )
    assert response.status_code == 202, response.text
    done = finished(client, response.json())
    assert done["status"] == "completed", done
    return client.get(f"/api/models/{done['result']['model_id']}").json()


@needs_hf
def test_hub_import_pins_the_commit_and_licence(client, training_set, data_dir, monkeypatch):  # noqa: F811
    ours, _, save = trained(client, training_set)
    fake_hub(monkeypatch, exported(client, save, data_dir))
    model = imported(client)
    assert model["source"] == "hub"
    assert model["config"]["hub"]["commit"].startswith("c0ffee")
    assert model["config"]["hub"]["license"] == "mit"
    assert model["n_layer"] == 1 and model["vocab_size"] == ours["vocab_size"]
    assert (data_dir / "models" / model["id"] / "model.safetensors").is_file()


@pytest.mark.parametrize(
    "files, said",
    [
        ({"config.json": {}, "tokenizer.json": {}, "pytorch_model.bin": {}}, "pickle weights"),
        ({"config.json": {}, "model.safetensors": {"sha256": "a"}}, "no tokenizer.json"),
        ({"tokenizer.json": {}, "model.safetensors": {"sha256": "a"}}, "no config.json"),
        ({"config.json": {}, "tokenizer.json": {}, "model.safetensors": {}}, "no checksum"),
    ],
)
def test_hub_import_refuses_before_downloading(client, monkeypatch, tmp_path, files, said):
    for entry in files.values():
        entry.setdefault("size", 10)
        entry.setdefault("sha256", None)
    (tmp_path / "config.json").write_text(json.dumps({"n_layer": 1}))
    fake_hub(monkeypatch, tmp_path, files)
    response = client.post("/api/models/import", json={"repo_id": "someone/tiny-gpt"})
    assert response.status_code == 409
    assert said in response.json()["error"]["message"]
    assert client.get("/api/jobs?job_type=model_import").json()["total"] == 0


def test_hub_import_refuses_a_gated_model_and_a_bad_name(client, monkeypatch, tmp_path):
    fake_hub(monkeypatch, tmp_path, {}, gated=True)
    response = client.post("/api/models/import", json={"repo_id": "someone/tiny-gpt"})
    assert "gated" in response.json()["error"]["message"]
    response = client.post("/api/models/import", json={"repo_id": "../../etc"})
    assert "owner/name" in response.json()["error"]["message"]


@needs_hf
def test_hub_import_checks_the_card_and_the_checksum(
    client,
    db,
    training_set,
    data_dir,
    monkeypatch,
    tmp_path,  # noqa: F811
):
    _, _, save = trained(client, training_set)
    folder = exported(client, save, data_dir)
    fake_hub(monkeypatch, folder)
    stand_in(monkeypatch, tmp_path, "0, GPU-1111aaaa, Card, 1024, 0, 595.79, 8.6\n")
    response = client.post("/api/models/import", json={"repo_id": "someone/tiny-gpt"})
    assert response.status_code == 409
    assert "Run it on the CPU" in response.json()["error"]["message"]

    response = client.post(
        "/api/models/import", json={"repo_id": "someone/tiny-gpt", "on_cpu": True}
    )
    assert response.status_code == 202
    (folder / "model.safetensors").write_bytes(b"not the same bytes")
    models = data_dir / "models"
    before = set(models.iterdir()) if models.exists() else set()
    done = finished(client, response.json())
    assert done["status"] == "failed"
    assert "checksum" in done["error"]
    assert db.scalar(select(Model).where(Model.source == "hub")) is None
    assert (set(models.iterdir()) if models.exists() else set()) == before


def test_benchmarks_are_refused_below_their_size(client, training_set):  # noqa: F811
    model = make_model(client, training_set)
    plan = client.get(f"/api/benchmarks/plan?model_id={model['id']}").json()
    assert plan["allowed"] == []
    assert len(plan["refused"]) == len(harness.BENCHMARKS)
    response = client.post("/api/benchmarks", json={"model_id": model["id"]})
    assert response.status_code == 409
    assert "chance-level" in response.json()["error"]["message"]


def test_a_mid_size_model_gets_ten_benchmarks_with_intervals(client, db, monkeypatch):
    model = Model(
        name="mid",
        n_layer=12,
        d_model=768,
        n_head=12,
        ctx_len=1024,
        vocab_size=50257,
        param_count=150_000_000,
        config={},
        source="hub",
        artifact_uri="models/none",
    )
    db.add(model)
    db.commit()
    plan = client.get(f"/api/benchmarks/plan?model_id={model.id}").json()
    assert len(plan["allowed"]) == 10
    assert {row["name"] for row in plan["refused"]} == {"ARC-Challenge", "MMLU"}
    assert all("about 1 billion" in row["reason"] for row in plan["refused"])

    class Loaded:
        name, checkpoint_id, device = "mid", None, "cpu"

    monkeypatch.setattr("forge.measuring.load", lambda *args: Loaded())
    seen = {}

    def fake_evaluate(loaded, tasks, limit):
        seen["tasks"], seen["limit"] = tasks, limit
        return {
            "results": {
                "hellaswag": {"acc,none": 0.3, "acc_norm,none": 0.31, "acc_norm_stderr,none": 0.01},
                "piqa": {"acc,none": 0.6, "acc_stderr,none": 0.02},
            },
            "n-samples": {"hellaswag": {"effective": 100}, "piqa": {"effective": 50}},
        }

    monkeypatch.setattr(harness, "evaluate", fake_evaluate)
    response = client.post(
        "/api/benchmarks",
        json={"model_id": str(model.id), "tasks": ["hellaswag", "piqa"], "limit": 50},
    )
    assert response.status_code == 202, response.text
    done = finished(client, response.json())
    assert done["status"] == "completed", done
    assert seen == {"tasks": ["hellaswag", "piqa"], "limit": 50}
    first = done["result"]["scores"][0]
    assert first["metric"] == "acc_norm" and first["chance"] == 0.25
    assert first["low"] == pytest.approx(0.31 - 1.96 * 0.01)
    assert first["high"] == pytest.approx(0.31 + 1.96 * 0.01)
    assert first["questions"] == 100

    side = client.get(f"/api/compare?model_ids={model.id}").json()["models"][0]
    assert side["benchmark"]["scores"][1]["name"] == "PIQA"
    assert side["license"] is None and side["bits_per_byte"] is None


@needs_hf
def test_two_models_answer_the_same_prompts_the_same_way(
    client,
    training_set,
    data_dir,
    monkeypatch,  # noqa: F811
):
    ours, _, save = trained(client, training_set)
    fake_hub(monkeypatch, exported(client, save, data_dir))
    theirs = imported(client)
    body = {
        "model_ids": [ours["id"], theirs["id"]],
        "prompts": ["Lesson 1.", "Text for"],
        "temperature": 0.9,
        "top_k": 20,
        "max_new_tokens": 8,
        "seed": 3,
    }
    first = finished(client, client.post("/api/prompt-comparisons", json=body).json())
    assert first["status"] == "completed", first
    rows = first["result"]["rows"]
    assert [row["prompt"] for row in rows] == ["Lesson 1.", "Text for"]
    assert all(len(row["answers"]) == 2 and row["answers"][0] == row["answers"][1] for row in rows)
    again = finished(client, client.post("/api/prompt-comparisons", json=body).json())
    assert again["result"]["rows"] == rows

    compared = client.get(f"/api/compare?model_ids={ours['id']}&model_ids={theirs['id']}").json()
    assert [side["source"] for side in compared["models"]] == ["forge", "hub"]
    assert compared["models"][0]["tokens_seen"] > 0
    assert compared["models"][1]["license"] == "mit"
    assert len(compared["prompt_runs"]) == 2


def test_prompt_comparison_needs_two_trained_models(client, training_set):  # noqa: F811
    first = make_model(client, training_set, name="a")
    second = make_model(client, training_set, name="b")
    body = {"model_ids": [first["id"], second["id"]], "prompts": ["Hi"]}
    response = client.post("/api/prompt-comparisons", json=body)
    assert response.status_code == 409
    assert "not been trained" in response.json()["error"]["message"]
    same = client.post("/api/prompt-comparisons", json={**body, "model_ids": [first["id"]] * 2})
    assert same.status_code == 422
    assert jobs.run_pending() == 0


def test_prompt_comparison_can_use_a_saved_prompt_set(client, training_set):
    first = make_model(client, training_set, name="a")
    second = make_model(client, training_set, name="b")
    saved = client.post("/api/prompt-sets", json={"name": "river", "prompts": ["Lesson 1."]})
    assert saved.status_code == 201, saved.text
    body = {"model_ids": [first["id"], second["id"]], "prompt_set_id": saved.json()["id"]}
    # the set is found; the models are not trained, so it stops there
    response = client.post("/api/prompt-comparisons", json=body)
    assert response.status_code == 409
    assert "not been trained" in response.json()["error"]["message"]
    empty = client.post("/api/prompt-comparisons", json={**body, "prompt_set_id": None})
    assert empty.json()["error"]["code"] == "no_prompts"


@needs_hf
def test_bits_per_byte_scores_an_imported_model(client, training_set, data_dir, monkeypatch):
    _, _, save = trained(client, training_set)
    fake_hub(monkeypatch, exported(client, save, data_dir))
    theirs = imported(client)
    response = client.post(
        "/api/bpb-evaluations",
        json={"model_id": theirs["id"], "dataset_id": training_set["dataset_id"]},
    )
    assert response.status_code == 201, response.text
    measured = response.json()
    assert measured["status"] == "completed", measured
    assert measured["model_source"] == "imported" and measured["bpb"] > 0
    side = client.get(f"/api/compare?model_ids={theirs['id']}").json()["models"][0]
    assert side["bits_per_byte"] == pytest.approx(measured["bpb"])


def test_bits_per_byte_reads_our_save_points(client, training_set):
    _, run, _ = trained(client, training_set)
    response = client.post(f"/api/runs/{run['id']}/bpb")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "completed", response.json()
    assert client.get(f"/api/runs/{run['id']}").json()["bpb"] > 0


@needs_hf
def test_prompt_set_evaluation_answers_with_the_trained_model(client, training_set):
    ours, _, _ = trained(client, training_set)
    saved = client.post("/api/prompt-sets", json={"name": "lessons", "prompts": ["Lesson 2."]})
    body = {"model_id": ours["id"], "prompt_set_id": saved.json()["id"], "max_new_tokens": 6}
    done = client.post("/api/evaluations", json=body).json()
    assert done["status"] == "completed", done
    assert "Evaluated response" not in done["results"][0]["output"]
    # the same seed and weights give the same answer
    again = client.post("/api/evaluations", json=body).json()
    assert again["results"][0]["output"] == done["results"][0]["output"]

    untrained = make_model(client, training_set, name="blank")
    failed = client.post("/api/evaluations", json={**body, "model_id": untrained["id"]}).json()
    assert failed["status"] == "failed" and "not been trained" in failed["error"]
