# ruff: noqa: F811
import pytest
from sqlalchemy import update
from test_training_loop import make_model, training_set  # noqa: F401

from forge import cli, jobs
from forge.models import Job

torch = pytest.importorskip("torch")

from forge.train import speed, trainer  # noqa: E402


@pytest.fixture(autouse=True)
def runner_in_this_process(monkeypatch):
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    monkeypatch.setattr(cli, "POLL_S", 0.01)
    monkeypatch.setattr(speed, "MOST_STEPS", 4)
    monkeypatch.setattr("forge.gpu.TOOL", ("forge-no-such-tool",))


def forge(capsys, *argv):
    # the app wakes a background process; here the job runs straight away
    quiet = jobs.start_runner
    jobs.start_runner = jobs.run_pending
    try:
        code = cli.main(list(argv))
    finally:
        jobs.start_runner = quiet
    out = capsys.readouterr()
    return code, out.out + out.err


def test_info_and_list_name_what_is_there(client, capsys, training_set):
    make_model(client, training_set, name="tiny")
    code, said = forge(capsys, "info")
    assert code == 0 and "Data folder:" in said and "PyTorch:" in said
    assert "Graphics card: none found" in said
    code, said = forge(capsys, "list")
    assert code == 0 and "tiny" in said and training_set["name"] in said


def test_train_makes_a_run_and_the_same_command_resumes_it(
    client, db, capsys, training_set, monkeypatch
):
    make_model(client, training_set, name="tiny")
    real_report = jobs.report

    def stop_at_ten(job_id, completed, total, message):
        if completed == 10:
            db.execute(update(Job).where(Job.id == job_id).values(cancel_requested=True))
            db.commit()
        real_report(job_id, completed, total, message)

    monkeypatch.setattr(trainer, "report", stop_at_ten)
    flags = ["--model", "tiny", "--set", training_set["name"], "--steps", "20", "--batch", "4"]
    code, said = forge(capsys, "train", "--name", "main", "--device", "cpu", *flags)
    assert code == 0, said
    assert "Made the run 'main': 20 steps." in said
    assert "Stopped 'main' at step 10. Run the same command to go on." in said

    monkeypatch.setattr(trainer, "report", real_report)
    code, said = forge(capsys, "train", "--name", "main")
    assert code == 0, said
    assert "Resuming 'main' from its last save point." in said
    assert "Finished 'main' at step 20." in said
    run = client.get("/api/runs").json()["items"][0]
    assert run["name"] == "main" and run["status"] == "completed"

    code, said = forge(capsys, "train", "--name", "main")
    assert code == 1 and "already finished" in said


def test_train_explains_what_is_missing(client, capsys, training_set):
    code, said = forge(capsys, "train", "--name", "new")
    assert code == 1 and "Give --model and --set" in said
    code, said = forge(capsys, "train", "--name", "new", "--model", "nope", "--set", "x")
    assert code == 1 and "No model called 'nope'" in said


def test_speed_test_then_budget_shows_its_arithmetic(capsys):
    code, said = forge(capsys, "budget", "--size", "nano", "--hours", "2")
    assert code == 1 and "forge speed-test --size nano" in said
    code, said = forge(
        capsys, "speed-test", "--size", "nano", "--sequences", "2", "--device", "cpu"
    )
    assert code == 0, said
    assert "The CPU:" in said and "word pieces per second" in said
    code, said = forge(capsys, "budget", "--size", "nano", "--sequences", "2", "--hours", "2")
    assert code == 0, said
    assert "measured on The CPU" in said
    assert "x 2 h x 3,600 s x 0.9" in said
    assert "word pieces per parameter" in said
    assert "forge train ... --tokens" in said


def test_sweep_ranks_short_runs_by_validation_loss(client, capsys, training_set):
    make_model(client, training_set, name="tiny")
    code, said = forge(
        capsys,
        "sweep",
        "--model",
        "tiny",
        "--set",
        training_set["name"],
        "--steps",
        "6",
        "--lr",
        "1e-4,3e-3",
        "--batch",
        "4",
        "--eval-every",
        "3",
        "--device",
        "cpu",
    )
    assert code == 0, said
    assert "2 runs queued" in said
    assert "sweep lr 0.0001 batch 4" in said and "sweep lr 0.003 batch 4" in said
    assert "Lowest validation loss: sweep lr" in said
    runs = client.get("/api/runs").json()["items"]
    assert sorted(run["status"] for run in runs) == ["completed", "completed"]


def test_a_time_limit_saves_and_stops_the_run(client, capsys, training_set):
    model = make_model(client, training_set, name="tiny")
    run = client.post(
        "/api/runs",
        json={"name": "limited", "model_id": model["id"], "build_id": training_set["id"]},
    ).json()
    # queued but not picked up yet, then the time limit passes
    assert client.post(f"/api/runs/{run['id']}/start").status_code == 200
    import uuid

    ended = cli._follow(uuid.UUID(run["id"]), stop_after_s=0)
    assert ended.status == "stopped"
    assert "Time limit reached: saving and stopping." in capsys.readouterr().out
