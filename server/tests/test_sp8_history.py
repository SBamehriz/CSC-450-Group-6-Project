import uuid
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from forge.models import Checkpoint, MetricPoint, Run


def test_history_ordering_newest_first(client: TestClient, db, make_model):
    model = make_model()
    now = datetime.now(UTC)

    run_old = Run(
        id=uuid.uuid4(),
        name="run-old",
        model_id=model.id,
        created_at=now - timedelta(hours=2),
    )
    run_mid = Run(
        id=uuid.uuid4(),
        name="run-mid",
        model_id=model.id,
        created_at=now - timedelta(hours=1),
    )
    run_new = Run(
        id=uuid.uuid4(),
        name="run-new",
        model_id=model.id,
        created_at=now,
    )
    db.add_all([run_old, run_mid, run_new])
    db.commit()

    resp = client.get("/api/runs?limit=10")
    assert resp.status_code == 200
    items = resp.json()["items"]
    names = [r["name"] for r in items]

    # Verify run-new appears before run-mid, and run-mid before run-old
    new_idx = names.index("run-new")
    mid_idx = names.index("run-mid")
    old_idx = names.index("run-old")
    assert new_idx < mid_idx < old_idx


def test_history_stable_pagination(client: TestClient, db, make_model):
    model = make_model()
    prefix = f"page-test-{uuid.uuid4().hex[:4]}"
    now = datetime.now(UTC)

    runs = [
        Run(
            id=uuid.uuid4(),
            name=f"{prefix}-{i}",
            model_id=model.id,
            created_at=now - timedelta(minutes=i),
        )
        for i in range(5)
    ]
    db.add_all(runs)
    db.commit()

    p1 = client.get("/api/runs?limit=2&offset=0").json()
    p2 = client.get("/api/runs?limit=2&offset=2").json()
    p3 = client.get("/api/runs?limit=2&offset=4").json()

    assert len(p1["items"]) == 2
    assert len(p2["items"]) == 2
    assert len(p3["items"]) >= 1

    ids1 = {r["id"] for r in p1["items"]}
    ids2 = {r["id"] for r in p2["items"]}
    # No overlap between distinct pages (stable pagination)
    assert ids1.isdisjoint(ids2)


def test_history_shows_model_and_dataset(client: TestClient, db, make_model, make_dataset):
    model = make_model(name=f"hist-model-{uuid.uuid4().hex[:6]}", preset="micro")
    dataset = make_dataset(name=f"hist-ds-{uuid.uuid4().hex[:6]}")

    run = Run(
        id=uuid.uuid4(),
        name=f"run-assoc-{uuid.uuid4().hex[:6]}",
        model_id=model.id,
        dataset_id=dataset.id,
        status="completed",
    )
    db.add(run)
    db.commit()

    resp = client.get(f"/api/runs/{run.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == model.name
    assert data["model_preset"] == "micro"
    assert data["dataset_name"] == dataset.name


def test_history_missing_val_loss_is_none(client: TestClient, db, make_model):
    model = make_model()
    run = Run(
        id=uuid.uuid4(),
        name=f"run-no-val-{uuid.uuid4().hex[:6]}",
        model_id=model.id,
        status="completed",
        eval_loss=None,
    )
    db.add(run)
    db.commit()

    resp = client.get(f"/api/runs/{run.id}")
    assert resp.status_code == 200
    assert resp.json()["best_val_loss"] is None


def test_history_best_validation_loss_calculated(client: TestClient, db, make_model):
    model = make_model()
    run = Run(
        id=uuid.uuid4(),
        name=f"run-val-{uuid.uuid4().hex[:6]}",
        model_id=model.id,
        status="completed",
        eval_loss=2.8,
    )
    db.add(run)
    db.commit()

    # Checkpoint with better validation loss
    ckpt = Checkpoint(
        id=uuid.uuid4(),
        run_id=run.id,
        step=50,
        loss=2.5,
        val_loss=2.1,
        uri="dummy.pt",
        is_best=True,
    )
    # Metric point with even better validation loss
    mp = MetricPoint(
        id=uuid.uuid4(),
        run_id=run.id,
        step=75,
        loss=2.0,
        val_loss=1.85,
    )
    db.add_all([ckpt, mp])
    db.commit()

    resp = client.get(f"/api/runs/{run.id}")
    assert resp.status_code == 200
    assert resp.json()["best_val_loss"] == 1.85


def test_run_deletion_blocked_when_running(client: TestClient, db, make_model):
    model = make_model()
    run = Run(
        id=uuid.uuid4(),
        name="active-run",
        model_id=model.id,
        status="running",
    )
    db.add(run)
    db.commit()

    resp = client.delete(f"/api/runs/{run.id}")
    assert resp.status_code == 400
    assert "actively running" in resp.json()["error"]["message"]


def test_run_deletion_blocked_when_checkpoints_exist(client: TestClient, db, make_model):
    model = make_model()
    run = Run(
        id=uuid.uuid4(),
        name="run-with-ckpt",
        model_id=model.id,
        status="completed",
    )
    db.add(run)
    db.commit()

    ckpt = Checkpoint(
        id=uuid.uuid4(),
        run_id=run.id,
        step=100,
        loss=2.0,
        val_loss=1.9,
        uri="ckpt.pt",
        is_best=True,
    )
    db.add(ckpt)
    db.commit()

    resp = client.delete(f"/api/runs/{run.id}")
    assert resp.status_code == 409
    assert "dependent checkpoint result" in resp.json()["error"]["message"]


def test_run_deletion_allowed_when_no_dependent_results(client: TestClient, db, make_model):
    model = make_model()
    run = Run(
        id=uuid.uuid4(),
        name="clean-run",
        model_id=model.id,
        status="failed",
    )
    db.add(run)
    db.commit()

    # Add metric points (they cascade delete, not independent checkpoint artifacts)
    mp = MetricPoint(id=uuid.uuid4(), run_id=run.id, step=1, loss=4.0)
    db.add(mp)
    db.commit()

    resp = client.delete(f"/api/runs/{run.id}")
    assert resp.status_code == 204


def test_run_metrics_data_access_endpoint(client: TestClient, db, make_model):
    model = make_model()
    run = Run(id=uuid.uuid4(), name="run-metrics-ep", model_id=model.id, status="completed")
    db.add(run)
    db.commit()

    # Add 3 metric points
    for s in (10, 20, 30):
        db.add(
            MetricPoint(
                id=uuid.uuid4(),
                run_id=run.id,
                step=s,
                loss=3.0 - (s * 0.05),
                speed=15000.0,
            )
        )
    db.commit()

    resp = client.get(f"/api/runs/{run.id}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == str(run.id)
    assert len(data["items"]) == 3
    steps = [p["step"] for p in data["items"]]
    assert steps == [10, 20, 30]
