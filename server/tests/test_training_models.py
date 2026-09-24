import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from forge.models import (
    DATASET_STATUSES,
    JOB_STATUSES,
    JOB_TYPES,
    RUN_STATUSES,
    Bucket,
    Checkpoint,
    Dataset,
    Job,
    Model,
    Run,
)


def test_create_dataset_with_bucket(db, make_dataset):
    bucket = Bucket(name=f"dataset-bkt-{uuid.uuid4().hex[:6]}", description="bucket for dataset")
    db.add(bucket)
    db.commit()

    ds = make_dataset(name="pretrain-v1", bucket_id=bucket.id, status="draft")
    assert ds.id is not None
    assert ds.name == "pretrain-v1"
    assert ds.bucket_id == bucket.id
    assert ds.status == "draft"
    assert ds.doc_count == 10
    assert ds.token_count == 1250
    assert ds.config["val_split"] == 0.1

    # Status update
    ds.status = "ready"
    ds.token_count = 50000
    db.commit()

    reloaded = db.execute(select(Dataset).where(Dataset.id == ds.id)).scalar_one()
    assert reloaded.status == "ready"
    assert reloaded.token_count == 50000


def test_dataset_name_unique_constraint(db, make_dataset):
    make_dataset(name="duplicate-dataset-name")
    with pytest.raises(IntegrityError):
        make_dataset(name="duplicate-dataset-name")
    db.rollback()


def test_dataset_status_check_constraint(db):
    ds = Dataset(
        name=f"bad-status-{uuid.uuid4().hex[:6]}",
        status="non_existent_status",
    )
    db.add(ds)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_job_lifecycle(db, make_job):
    job = make_job(job_type="training", status="pending")
    assert job.id is not None
    assert job.status == "pending"
    assert job.job_type == "training"

    now = datetime.now(UTC)
    job.status = "running"
    job.started_at = now
    job.progress = {"step": 50, "total": 500, "pct": 10.0}
    db.commit()

    reloaded = db.execute(select(Job).where(Job.id == job.id)).scalar_one()
    assert reloaded.status == "running"
    assert reloaded.progress["step"] == 50

    job.status = "completed"
    job.completed_at = datetime.now(UTC)
    db.commit()

    finished = db.execute(select(Job).where(Job.id == job.id)).scalar_one()
    assert finished.status == "completed"
    assert finished.completed_at is not None


def test_job_check_constraints(db):
    bad_type = Job(job_type="invalid_type", status="pending")
    db.add(bad_type)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    bad_status = Job(job_type="training", status="invalid_status")
    db.add(bad_status)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_model_creation_and_uniqueness(db, make_model):
    m1 = make_model(name="tiny-gpt-nano", preset="nano")
    assert m1.id is not None
    assert m1.preset == "nano"
    assert m1.n_layer == 4
    assert m1.d_model == 128
    assert m1.n_head == 4
    assert m1.ctx_len == 512
    assert m1.vocab_size == 16384
    assert m1.param_count == 920000

    # Uniqueness on name
    with pytest.raises(IntegrityError):
        make_model(name="tiny-gpt-nano")
    db.rollback()


def test_run_lifecycle_and_metrics(db, make_model, make_dataset, make_run):
    model = make_model(name=f"model-for-run-{uuid.uuid4().hex[:6]}")
    dataset = make_dataset(name=f"dataset-for-run-{uuid.uuid4().hex[:6]}")

    run = make_run(model_id=model.id, dataset_id=dataset.id, status="pending")
    assert run.model_id == model.id
    assert run.dataset_id == dataset.id
    assert run.status == "pending"
    assert run.device == "cpu"
    assert run.batch_size == 2
    assert run.learning_rate == 0.001

    # Transition to running
    run.status = "running"
    run.current_step = 100
    run.loss = 2.05
    run.tokens_per_sec = 14500.0
    run.metrics = {"loss_history": [3.5, 2.8, 2.05]}
    db.commit()

    reloaded = db.execute(select(Run).where(Run.id == run.id)).scalar_one()
    assert reloaded.status == "running"
    assert reloaded.current_step == 100
    assert reloaded.loss == 2.05
    assert reloaded.tokens_per_sec == 14500.0
    assert len(reloaded.metrics["loss_history"]) == 3


def test_run_status_check_constraint(db, make_model):
    model = make_model(name=f"model-ck-{uuid.uuid4().hex[:6]}")
    bad_run = Run(
        name="bad-run",
        model_id=model.id,
        status="bogus_status",
    )
    db.add(bad_run)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_checkpoint_uniqueness_and_cascade_delete(db, make_run, make_checkpoint):
    run = make_run()
    ckpt1 = make_checkpoint(run_id=run.id, step=100, is_best=False)
    ckpt2 = make_checkpoint(run_id=run.id, step=200, is_best=True)

    assert ckpt1.id is not None
    assert ckpt2.id is not None
    assert ckpt2.is_best is True

    # Duplicate step for same run must violate unique constraint
    with pytest.raises(IntegrityError):
        make_checkpoint(run_id=run.id, step=100)
    db.rollback()

    # Verify CASCADE delete: deleting run deletes its checkpoints
    run_to_delete = db.execute(select(Run).where(Run.id == run.id)).scalar_one()
    db.delete(run_to_delete)
    db.commit()

    ckpts_left = db.execute(
        select(Checkpoint).where(Checkpoint.run_id == run.id)
    ).scalars().all()
    assert len(ckpts_left) == 0
