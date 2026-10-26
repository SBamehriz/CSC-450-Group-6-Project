import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from forge.models import Base, MetricPoint, Model, Run
from forge.train.metrics import MetricBuffer, get_best_validation_loss, get_metric_points
from forge.train.worker import TrainingWorker


def _setup_test_db(tmp_path):
    db_file = tmp_path / "metrics_test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, SessionMaker


def test_metric_buffer_batching_and_pending(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-buf",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-buf", model_id=model.id, status="running")
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    buffer = MetricBuffer(session_factory=SessionMaker, run_id=run_id, batch_size=20)
    assert buffer.pending_count == 0

    # Record 45 steps. Batches of 20 should flush at step 20 and step 40.
    for step in range(1, 46):
        buffer.record(
            step=step,
            loss=3.0 - (step * 0.02),
            val_loss=3.1 - (step * 0.02) if step % 10 == 0 else None,
            learning_rate=0.001,
            speed=12000.0,
        )

    # 40 persisted, 5 pending in memory
    assert buffer.pending_count == 5

    with SessionMaker() as session:
        persisted = session.execute(
            select(MetricPoint).where(MetricPoint.run_id == run_id)
        ).scalars().all()
        assert len(persisted) == 40

    # Flushing commits the remaining 5 points
    flushed = buffer.flush()
    assert flushed == 5
    assert buffer.pending_count == 0

    with SessionMaker() as session:
        persisted = session.execute(
            select(MetricPoint).where(MetricPoint.run_id == run_id)
        ).scalars().all()
        assert len(persisted) == 45
    engine.dispose()


def test_metric_buffer_flush_on_context_exit(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-ctx",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-ctx", model_id=model.id, status="running")
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    # Normal completion path
    with MetricBuffer(session_factory=SessionMaker, run_id=run_id, batch_size=25) as buf:
        for s in range(1, 11):  # 10 points < batch_size 25
            buf.record(step=s, loss=2.5, speed=10000.0)
        assert buf.pending_count == 10

    # Context exit flushed remaining points
    with SessionMaker() as session:
        points = get_metric_points(session, run_id)
        assert len(points) == 10
    engine.dispose()


def test_metric_buffer_flush_on_failure_path(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-fail",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-fail", model_id=model.id, status="running")
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    # Exception raised inside block
    with pytest.raises(RuntimeError, match="GPU memory exhausted"):
        with MetricBuffer(session_factory=SessionMaker, run_id=run_id, batch_size=50) as buf:
            for s in range(1, 15):
                buf.record(step=s, loss=2.0)
            raise RuntimeError("GPU memory exhausted at step 15")

    # Measurements before the failure must have been flushed!
    with SessionMaker() as session:
        points = get_metric_points(session, run_id)
        assert len(points) == 14
        assert points[-1].step == 14
    engine.dispose()


def test_metric_retry_behavior_no_duplicates(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-retry",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-retry", model_id=model.id, status="running")
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    # First attempt: records steps 1 to 5
    with MetricBuffer(session_factory=SessionMaker, run_id=run_id, batch_size=10) as buf:
        for s in range(1, 6):
            buf.record(step=s, loss=3.0, learning_rate=0.001)

    # Retry attempt: re-runs steps 3 to 7 with improved loss
    with MetricBuffer(session_factory=SessionMaker, run_id=run_id, batch_size=10) as buf:
        for s in range(3, 8):
            buf.record(step=s, loss=1.5, learning_rate=0.0005)

    with SessionMaker() as session:
        points = get_metric_points(session, run_id)
        # Should have exactly 7 steps (1 through 7), without duplicate key errors
        assert len(points) == 7
        steps = [p.step for p in points]
        assert steps == [1, 2, 3, 4, 5, 6, 7]
        # Step 3 must have the updated value from the retry
        p3 = next(p for p in points if p.step == 3)
        assert p3.loss == 1.5
        assert p3.learning_rate == 0.0005
    engine.dispose()


def test_metrics_survive_reopening_database(tmp_path):
    db_file = tmp_path / "durable_metrics.db"
    url = f"sqlite:///{db_file}"

    # Phase 1: create and write metrics
    engine1 = create_engine(url, future=True)
    Base.metadata.create_all(engine1)
    SM1 = sessionmaker(bind=engine1, expire_on_commit=False)
    with SM1() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-dur",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-dur", model_id=model.id, status="running")
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    with MetricBuffer(session_factory=SM1, run_id=run_id, batch_size=10) as buf:
        for s in range(1, 16):
            buf.record(step=s, loss=2.5 - s * 0.05, val_loss=2.0 if s == 10 else None)

    # Close and dispose the database completely
    engine1.dispose()

    # Phase 2: reopen with a fresh engine and verify committed data survived
    engine2 = create_engine(url, future=True)
    SM2 = sessionmaker(bind=engine2, expire_on_commit=False)
    with SM2() as session:
        persisted_points = get_metric_points(session, run_id)
        assert len(persisted_points) == 15
        assert persisted_points[0].step == 1
        assert persisted_points[-1].step == 15

        best_val = get_best_validation_loss(session, run_id)
        assert best_val == 2.0
    engine2.dispose()


def test_training_worker_integration(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-worker",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-worker", model_id=model.id, max_steps=30)
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    worker = TrainingWorker(session_factory=SessionMaker, metric_batch_size=10)
    finished_run = worker.run_training_loop(run_id=run_id, steps=25, eval_every=5)
    assert finished_run.status == "completed"
    assert finished_run.completed_at is not None

    with SessionMaker() as session:
        points = get_metric_points(session, run_id)
        assert len(points) == 25
        best_val = get_best_validation_loss(session, run_id)
        assert best_val is not None
        assert best_val < 3.5
    engine.dispose()


def test_training_worker_stop_path(tmp_path):
    engine, SessionMaker = _setup_test_db(tmp_path)
    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="m-stop",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(id=uuid.uuid4(), name="r-stop", model_id=model.id, max_steps=50)
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    worker = TrainingWorker(session_factory=SessionMaker, metric_batch_size=10)
    stopped_run = worker.run_training_loop(run_id=run_id, steps=50, stop_at_step=15)
    assert stopped_run.status == "stopped"

    with SessionMaker() as session:
        points = get_metric_points(session, run_id)
        # All 15 steps were recorded and flushed on the stop path
        assert len(points) == 15
    engine.dispose()
