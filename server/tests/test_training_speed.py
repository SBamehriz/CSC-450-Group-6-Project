"""Benchmark tests for simulated training performance (SP9.3)."""

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from forge.models import Base, Model, Run
from forge.train.speed_test import (
    calculate_steps_per_second,
    measure_execution_time,
)
from forge.train.worker import TrainingWorker


def create_test_database():
    """Create a temporary SQLite database for speed testing."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    SessionMaker = sessionmaker(bind=engine, expire_on_commit=False)

    return engine, SessionMaker


def test_training_worker_speed():
    """Measure the speed of a simulated training run."""

    engine, SessionMaker = create_test_database()

    try:
        with SessionMaker() as session:
            model = Model(
                id=uuid.uuid4(),
                name="speed-test-model",
                n_layer=4,
                d_model=128,
                n_head=4,
                ctx_len=512,
                vocab_size=16384,
                param_count=920000,
            )

            run = Run(
                id=uuid.uuid4(),
                name="speed-test-run",
                model_id=model.id,
                max_steps=100,
            )

            session.add_all([model, run])
            session.commit()
            run_id = run.id

        worker = TrainingWorker(
            session_factory=SessionMaker,
            metric_batch_size=10,
        )

        finished_run, elapsed_seconds = measure_execution_time(
            lambda: worker.run_training_loop(
                run_id=run_id,
                steps=100,
                eval_every=10,
            )
        )

        steps_per_second = calculate_steps_per_second(100, elapsed_seconds)

        print(f"\nTraining time: {elapsed_seconds:.4f} seconds")
        print(f"Training speed: {steps_per_second:.2f} steps/second")

        assert finished_run.status == "completed"
        assert steps_per_second > 0

    finally:
        engine.dispose()
