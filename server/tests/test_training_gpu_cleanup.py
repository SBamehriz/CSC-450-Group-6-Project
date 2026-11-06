import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forge.models import Base, Model, Run
from forge.train.worker import TrainingWorker


def test_gpu_cleanup_after_successful_training(tmp_path):
    db_file = tmp_path / "gpu_cleanup.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)

    SessionMaker = sessionmaker(bind=engine, expire_on_commit=False)

    with SessionMaker() as session:
        model = Model(
            id=uuid.uuid4(),
            name="gpu-cleanup-test",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )

        run = Run(
            id=uuid.uuid4(),
            name="gpu-cleanup-run",
            model_id=model.id,
            max_steps=5,
        )

        session.add_all([model, run])
        session.commit()
        run_id = run.id

    worker = TrainingWorker(session_factory=SessionMaker)

    with patch("forge.train.worker.free_gpu_memory") as cleanup:
        result = worker.run_training_loop(
            run_id=run_id,
            steps=5,
        )

    assert result.status == "completed"
    cleanup.assert_called_once()

    engine.dispose()


def test_gpu_cleanup_after_stopped_training(tmp_path):
    from uuid import uuid4

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from forge.models import Base, Model, Run

    engine = create_engine(f"sqlite:///{tmp_path / 'gpu_stop_test.db'}")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine, expire_on_commit=False)

    with SessionMaker() as session:
        model = Model(
            id=uuid4(),
            name="gpu-stop-model",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(
            id=uuid4(),
            name="gpu-stop-run",
            model_id=model.id,
            max_steps=50,
        )
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    worker = TrainingWorker(session_factory=SessionMaker)

    with patch("forge.train.worker.free_gpu_memory") as cleanup:
        result = worker.run_training_loop(
            run_id=run_id,
            steps=50,
            stop_at_step=15,
        )

    assert result.status == "stopped"
    cleanup.assert_called_once()

    engine.dispose()


def test_gpu_cleanup_after_failed_training(tmp_path):
    from uuid import uuid4

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from forge.models import Base, Model, Run

    engine = create_engine(f"sqlite:///{tmp_path / 'gpu_fail_test.db'}")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine, expire_on_commit=False)

    with SessionMaker() as session:
        model = Model(
            id=uuid4(),
            name="gpu-fail-model",
            n_layer=4,
            d_model=128,
            n_head=4,
            ctx_len=512,
            vocab_size=16384,
            param_count=920000,
        )
        run = Run(
            id=uuid4(),
            name="gpu-fail-run",
            model_id=model.id,
            max_steps=50,
        )
        session.add_all([model, run])
        session.commit()
        run_id = run.id

    worker = TrainingWorker(session_factory=SessionMaker)

    with patch("forge.train.worker.free_gpu_memory") as cleanup:
        with pytest.raises(RuntimeError):
            worker.run_training_loop(
                run_id=run_id,
                steps=50,
                fail_at_step=10,
            )

    with SessionMaker() as session:
        failed_run = session.get(Run, run_id)
        assert failed_run is not None
        assert failed_run.status == "failed"

    cleanup.assert_called_once()
    engine.dispose()
