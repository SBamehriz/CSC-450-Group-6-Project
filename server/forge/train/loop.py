import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import JobFailed
from forge.models import Checkpoint, Job, Run


def run(job_id: uuid.UUID) -> dict:
    try:
        from forge.train.trainer import train
    except ImportError as problem:
        raise JobFailed(
            "Training needs PyTorch. Install the train extra from the README, then Resume."
        ) from problem
    return train(job_id)


def _run_of(session: Session, job: Job) -> Run | None:
    run_id = job.payload.get("run_id")
    return session.get(Run, uuid.UUID(run_id)) if run_id else None


def _last_save(session: Session, run_id: uuid.UUID) -> int | None:
    return session.scalar(
        select(Checkpoint.step).where(Checkpoint.run_id == run_id).order_by(Checkpoint.step.desc())
    )


def finish(job_id: uuid.UUID, status: str, message: str | None) -> None:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        run = _run_of(session, job)
        if run is None:
            return
        run.completed_at = datetime.now(UTC)
        if status == "completed":
            run.status, run.error = "completed", None
        elif status == "cancelled":
            run.status, run.error = "stopped", None
        else:
            run.status, run.error = "failed", message
        session.commit()


def recover(session: Session, job: Job) -> str | None:
    # the app or the computer stopped while this run was training
    run = _run_of(session, job)
    if run is None:
        return None
    saved = _last_save(session, run.id)
    if job.cancel_requested:
        run.status, run.error = "stopped", None
        message = "Stopped."
    else:
        where = (
            f"Resume to carry on from its last save point, at step {saved:,}."
            if saved is not None
            else "It had no save point yet, so start it again."
        )
        message = f"The app stopped while this run was training at step {run.current_step:,}. "
        message += where
        run.status, run.error = "failed", message
    run.completed_at = datetime.now(UTC)
    return message
