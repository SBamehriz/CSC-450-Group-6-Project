"""One local queue, one process at a time. No network worker protocol."""

import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from forge.config import ensure_data_dir, get_settings
from forge.db import get_engine
from forge.models import Dataset, DatasetBuild, Job, Tokenizer
from forge.schemas import JobPayload, JobProgress

log = logging.getLogger(__name__)

RUNNER_TYPES = (
    "tokenizer_train",
    "dataset_build",
    "dataset_import",
    "snapshot_create",
    "eval",
)


class Canceled(Exception):
    """The user asked us to stop; throw away unfinished artifacts."""


def target_for(session: Session, job: Job) -> Tokenizer | DatasetBuild:
    target_id = JobPayload.model_validate(job.payload).target_id
    kind = Tokenizer if job.job_type == "tokenizer_train" else DatasetBuild
    target = session.get(kind, target_id)
    if target is None:
        raise ValueError("The item this job was building no longer exists.")
    return target


def artifact_directory(target: Tokenizer | DatasetBuild) -> Path:
    folder = "tokenizers" if isinstance(target, Tokenizer) else "builds"
    return get_settings().data_dir / folder / str(target.id)


def _discard(target: Tokenizer | DatasetBuild) -> None:
    # A ready build is immutable, including during recovery.
    if target.status == "ready":
        return
    directory = artifact_directory(target)
    for path in (directory, directory.with_name(directory.name + ".tmp")):
        if path.exists():
            shutil.rmtree(path)


def _lock() -> FileLock:
    return FileLock(ensure_data_dir() / "runner.lock")


def _recover_unlocked(session: Session) -> None:
    # Holding the OS lock proves no old runner is still writing. A PID alone
    # cannot prove that: the OS may have reused it since the previous crash.
    running = select(Job).where(Job.status == "running", Job.job_type.in_(RUNNER_TYPES))
    for job in session.scalars(running):
        if job.job_type in ("dataset_import", "snapshot_create", "eval"):
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.error = (
                "Canceled before the operation was finished."
                if job.cancel_requested
                else "The background process stopped unexpectedly. Retry this job."
            )
            job.pid = None
            job.completed_at = datetime.now(UTC)
            if job.job_type == "snapshot_create":
                target_id = JobPayload.model_validate(job.payload).target_id
                target = session.get(Dataset, target_id)
                if target and target.status != "ready":
                    target.status = "failed"
            elif job.job_type == "eval":
                if job.payload.get("eval_type") == "bpb":
                    bpb_eval_id = job.payload.get("bpb_eval_id")
                    if bpb_eval_id:
                        from forge.models import BpbEvaluation

                        bpb_obj = session.get(BpbEvaluation, uuid.UUID(bpb_eval_id))
                        if bpb_obj and bpb_obj.status != "completed":
                            bpb_obj.status = "failed"
                            bpb_obj.error = job.error
                else:
                    eval_id = job.payload.get("evaluation_id")
                    if eval_id:
                        from forge.models import Evaluation

                        eval_obj = session.get(Evaluation, uuid.UUID(eval_id))
                        if eval_obj and eval_obj.status != "completed":
                            eval_obj.status = "failed"
                            eval_obj.error = job.error
            continue
        target = target_for(session, job)
        if target.status == "ready":
            job.status = "completed"
        else:
            _discard(target)
            job.status = target.status = "cancelled" if job.cancel_requested else "failed"
            job.error = target.error = (
                "Canceled before the artifact was finished."
                if job.cancel_requested
                else "The background process stopped unexpectedly. Retry this job."
            )
        job.pid = None
        job.completed_at = datetime.now(UTC)
    session.commit()


def recover_jobs() -> None:
    try:
        with _lock().acquire(timeout=0), Session(get_engine()) as session:
            _recover_unlocked(session)
    except Timeout:
        return  # the runner is alive and owns its current job


def start_runner() -> None:
    # Each enqueue gets a wake-up process. Waiting for the lock avoids losing a
    # newly queued job in the small window where an older runner is exiting.
    # This is a local lab queue (~10 jobs/day), not a fleet of workers.
    log_path = ensure_data_dir() / "jobs.log"
    settings = get_settings()
    database_url = settings.database_url
    if settings.database_path is not None:
        database_url = f"sqlite:///{settings.database_path.resolve()}"
    with log_path.open("ab") as output:
        subprocess.Popen(
            [sys.executable, "-m", "forge.jobs"],
            cwd=Path(__file__).resolve().parents[1],
            env={
                **os.environ,
                "FORGE_DATA_DIR": str(settings.data_dir.resolve()),
                "FORGE_DATABASE_URL": database_url,
            },
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt"
            else 0,
        )


def report(
    job_id: uuid.UUID,
    completed: int,
    total: int,
    message: str,
    *,
    processed: int = 0,
    imported: int = 0,
    skipped: int = 0,
    failed: int = 0,
) -> None:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.cancel_requested:
            raise Canceled()
        job.progress = JobProgress(
            completed=completed,
            total=total,
            message=message,
            processed=processed,
            imported=imported,
            skipped=skipped,
            failed=failed,
        ).model_dump()
        session.commit()


def execute_job(job_id: uuid.UUID) -> None:
    # The queue lock is held by run_pending for the whole operation. Only small
    # database transactions stay open; tokenizing never holds a write lock.
    directory: Path | None = None
    temporary: Path | None = None
    target_id: uuid.UUID | None = None

    with Session(get_engine()) as session:
        session.execute(text("BEGIN IMMEDIATE"))
        job = session.get(Job, job_id)
        if job is None or job.status != "pending":
            return
        kind = job.job_type
        if kind in ("dataset_import", "snapshot_create", "eval"):
            if job.cancel_requested:
                job.status = "cancelled"
                job.completed_at = datetime.now(UTC)
                if kind == "snapshot_create":
                    target_id = JobPayload.model_validate(job.payload).target_id
                    target = session.get(Dataset, target_id)
                    if target and target.status != "ready":
                        target.status = "failed"
                elif kind == "eval":
                    if job.payload.get("eval_type") == "bpb":
                        bpb_eval_id = job.payload.get("bpb_eval_id")
                        if bpb_eval_id:
                            from forge.models import BpbEvaluation

                            bpb_eval = session.get(BpbEvaluation, uuid.UUID(bpb_eval_id))
                            if bpb_eval and bpb_eval.status != "completed":
                                bpb_eval.status = "cancelled"
                    else:
                        eval_id = job.payload.get("evaluation_id")
                        if eval_id:
                            from forge.models import Evaluation

                            ev = session.get(Evaluation, uuid.UUID(eval_id))
                            if ev and ev.status != "completed":
                                ev.status = "cancelled"
                session.commit()
                return
            job.status = "running"
            job.pid = os.getpid()
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            job.error = None
            if kind == "snapshot_create":
                target_id = JobPayload.model_validate(job.payload).target_id
                target = session.get(Dataset, target_id)
                if target and target.status != "ready":
                    target.status = "processing"
            elif kind == "eval":
                if job.payload.get("eval_type") == "bpb":
                    bpb_eval_id = job.payload.get("bpb_eval_id")
                    if bpb_eval_id:
                        from forge.models import BpbEvaluation

                        bpb_eval = session.get(BpbEvaluation, uuid.UUID(bpb_eval_id))
                        if bpb_eval and bpb_eval.status != "completed":
                            bpb_eval.status = "running"
                            bpb_eval.started_at = datetime.now(UTC)
                else:
                    eval_id = job.payload.get("evaluation_id")
                    if eval_id:
                        from forge.models import Evaluation

                        ev = session.get(Evaluation, uuid.UUID(eval_id))
                        if ev and ev.status != "completed":
                            ev.status = "running"
                            ev.started_at = datetime.now(UTC)
            session.commit()
        else:
            target = target_for(session, job)
            if job.cancel_requested:
                job.status = target.status = "cancelled"
                job.completed_at = datetime.now(UTC)
                session.commit()
                return
            job.status, target.status = "running", "building"
            job.pid = os.getpid()
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            job.error = target.error = None
            session.commit()
            directory = artifact_directory(target)
            temporary = directory.with_name(directory.name + ".tmp")
            target_id = target.id

    if kind == "dataset_import":
        from forge.dataset_import import import_dataset

        try:
            import_dataset(job_id)
            with Session(get_engine()) as session:
                session.execute(text("BEGIN IMMEDIATE"))
                job = session.get(Job, job_id)
                assert job is not None
                if job.cancel_requested:
                    raise Canceled()
                job.status = "completed"
                job.pid = None
                job.completed_at = datetime.now(UTC)
                session.commit()
        except Exception as problem:
            log.exception("import job failed job_id=%s", job_id)
            with Session(get_engine()) as session:
                job = session.get(Job, job_id)
                assert job is not None
                canceled = isinstance(problem, Canceled) or job.cancel_requested
                job.status = "cancelled" if canceled else "failed"
                job.error = (
                    "Canceled before the import was finished."
                    if canceled
                    else f"Could not finish this import: {problem}"
                )
                job.pid = None
                job.completed_at = datetime.now(UTC)
                session.commit()
        return

    if kind == "snapshot_create":
        from forge.snapshots import execute_snapshot_job

        execute_snapshot_job(job_id)
        return

    if kind == "eval":
        if job.payload.get("eval_type") == "bpb":
            from forge.core.metrics import execute_bpb_eval_job

            execute_bpb_eval_job(job_id)
            return

        from forge.eval import execute_eval_job

        execute_eval_job(job_id)
        return

    from forge.builds import build_dataset, train_tokenizer

    assert directory is not None and temporary is not None and target_id is not None
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        if kind == "tokenizer_train":
            result = train_tokenizer(job_id, target_id, temporary)
        else:
            result = build_dataset(job_id, target_id, temporary)
        with Session(get_engine()) as session:
            # Cancel and publish must not pass each other. Take SQLite's write
            # lock before the final check, then publish and commit together.
            session.execute(text("BEGIN IMMEDIATE"))
            job = session.get(Job, job_id)
            assert job is not None
            if job.cancel_requested:
                raise Canceled()
            target = target_for(session, job)
            temporary.rename(directory)
            if isinstance(target, Tokenizer):
                target.vocab_size = result["vocab_size"]
                target.trained_on = result["trained_on"]
                target.artifact_uri = (
                    (directory / "tokenizer.json").relative_to(get_settings().data_dir).as_posix()
                )
            else:
                target.stats = result["stats"]
                target.manifest_uri = (
                    (directory / "manifest.json").relative_to(get_settings().data_dir).as_posix()
                )
            target.status, job.status = "ready", "completed"
            job.pid = None
            job.completed_at = datetime.now(UTC)
            session.commit()
    except Exception as problem:
        log.exception("job failed job_id=%s", job_id)
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            assert job is not None
            target = target_for(session, job)
            _discard(target)
            canceled = isinstance(problem, Canceled) or job.cancel_requested
            job.status = target.status = "cancelled" if canceled else "failed"
            job.error = target.error = (
                "Canceled before the artifact was finished."
                if canceled
                else f"Could not finish this job: {problem}"
            )
            job.pid = None
            job.completed_at = datetime.now(UTC)
            session.commit()


def run_pending() -> int:
    done = 0
    with _lock():
        with Session(get_engine()) as session:
            _recover_unlocked(session)
        while True:
            with Session(get_engine()) as session:
                job_id = session.scalar(
                    select(Job.id)
                    .where(Job.status == "pending", Job.job_type.in_(RUNNER_TYPES))
                    .order_by(Job.created_at, Job.id)
                    .limit(1)
                )
            if job_id is None:
                return done
            execute_job(job_id)
            done += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pending()
