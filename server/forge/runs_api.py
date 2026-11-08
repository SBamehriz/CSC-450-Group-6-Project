import io
import json
import uuid
import zipfile
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from forge import jobs
from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.jobs_api import queue_task
from forge.models import (
    BpbEvaluation,
    Checkpoint,
    Dataset,
    DatasetBuild,
    Job,
    MetricPoint,
    Model,
    Run,
    Tokenizer,
)
from forge.schemas import (
    BpbEvaluationOut,
    CheckpointOut,
    JobOut,
    MetricPointOut,
    Page,
    RunCreate,
    RunMetricsOut,
    RunOut,
)
from forge.train.metrics import get_best_validation_loss, get_metric_points
from forge.train.settings import TrainSettings, steps_for_tokens

router = APIRouter(tags=["runs"])


def _run_to_out(run: Run, session: Session) -> RunOut:
    model = session.get(Model, run.model_id) if run.model_id else None
    dataset = session.get(Dataset, run.dataset_id) if run.dataset_id else None
    best_val = get_best_validation_loss(session, run.id)
    build = session.get(DatasetBuild, run.build_id) if run.build_id else None
    job = _run_job(session, run.id)
    raw_settings = run.hyperparams.get("settings")

    # Resolve BPB if recorded in metrics or associated BpbEvaluation
    bpb_val: float | None = None
    bpb_details: dict | None = None
    if run.metrics and isinstance(run.metrics, dict):
        if "bpb" in run.metrics and isinstance(run.metrics["bpb"], (int, float)):
            bpb_val = float(run.metrics["bpb"])
            bpb_details = run.metrics.get("bpb_details")

    if bpb_val is None and run.id:
        completed_eval = session.execute(
            select(BpbEvaluation)
            .where(BpbEvaluation.run_id == run.id, BpbEvaluation.status == "completed")
            .order_by(BpbEvaluation.completed_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if completed_eval and completed_eval.bpb is not None:
            bpb_val = completed_eval.bpb
            bpb_details = {
                "bpb_eval_id": str(completed_eval.id),
                "bpb": completed_eval.bpb,
                "total_nll": completed_eval.total_nll,
                "scored_bytes": completed_eval.scored_bytes,
                "scored_tokens": completed_eval.scored_tokens,
                "validation_file_hash": completed_eval.validation_file_hash,
                "model_name": completed_eval.model_name,
                "tokenizer_identity": completed_eval.tokenizer_identity,
            }

    return RunOut(
        build_id=run.build_id,
        build_name=build.name if build else None,
        seed=run.seed,
        tokens_seen=run.tokens_seen,
        settings=TrainSettings.model_validate(raw_settings) if raw_settings else None,
        job_id=job.id if job else None,
        job_status=job.status if job else None,
        id=run.id,
        name=run.name,
        model_id=run.model_id,
        model_name=model.name if model else None,
        model_preset=model.preset if model else None,
        dataset_id=run.dataset_id,
        dataset_name=dataset.name if dataset else None,
        status=run.status,
        device=run.device,
        batch_size=run.batch_size,
        learning_rate=run.learning_rate,
        max_steps=run.max_steps,
        current_step=run.current_step,
        loss=run.loss,
        eval_loss=run.eval_loss,
        best_val_loss=best_val,
        bpb=bpb_val,
        bpb_details=bpb_details,
        tokens_per_sec=run.tokens_per_sec,
        metrics=run.metrics,
        hyperparams=run.hyperparams,
        error=run.error,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
    )


@router.get("/runs", response_model=Page[RunOut])
def list_runs(
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> Page[RunOut]:
    """List training runs newest first, with stable pagination."""
    query = select(Run)
    if status:
        query = query.where(Run.status == status)

    total_query = select(func.count(Run.id))
    if status:
        total_query = total_query.where(Run.status == status)
    total = session.scalar(total_query) or 0

    records = (
        session.execute(
            query.order_by(Run.created_at.desc(), Run.id.desc()).offset(offset).limit(limit)
        )
        .scalars()
        .all()
    )

    return Page[RunOut](items=[_run_to_out(r, session) for r in records], total=total)


@router.post("/runs", response_model=RunOut, status_code=201)
def create_run(
    payload: RunCreate,
    session: Session = Depends(get_session),
) -> RunOut:
    """Create a training run: a model, a training set and its settings."""
    model = session.get(Model, payload.model_id)
    if not model:
        raise ApiError(404, "not_found", "Selected model was not found.")
    if model.source != "forge":
        raise ApiError(
            409, "not_trainable", "Imported models are measured here, not trained from scratch."
        )

    dataset_id = payload.dataset_id
    if dataset_id and not session.get(Dataset, dataset_id):
        raise ApiError(404, "not_found", "Selected dataset was not found.")

    settings = payload.settings.model_copy()
    if payload.batch_size:
        settings.batch_size = payload.batch_size
    if payload.learning_rate:
        settings.learning_rate = payload.learning_rate
    if payload.max_steps:
        settings.max_steps = payload.max_steps
    TrainSettings.model_validate(settings.model_dump())

    if payload.build_id:
        build = _usable_build(session, payload.build_id)
        _vocabulary_matches(session, model, build)
        dataset_id = dataset_id or build.dataset_id
        if payload.train_tokens:
            settings.max_steps = steps_for_tokens(
                payload.train_tokens, settings.batch_size, model.ctx_len
            )

    run = Run(
        id=uuid.uuid4(),
        name=payload.name.strip(),
        model_id=payload.model_id,
        dataset_id=dataset_id,
        build_id=payload.build_id,
        seed=payload.seed,
        device=payload.device,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        max_steps=settings.max_steps,
        status="pending",
        hyperparams={**payload.hyperparams, "settings": settings.model_dump()},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return _run_to_out(run, session)


def _usable_build(session: Session, build_id: uuid.UUID) -> DatasetBuild:
    build = session.get(DatasetBuild, build_id)
    if build is None:
        raise ApiError(404, "not_found", "Selected training set was not found.")
    if build.status != "ready":
        raise ApiError(409, "build_not_ready", "This training set isn't ready yet.")
    return build


def _vocabulary_matches(session: Session, model: Model, build: DatasetBuild) -> None:
    vocabulary = session.get(Tokenizer, build.tokenizer_id)
    name = vocabulary.name if vocabulary else "another vocabulary"
    if model.tokenizer_id is not None:
        if model.tokenizer_id != build.tokenizer_id:
            raise ApiError(
                409,
                "vocabulary_mismatch",
                f"This training set was made with the vocabulary {name!r}, but the model "
                "reads a different one. Choose a training set made with the model's vocabulary.",
            )
        return
    if vocabulary is None or vocabulary.vocab_size != model.vocab_size:
        raise ApiError(
            409,
            "vocabulary_mismatch",
            f"The model reads {model.vocab_size:,} word pieces, but this training set uses "
            f"{name!r}. Make a model with that vocabulary first.",
        )


def _run_job(session: Session, run_id: uuid.UUID) -> Job | None:
    # the newest training job for this run
    for job in session.scalars(
        select(Job).where(Job.job_type == "train_run").order_by(Job.created_at.desc())
    ):
        if job.payload.get("run_id") == str(run_id):
            return job
    return None


def _queue(session: Session, run: Run, resume: bool) -> Job:
    job = jobs.enqueue_task(
        session, "train_run", {"run_id": str(run.id), "resume": resume, "target_name": run.name}
    )
    run.status, run.error, run.completed_at = "pending", None, None
    session.commit()
    try:
        jobs.start_runner()
    except OSError as problem:
        message = f"The background process could not start: {problem}"
        job.status, run.status = "failed", "failed"
        job.error = run.error = message
        session.commit()
        raise ApiError(503, "runner_unavailable", message) from problem
    return job


def _active(session: Session, run: Run) -> Job | None:
    job = _run_job(session, run.id)
    return job if job and job.status in ("pending", "running") else None


@router.post("/runs/{run_id}/start", response_model=RunOut)
def start_run(run_id: uuid.UUID, session: Session = Depends(get_session)) -> RunOut:
    jobs.recover_jobs()
    run = _load_run(session, run_id)
    if _active(session, run):
        raise ApiError(409, "already_running", "This run is already queued or training.")
    if run.status == "completed" or session.scalar(
        select(Checkpoint.id).where(Checkpoint.run_id == run.id)
    ):
        raise ApiError(409, "already_started", "This run already trained. Use Resume instead.")
    # stopped before its first save point: start again from the beginning
    session.execute(delete(MetricPoint).where(MetricPoint.run_id == run.id))
    run.current_step, run.tokens_seen = 0, 0
    if run.build_id is None:
        raise ApiError(409, "no_training_set", "Choose a training set for this run first.")
    model = session.get(Model, run.model_id)
    assert model is not None
    _vocabulary_matches(session, model, _usable_build(session, run.build_id))
    _queue(session, run, resume=False)
    return _run_to_out(run, session)


@router.post("/runs/{run_id}/stop", response_model=RunOut)
def stop_run(run_id: uuid.UUID, session: Session = Depends(get_session)) -> RunOut:
    jobs.recover_jobs()
    session.execute(text("BEGIN IMMEDIATE"))
    run = _load_run(session, run_id)
    job = _active(session, run)
    if job is None:
        raise ApiError(409, "not_running", "This run isn't training.")
    job.cancel_requested = True
    if job.status == "pending":
        job.status, run.status = "cancelled", "stopped"
        job.completed_at = run.completed_at = datetime.now(UTC)
    session.commit()
    return _run_to_out(run, session)


@router.post("/runs/{run_id}/resume", response_model=RunOut)
def resume_run(run_id: uuid.UUID, session: Session = Depends(get_session)) -> RunOut:
    jobs.recover_jobs()
    run = _load_run(session, run_id)
    if _active(session, run):
        raise ApiError(409, "already_running", "This run is already queued or training.")
    if run.status == "completed":
        raise ApiError(409, "finished", "This run already finished.")
    if not session.scalar(select(Checkpoint.id).where(Checkpoint.run_id == run.id)):
        raise ApiError(409, "no_save_point", "There is no save point yet. Start the run instead.")
    _queue(session, run, resume=True)
    return _run_to_out(run, session)


def _checkpoint_out(row: Checkpoint) -> CheckpointOut:
    return CheckpointOut(
        id=row.id,
        run_id=row.run_id,
        step=row.step,
        epoch=row.epoch,
        loss=row.loss,
        val_loss=row.val_loss,
        uri=row.uri,
        size_bytes=row.size_bytes,
        is_best=row.is_best,
        metrics=row.metrics,
        sha256=row.sha256,
        created_at=row.created_at,
    )


@router.get("/runs/{run_id}/checkpoints", response_model=list[CheckpointOut])
def list_checkpoints(run_id: uuid.UUID, session: Session = Depends(get_session)):
    _load_run(session, run_id)
    rows = session.scalars(
        select(Checkpoint).where(Checkpoint.run_id == run_id).order_by(Checkpoint.step.desc())
    )
    return [_checkpoint_out(row) for row in rows]


@router.get("/checkpoints/{checkpoint_id}/download")
def download_checkpoint(checkpoint_id: uuid.UUID, session: Session = Depends(get_session)):
    """One zip: the weights, the model's shape and its vocabulary."""
    checkpoint = session.get(Checkpoint, checkpoint_id)
    if checkpoint is None:
        raise ApiError(404, "not_found", "No save point with that id.")
    run = session.get(Run, checkpoint.run_id)
    model = session.get(Model, run.model_id) if run else None
    path = get_settings().data_dir / checkpoint.uri
    if not path.is_file():
        raise ApiError(404, "missing_file", "This save point's file is missing from the disk.")
    from forge.train.trainer import file_sha256

    if checkpoint.sha256 and file_sha256(path) != checkpoint.sha256:
        raise ApiError(
            409, "changed_on_disk", "This save point changed on disk since it was saved."
        )
    vocabulary = None
    build = session.get(DatasetBuild, run.build_id) if run and run.build_id else None
    if build is not None:
        vocabulary = session.get(Tokenizer, build.tokenizer_id)
    shape = {
        "format": "forge-gpt-v1",
        "run": run.name if run else None,
        "model": model.name if model else None,
        "step": checkpoint.step,
        "n_layer": model.n_layer if model else None,
        "d_model": model.d_model if model else None,
        "n_head": model.n_head if model else None,
        "ctx_len": model.ctx_len if model else None,
        "vocab_size": model.vocab_size if model else None,
        "checkpoint_sha256": checkpoint.sha256,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.write(path, "save_point.pt")
        archive.writestr("config.json", json.dumps(shape, indent=2))
        if vocabulary and vocabulary.artifact_uri:
            vocab_path = get_settings().data_dir / vocabulary.artifact_uri
            if vocab_path.is_file():
                archive.write(vocab_path, "tokenizer.json")
    buffer.seek(0)
    name = f"{(run.name if run else 'run').replace(' ', '-')}-step-{checkpoint.step}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/checkpoints/{checkpoint_id}/export", response_model=JobOut, status_code=202)
def export_checkpoint(checkpoint_id: uuid.UUID, session: Session = Depends(get_session)) -> JobOut:
    """Turn a save point into a Hugging Face folder other tools load."""
    checkpoint = session.get(Checkpoint, checkpoint_id)
    if checkpoint is None:
        raise ApiError(404, "not_found", "No save point with that id.")
    run = session.get(Run, checkpoint.run_id)
    return queue_task(
        session,
        "model_export",
        {
            "checkpoint_id": str(checkpoint_id),
            "target_name": f"{run.name if run else 'Run'}, step {checkpoint.step:,}",
        },
    )


@router.get("/checkpoints/{checkpoint_id}/export")
def download_export(checkpoint_id: uuid.UUID, session: Session = Depends(get_session)):
    if session.get(Checkpoint, checkpoint_id) is None:
        raise ApiError(404, "not_found", "No save point with that id.")
    folder = get_settings().data_dir / "exports" / str(checkpoint_id)
    if not folder.is_dir():
        raise ApiError(404, "not_exported", "This save point has not been exported yet.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            archive.write(path, path.name)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="hf-{checkpoint_id}.zip"'},
    )


def _load_run(session: Session, run_id: uuid.UUID) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise ApiError(404, "not_found", "Training run not found.")
    return run


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(
    run_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> RunOut:
    """Get detailed information for a single training run."""
    run = session.get(Run, run_id)
    if not run:
        raise ApiError(404, "not_found", "Training run not found.")
    return _run_to_out(run, session)


@router.get("/runs/{run_id}/metrics", response_model=RunMetricsOut)
def get_run_metrics(
    run_id: uuid.UUID,
    limit: int | None = Query(default=None, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> RunMetricsOut:
    """Retrieve chronologically ordered metric points for chart display."""
    run = session.get(Run, run_id)
    if not run:
        raise ApiError(404, "not_found", "Training run not found.")

    points = get_metric_points(session, run_id, limit=limit, offset=offset)
    total = session.scalar(
        select(func.count()).select_from(Checkpoint).where(Checkpoint.run_id == run_id)
    ) or len(points)

    items = [
        MetricPointOut(
            id=p.id,
            run_id=p.run_id,
            step=p.step,
            loss=p.loss,
            val_loss=p.val_loss,
            learning_rate=p.learning_rate,
            speed=p.speed,
            tokens_per_sec=p.speed,
            created_at=p.created_at,
        )
        for p in points
    ]
    return RunMetricsOut(run_id=run_id, items=items, total=total)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(
    run_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a run only when no dependent result requires it."""
    run = session.get(Run, run_id)
    if not run:
        raise ApiError(404, "not_found", "Training run not found.")

    # Rule 1: Cannot delete an active run
    if run.status == "running":
        raise ApiError(
            400,
            "run_running",
            "Cannot delete a training run while it is actively running. Stop the run first.",
        )

    # Rule 2: Cannot delete a run if dependent checkpoint results exist
    checkpoint_count = (
        session.scalar(select(func.count(Checkpoint.id)).where(Checkpoint.run_id == run_id)) or 0
    )
    if checkpoint_count > 0:
        raise ApiError(
            409,
            "run_has_results",
            f"Cannot delete run '{run.name}' because it has {checkpoint_count} dependent "
            "checkpoint result(s). Remove checkpoint results before deleting the run.",
        )

    # Rule 3: Cannot delete if an associated job is actively running
    active_jobs = (
        session.scalar(
            select(func.count(Job.id)).where(
                Job.job_type == "training",
                Job.status == "running",
            )
        )
        or 0
    )
    if active_jobs > 0:
        # Check payload if relevant
        jobs = (
            session.execute(select(Job).where(Job.job_type == "training", Job.status == "running"))
            .scalars()
            .all()
        )
        for j in jobs:
            if j.payload.get("run_id") == str(run_id):
                raise ApiError(
                    409,
                    "job_running",
                    "Cannot delete run while an associated training job is running.",
                )

    session.delete(run)
    session.commit()
    return Response(status_code=204)


@router.post("/runs/{run_id}/bpb", response_model=BpbEvaluationOut, status_code=202)
def compute_run_bpb(
    run_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> BpbEvaluationOut:
    """Launch a real bits-per-byte recomputation for a run using its saved model and snapshot."""
    import os

    from forge.core.metrics import _get_validation_file_info
    from forge.eval_api import _bpb_eval_to_out

    run = session.get(Run, run_id)
    if not run:
        raise ApiError(404, "not_found", "Training run not found.")

    if not run.model_id:
        raise ApiError(400, "missing_model", "Run has no associated model for BPB calculation.")
    model = session.get(Model, run.model_id)
    if not model:
        raise ApiError(404, "not_found", "Associated model was removed.")

    if not run.dataset_id:
        raise ApiError(
            400, "missing_dataset", "Run has no associated dataset snapshot for BPB calculation."
        )
    dataset = session.get(Dataset, run.dataset_id)
    if not dataset or dataset.status != "ready":
        raise ApiError(
            400, "snapshot_not_ready", "Associated dataset snapshot is not ready or was removed."
        )

    _, val_hash = _get_validation_file_info(dataset)

    latest_ckpt = session.execute(
        select(Checkpoint)
        .where(Checkpoint.run_id == run.id)
        .order_by(Checkpoint.step.desc())
        .limit(1)
    ).scalar_one_or_none()

    eval_id = uuid.uuid4()
    job_id = uuid.uuid4()
    eval_name = f"BPB - Run {run.name} ({model.name})"[:128]

    bpb_eval = BpbEvaluation(
        id=eval_id,
        name=eval_name,
        model_id=model.id,
        model_name=model.name,
        model_source="native",
        checkpoint_id=latest_ckpt.id if latest_ckpt else None,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        validation_file_hash=val_hash,
        tokenizer_identity="native-bpe",
        bpb=None,
        total_nll=None,
        scored_bytes=None,
        scored_tokens=None,
        status="pending",
        error=None,
        settings={
            "ctx_len": model.ctx_len,
            "run_id": str(run.id),
            "checkpoint_id": str(latest_ckpt.id) if latest_ckpt else None,
        },
        job_id=job_id,
        run_id=run.id,
    )
    session.add(bpb_eval)

    job = Job(
        id=job_id,
        job_type="eval",
        status="pending",
        payload={
            "eval_type": "bpb",
            "bpb_eval_id": str(eval_id),
            "model_id": str(model.id),
            "model_source": "native",
            "checkpoint_id": str(latest_ckpt.id) if latest_ckpt else None,
            "dataset_id": str(dataset.id),
            "ctx_len": model.ctx_len,
            "run_id": str(run.id),
        },
        progress={"current": 0, "total": 1, "stage": "Queued"},
    )
    session.add(job)
    session.commit()
    session.refresh(bpb_eval)

    if os.environ.get("PYTEST_CURRENT_TEST"):
        from forge.core.metrics import execute_bpb_eval_job

        execute_bpb_eval_job(job_id)
        session.refresh(bpb_eval)
    else:
        try:
            from forge.jobs import start_runner

            start_runner()
        except OSError:
            pass

    return _bpb_eval_to_out(bpb_eval)
