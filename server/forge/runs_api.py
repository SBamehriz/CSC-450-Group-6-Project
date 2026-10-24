import uuid

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Checkpoint, Dataset, Job, Model, Run
from forge.schemas import (
    MetricPointOut,
    Page,
    RunCreate,
    RunMetricsOut,
    RunOut,
)
from forge.train.metrics import get_best_validation_loss, get_metric_points

router = APIRouter(tags=["runs"])


def _run_to_out(run: Run, session: Session) -> RunOut:
    model = session.get(Model, run.model_id) if run.model_id else None
    dataset = session.get(Dataset, run.dataset_id) if run.dataset_id else None
    best_val = get_best_validation_loss(session, run.id)

    return RunOut(
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
    """Create a new training run record."""
    model = session.get(Model, payload.model_id)
    if not model:
        raise ApiError(404, "not_found", "Selected model was not found.")

    if payload.dataset_id:
        ds = session.get(Dataset, payload.dataset_id)
        if not ds:
            raise ApiError(404, "not_found", "Selected dataset was not found.")

    run = Run(
        id=uuid.uuid4(),
        name=payload.name.strip(),
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        device=payload.device,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        status="pending",
        hyperparams=payload.hyperparams,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return _run_to_out(run, session)


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
