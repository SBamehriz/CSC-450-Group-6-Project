import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import ColumnElement, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forge import jobs
from forge.config import get_settings
from forge.datasets import _load_one
from forge.db import get_session
from forge.errors import ApiError
from forge.models import DatasetBuild, Job, Tokenizer
from forge.preparation import tools_available
from forge.schemas import (
    DatasetBuildCreate,
    DatasetBuildOut,
    JobOut,
    JobPayload,
    JobProgress,
    Page,
    TokenizerCreate,
    TokenizerOut,
    TokenPreview,
    TokenPreviewCreate,
)
from forge.snapshots import read_snapshot

router = APIRouter(tags=["jobs"])


def _tools() -> None:
    if not tools_available():
        raise ApiError(
            503,
            "tools_missing",
            "Tokenizer tools are not installed. Add the prepare extra from the README.",
        )


def _enqueue(session: Session, target: Tokenizer | DatasetBuild, kind: str) -> None:
    session.add(target)
    try:
        session.flush()
        job = Job(
            job_type=kind,
            payload=JobPayload(target_id=target.id, target_name=target.name).model_dump(
                mode="json"
            ),
        )
        session.add(job)
        session.commit()
    except IntegrityError as problem:
        session.rollback()
        raise ApiError(
            409, "name_taken", "That name is already in use. Choose another name."
        ) from problem
    _start(session, job)


def _start(session: Session, job: Job) -> None:
    try:
        jobs.start_runner()
    except OSError as problem:
        # Another runner may have picked this up after we committed it.
        session.rollback()
        session.execute(text("BEGIN IMMEDIATE"))
        session.refresh(job)
        if job.status != "pending":
            return
        job.status = "failed"
        target = jobs.target_for(session, job)
        target.status = "failed"
        message = f"The background process could not start: {problem}"
        target.error = job.error = message
        job.completed_at = datetime.now(UTC)
        session.commit()
        raise ApiError(503, "runner_unavailable", message) from problem


def _snapshot_ready(session: Session, dataset_id: uuid.UUID):
    dataset = _load_one(session, dataset_id)
    read_snapshot(dataset)
    return dataset


def _busy(session: Session, target_id: uuid.UUID) -> bool:
    return any(
        JobPayload.model_validate(job.payload).target_id == target_id
        for job in session.scalars(
            select(Job).where(
                Job.job_type.in_(jobs.RUNNER_TYPES), Job.status.in_(("pending", "running"))
            )
        )
    )


def _remove(session: Session, target: Tokenizer | DatasetBuild) -> Response:
    if _busy(session, target.id):
        raise ApiError(409, "busy", "Cancel its job and wait for it to stop first.")
    for job in session.scalars(select(Job).where(Job.job_type.in_(jobs.RUNNER_TYPES))):
        if JobPayload.model_validate(job.payload).target_id == target.id:
            session.delete(job)
    directory = jobs.artifact_directory(target)
    session.delete(target)
    session.commit()
    shutil.rmtree(directory, ignore_errors=True)
    return Response(status_code=204)


@router.post("/tokenizers", response_model=TokenizerOut, status_code=201)
def create_tokenizer(
    payload: TokenizerCreate, session: Session = Depends(get_session)
) -> TokenizerOut:
    _tools()
    dataset = _snapshot_ready(session, payload.dataset_id)
    target = Tokenizer(
        name=payload.name,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        requested_vocab_size=payload.vocab_size,
    )
    _enqueue(session, target, "tokenizer_train")
    return TokenizerOut.model_validate(target)


@router.get("/tokenizers", response_model=Page[TokenizerOut])
def list_tokenizers(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[TokenizerOut]:
    jobs.recover_jobs()
    rows = session.scalars(
        select(Tokenizer)
        .order_by(Tokenizer.created_at.desc(), Tokenizer.id)
        .limit(limit)
        .offset(offset)
    )
    total = session.scalar(select(func.count()).select_from(Tokenizer)) or 0
    return Page(items=[TokenizerOut.model_validate(row) for row in rows], total=total)


@router.get("/tokenizers/{target_id}", response_model=TokenizerOut)
def get_tokenizer(target_id: uuid.UUID, session: Session = Depends(get_session)) -> TokenizerOut:
    jobs.recover_jobs()
    return TokenizerOut.model_validate(_tokenizer(session, target_id))


def _tokenizer(session: Session, target_id: uuid.UUID) -> Tokenizer:
    target = session.get(Tokenizer, target_id)
    if target is None:
        raise ApiError(404, "not_found", "No vocabulary with that id.")
    return target


@router.post("/tokenizers/{target_id}/preview", response_model=TokenPreview)
def preview_tokenizer(
    target_id: uuid.UUID, payload: TokenPreviewCreate, session: Session = Depends(get_session)
) -> TokenPreview:
    _tools()
    from tokenizers import Tokenizer as BpeTokenizer

    target = _tokenizer(session, target_id)
    tokenizer = BpeTokenizer.from_file(str(_artifact(target.artifact_uri, target.status)))
    encoded = tokenizer.encode(payload.text)
    return TokenPreview(token_count=len(encoded.ids), ids=encoded.ids, pieces=encoded.tokens)


@router.delete("/tokenizers/{target_id}", status_code=204)
def delete_tokenizer(target_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    jobs.recover_jobs()
    target = _tokenizer(session, target_id)
    if session.scalar(select(DatasetBuild.id).where(DatasetBuild.tokenizer_id == target_id)):
        raise ApiError(
            409, "in_use", "A training set uses this vocabulary. Delete the training set first."
        )
    return _remove(session, target)


@router.post("/dataset-builds", response_model=DatasetBuildOut, status_code=201)
def create_build(
    payload: DatasetBuildCreate, session: Session = Depends(get_session)
) -> DatasetBuildOut:
    _tools()
    tokenizer = session.get(Tokenizer, payload.tokenizer_id)
    if tokenizer is None or tokenizer.status != "ready":
        raise ApiError(
            409, "tokenizer_not_ready", "Choose a ready vocabulary before making a training set."
        )
    dataset = _snapshot_ready(session, payload.dataset_id)
    target = DatasetBuild(
        name=payload.name,
        tokenizer_id=tokenizer.id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
    )
    _enqueue(session, target, "dataset_build")
    return DatasetBuildOut.model_validate(target)


@router.get("/dataset-builds", response_model=Page[DatasetBuildOut])
def list_builds(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[DatasetBuildOut]:
    jobs.recover_jobs()
    rows = session.scalars(
        select(DatasetBuild)
        .order_by(DatasetBuild.created_at.desc(), DatasetBuild.id)
        .limit(limit)
        .offset(offset)
    )
    total = session.scalar(select(func.count()).select_from(DatasetBuild)) or 0
    return Page(items=[DatasetBuildOut.model_validate(row) for row in rows], total=total)


@router.get("/dataset-builds/{target_id}", response_model=DatasetBuildOut)
def get_build(target_id: uuid.UUID, session: Session = Depends(get_session)) -> DatasetBuildOut:
    jobs.recover_jobs()
    return DatasetBuildOut.model_validate(_build(session, target_id))


def _build(session: Session, target_id: uuid.UUID) -> DatasetBuild:
    target = session.get(DatasetBuild, target_id)
    if target is None:
        raise ApiError(404, "not_found", "No training set with that id.")
    return target


@router.delete("/dataset-builds/{target_id}", status_code=204)
def delete_build(target_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    jobs.recover_jobs()
    return _remove(session, _build(session, target_id))


def _artifact(path: str | None, status: str) -> Path:
    if status != "ready" or not path:
        raise ApiError(409, "not_ready", "This is not ready yet.")
    candidate = get_settings().data_dir / path
    if not candidate.is_file():
        raise ApiError(404, "not_found", "The saved file is missing. Restore it from your backup.")
    return candidate


@router.get("/tokenizers/{target_id}/artifact", response_class=FileResponse)
def tokenizer_artifact(
    target_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    target = _tokenizer(session, target_id)
    return FileResponse(
        _artifact(target.artifact_uri, target.status),
        media_type="application/json",
        filename="tokenizer.json",
    )


@router.get("/dataset-builds/{target_id}/manifest", response_class=FileResponse)
def build_manifest(target_id: uuid.UUID, session: Session = Depends(get_session)) -> FileResponse:
    target = _build(session, target_id)
    return FileResponse(
        _artifact(target.manifest_uri, target.status),
        media_type="application/json",
        filename="manifest.json",
    )


@router.get("/jobs", response_model=Page[JobOut])
def list_jobs(
    session: Session = Depends(get_session),
    status: str | None = None,
    job_type: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[JobOut]:
    jobs.recover_jobs()
    filters: list[ColumnElement[bool]] = [Job.job_type.in_(jobs.RUNNER_TYPES)]
    if status:
        filters.append(Job.status == status)
    if job_type:
        filters.append(Job.job_type == job_type)
    rows = session.scalars(
        select(Job)
        .where(*filters)
        .order_by(Job.created_at.desc(), Job.id)
        .limit(limit)
        .offset(offset)
    )
    total = session.scalar(select(func.count()).select_from(Job).where(*filters)) or 0
    return Page(items=[JobOut.model_validate(row) for row in rows], total=total)


def _job(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None or job.job_type not in jobs.RUNNER_TYPES:
        raise ApiError(404, "not_found", "No job with that id.")
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, session: Session = Depends(get_session)) -> JobOut:
    jobs.recover_jobs()
    return JobOut.model_validate(_job(session, job_id))


@router.post("/jobs/{job_id}/{action}", response_model=JobOut)
def change_job(
    job_id: uuid.UUID, action: Literal["cancel", "retry"], session: Session = Depends(get_session)
) -> JobOut:
    jobs.recover_jobs()
    session.execute(text("BEGIN IMMEDIATE"))
    job = _job(session, job_id)
    target = jobs.target_for(session, job)
    if action == "cancel":
        if job.status in ("pending", "running"):
            job.cancel_requested = True
            if job.status == "pending":
                job.status = target.status = "cancelled"
                job.completed_at = datetime.now(UTC)
    else:
        if job.status not in ("failed", "cancelled") or target.status == "ready":
            raise ApiError(409, "not_retryable", "Only a failed or canceled job can be retried.")
        job.status = target.status = "pending"
        job.error = target.error = None
        job.cancel_requested = False
        job.started_at = job.completed_at = None
        job.progress = JobProgress().model_dump()
    session.commit()
    if action == "retry":
        _start(session, job)
    return JobOut.model_validate(job)
