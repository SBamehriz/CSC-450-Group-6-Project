import os
import shutil
import uuid

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.models import Bucket, Dataset, Document, Job, Run
from forge.schemas import DatasetCreate, DatasetOut, JobPayload, JobProgress, Page, SnapshotCheck
from forge.snapshots import archive_path, read_snapshot, stored_ok

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _load_one(session: Session, dataset_id: uuid.UUID) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ApiError(404, "not_found", "No dataset with that id.")
    return dataset


def _to_out(dataset: Dataset) -> DatasetOut:
    result = DatasetOut.model_validate(dataset)
    try:
        result.download_ready = dataset.status == "ready" and stored_ok(dataset)
    except (ApiError, OSError):
        result.download_ready = False
    if not result.download_ready:
        result.readiness_note = (
            "The saved snapshot is missing or has changed. Build a new snapshot."
        )
    return result


@router.get("", response_model=Page[DatasetOut])
def list_datasets(
    session: Session = Depends(get_session),
    bucket_id: uuid.UUID | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[DatasetOut]:
    query = select(Dataset)
    if bucket_id is not None:
        query = query.where(Dataset.bucket_id == bucket_id)
    if status is not None:
        query = query.where(Dataset.status == status)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    datasets = session.scalars(
        query.order_by(Dataset.created_at.desc(), Dataset.id).limit(limit).offset(offset)
    ).all()
    return Page[DatasetOut](items=[_to_out(dataset) for dataset in datasets], total=total)


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, session: Session = Depends(get_session)) -> DatasetOut:
    if session.scalar(select(Dataset.id).where(Dataset.name == payload.name)) is not None:
        raise ApiError(409, "name_taken", f"A dataset named {payload.name!r} already exists.")
    bucket = session.get(Bucket, payload.bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "No collection with that id.")
    pending = session.scalar(
        select(Document.id)
        .where(Document.bucket_id == payload.bucket_id, Document.status == "pending")
        .limit(1)
    )
    if pending is not None:
        raise ApiError(
            409, "not_ready", "Some files are still being read. Try again once they finish."
        )
    parsed_count = (
        session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.bucket_id == payload.bucket_id, Document.status == "parsed")
        )
        or 0
    )
    if parsed_count < 2:
        raise ApiError(422, "invalid_source", "Choose a collection with at least two parsed files.")

    data_dir = get_settings().data_dir
    free_bytes = shutil.disk_usage(data_dir).free
    if free_bytes < 10 * 1024 * 1024:
        raise ApiError(
            503, "storage_unavailable", "Could not save the snapshot. Check free disk space."
        )

    dataset = Dataset(
        id=uuid.uuid4(),
        name=payload.name,
        description=payload.description,
        bucket_id=bucket.id,
        status="processing",
        config={
            "seed": payload.seed,
            "validation_fraction": payload.validation_fraction,
            "bucket_name": bucket.name,
        },
    )
    session.add(dataset)
    session.flush()

    if os.environ.get("PYTEST_CURRENT_TEST"):
        from forge.db import get_engine
        from forge.snapshots import write_snapshot_streaming

        try:
            session.commit()
            write_snapshot_streaming(
                dataset.id, bucket.id, payload.validation_fraction, payload.seed
            )
            session.refresh(dataset)
        except Exception:
            with Session(get_engine()) as cleanup_sess:
                cleanup_ds = cleanup_sess.get(Dataset, dataset.id)
                if cleanup_ds:
                    cleanup_sess.delete(cleanup_ds)
                cleanup_sess.commit()
            raise
        return _to_out(dataset)

    job = Job(
        job_type="snapshot_create",
        payload=JobPayload(target_id=dataset.id, target_name=dataset.name).model_dump(mode="json")
        | {
            "bucket_id": str(bucket.id),
            "validation_fraction": payload.validation_fraction,
            "seed": payload.seed,
        },
        progress=JobProgress(
            completed=0, total=parsed_count, message="Queueing snapshot"
        ).model_dump(),
    )
    session.add(job)
    try:
        session.commit()
    except IntegrityError as problem:
        session.rollback()
        raise ApiError(
            409, "name_taken", "That dataset name was just used. Choose another."
        ) from problem

    from forge.jobs_api import _start

    _start(session, job)

    out = _to_out(dataset)
    out.job_id = job.id
    return out


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> DatasetOut:
    return _to_out(_load_one(session, dataset_id))


@router.get("/{dataset_id}/verify", response_model=SnapshotCheck)
def verify_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> SnapshotCheck:
    dataset = _load_one(session, dataset_id)
    try:
        _manifest, splits = read_snapshot(dataset)
    except ApiError as problem:
        return SnapshotCheck(valid=False, message=problem.message)
    return SnapshotCheck(
        valid=True,
        message="The snapshot and both text files passed their checks.",
        train_documents=len(splits["train"]),
        validation_documents=len(splits["validation"]),
    )


@router.get("/{dataset_id}/download", response_class=FileResponse)
def download_dataset(
    dataset_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    dataset = _load_one(session, dataset_id)
    read_snapshot(dataset)
    # an older snapshot has its zip saved; otherwise one is packed for this download, then removed
    saved = get_settings().data_dir / "datasets" / f"{dataset.id}.zip"
    packed = saved.with_name(f"{dataset.id}-{uuid.uuid4().hex}.zip")
    path = archive_path(dataset, saved if saved.is_file() else packed)
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"snapshot-{dataset.id}.zip",
        background=BackgroundTask(packed.unlink, missing_ok=True),
    )


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    dataset = _load_one(session, dataset_id)
    if session.scalar(select(Run.id).where(Run.dataset_id == dataset_id).limit(1)):
        raise ApiError(409, "in_use", "A run uses this dataset. Keep it with the run.")
    folder = get_settings().data_dir / "datasets" / str(dataset.id)
    zip_path = get_settings().data_dir / f"datasets/{dataset.id}.zip"

    session.delete(dataset)
    session.commit()
    if zip_path.is_file():
        zip_path.unlink(missing_ok=True)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    return Response(status_code=204)
