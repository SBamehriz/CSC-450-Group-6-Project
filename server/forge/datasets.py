import uuid

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Dataset, Run
from forge.schemas import DatasetCreate, DatasetOut, Page, SnapshotCheck
from forge.snapshots import archive_path, file_hash, read_snapshot, source_rows, write_snapshot

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _load_one(session: Session, dataset_id: uuid.UUID) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ApiError(404, "not_found", "No dataset with that id.")
    return dataset


def _to_out(dataset: Dataset) -> DatasetOut:
    result = DatasetOut.model_validate(dataset)
    try:
        path = archive_path(dataset)
        result.download_ready = (
            dataset.status == "ready"
            and path.is_file()
            and file_hash(path) == dataset.config.get("sha256")
        )
    except (ApiError, OSError):
        result.download_ready = False
    if not result.download_ready:
        result.readiness_note = "The saved ZIP is missing or has changed. Build a new snapshot."
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
    bucket, rows = source_rows(session, payload.bucket_id)
    dataset = Dataset(
        id=uuid.uuid4(), name=payload.name, description=payload.description, bucket_id=bucket.id
    )
    write_snapshot(dataset, bucket, rows, payload.validation_fraction, payload.seed)
    session.add(dataset)
    try:
        session.commit()
    except IntegrityError as problem:
        session.rollback()
        archive_path(dataset).unlink(missing_ok=True)
        raise ApiError(
            409, "name_taken", "That dataset name was just used. Choose another."
        ) from problem
    return _to_out(dataset)


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
        message="The ZIP and both text files passed their checks.",
        train_documents=len(splits["train"]),
        validation_documents=len(splits["validation"]),
    )


@router.get("/{dataset_id}/download", response_class=FileResponse)
def download_dataset(
    dataset_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    dataset = _load_one(session, dataset_id)
    read_snapshot(dataset)
    return FileResponse(
        archive_path(dataset), media_type="application/zip", filename=f"snapshot-{dataset.id}.zip"
    )


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    from forge.prepared_api import _prepare_lock

    if not _prepare_lock.acquire(blocking=False):
        raise ApiError(409, "busy", "Wait for preparation to finish before deleting a snapshot.")
    try:
        return _delete_dataset(dataset_id, session)
    finally:
        _prepare_lock.release()


def _delete_dataset(dataset_id: uuid.UUID, session: Session) -> Response:
    from forge.preparation import prepared_path

    dataset = _load_one(session, dataset_id)
    if session.scalar(select(Run.id).where(Run.dataset_id == dataset_id).limit(1)):
        raise ApiError(409, "in_use", "A run uses this dataset. Keep it with the run.")
    try:
        path = archive_path(dataset)
    except ApiError:
        path = None
    session.delete(dataset)
    session.commit()
    prepared_path(dataset).unlink(missing_ok=True)
    if path is not None:
        path.unlink(missing_ok=True)
    return Response(status_code=204)
