import uuid

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Bucket, Dataset, Document
from forge.schemas import DatasetCreate, DatasetOut, Page

router = APIRouter(prefix="/datasets", tags=["datasets"])

CHARS_PER_TOKEN = 4


def _to_out(dataset: Dataset) -> DatasetOut:
    return DatasetOut(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        bucket_id=dataset.bucket_id,
        status=dataset.status,
        doc_count=dataset.doc_count,
        char_count=dataset.char_count,
        token_count=dataset.token_count,
        config=dataset.config,
        artifact_uri=dataset.artifact_uri,
        created_at=dataset.created_at,
    )


def _load_one(session: Session, dataset_id: uuid.UUID) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ApiError(404, "not_found", "No dataset with that id.")
    return dataset


def _bucket_stats(session: Session, bucket_id: uuid.UUID) -> tuple[int, int, int]:
    row = session.execute(
        select(
            func.count().filter(Document.status == "parsed").label("parsed_documents"),
            func.coalesce(func.sum(Document.char_count), 0).label("chars"),
        ).where(Document.bucket_id == bucket_id)
    ).one_or_none()
    if row is None:
        return (0, 0, 0)
    parsed_docs = row[0] or 0
    chars = row[1] or 0
    return (parsed_docs, chars, chars // CHARS_PER_TOKEN)


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

    total = session.execute(
        select(func.count()).select_from(query.subquery())
    ).scalar_one()

    datasets = session.execute(
        query.order_by(Dataset.name).limit(limit).offset(offset)
    ).scalars().all()

    return Page[DatasetOut](items=[_to_out(d) for d in datasets], total=total)


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, session: Session = Depends(get_session)) -> DatasetOut:
    existing = session.execute(
        select(Dataset).where(Dataset.name == payload.name)
    ).scalar_one_or_none()
    if existing is not None:
        raise ApiError(409, "name_taken", f"A dataset named {payload.name!r} already exists.")

    doc_count, char_count, token_count = 0, 0, 0
    status = "draft"

    if payload.bucket_id is not None:
        bucket = session.get(Bucket, payload.bucket_id)
        if bucket is None:
            raise ApiError(404, "not_found", "No bucket with that id.")
        doc_count, char_count, token_count = _bucket_stats(session, payload.bucket_id)
        if doc_count > 0:
            status = "ready"

    dataset = Dataset(
        name=payload.name,
        description=payload.description,
        bucket_id=payload.bucket_id,
        status=status,
        doc_count=doc_count,
        char_count=char_count,
        token_count=token_count,
        config=payload.config,
    )
    session.add(dataset)
    session.commit()
    return _to_out(dataset)


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> DatasetOut:
    dataset = _load_one(session, dataset_id)
    return _to_out(dataset)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    dataset = _load_one(session, dataset_id)
    session.delete(dataset)
    session.commit()
    return Response(status_code=204)

@router.get("/{dataset_id}/download")
def download_dataset(
dataset_id: uuid.UUID,
session: Session = Depends(get_session),
):
    _load_one(session, dataset_id)
    return Response(status_code=200)
