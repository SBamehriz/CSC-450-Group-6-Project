import uuid

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Bucket, Document
from forge.schemas import BucketCreate, BucketOut, BucketStats, BucketUpdate, Page

router = APIRouter(prefix="/buckets", tags=["buckets"])

CHARS_PER_TOKEN = 4


def _stats_subquery():
    return (
        select(
            Document.bucket_id.label("bucket_id"),
            func.count().label("documents"),
            func.count().filter(Document.status == "parsed").label("parsed_documents"),
            func.coalesce(func.sum(Document.char_count), 0).label("chars"),
        )
        .group_by(Document.bucket_id)
        .subquery()
    )


def _to_out(bucket: Bucket, documents: int, parsed: int, chars: int) -> BucketOut:
    return BucketOut(
        id=bucket.id,
        name=bucket.name,
        description=bucket.description,
        created_at=bucket.created_at,
        stats=BucketStats(
            documents=documents,
            parsed_documents=parsed,
            chars=chars,
            est_tokens=chars // CHARS_PER_TOKEN,
        ),
    )


def _load_one(session: Session, bucket_id: uuid.UUID) -> Bucket:
    bucket = session.get(Bucket, bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "No bucket with that id.")
    return bucket


def _stats_for(session: Session, bucket_id: uuid.UUID) -> tuple[int, int, int]:
    stats = _stats_subquery()
    row = session.execute(
        select(stats.c.documents, stats.c.parsed_documents, stats.c.chars).where(
            stats.c.bucket_id == bucket_id
        )
    ).one_or_none()
    return (0, 0, 0) if row is None else (row[0], row[1], row[2])


@router.get("", response_model=Page[BucketOut])
def list_buckets(
    session: Session = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[BucketOut]:
    total = session.execute(select(func.count()).select_from(Bucket)).scalar_one()
    stats = _stats_subquery()
    rows = session.execute(
        select(Bucket, stats.c.documents, stats.c.parsed_documents, stats.c.chars)
        .outerjoin(stats, stats.c.bucket_id == Bucket.id)
        .order_by(Bucket.name)
        .limit(limit)
        .offset(offset)
    ).all()
    # nulls for empty buckets
    items = [_to_out(row[0], row[1] or 0, row[2] or 0, row[3] or 0) for row in rows]
    return Page[BucketOut](items=items, total=total)


@router.post("", response_model=BucketOut, status_code=201)
def create_bucket(payload: BucketCreate, session: Session = Depends(get_session)) -> BucketOut:
    existing = session.execute(
        select(Bucket).where(Bucket.name == payload.name)
    ).scalar_one_or_none()
    if existing is not None:
        raise ApiError(409, "name_taken", f"A bucket named {payload.name!r} already exists.")

    bucket = Bucket(name=payload.name, description=payload.description)
    session.add(bucket)
    session.commit()
    return _to_out(bucket, 0, 0, 0)


@router.get("/{bucket_id}", response_model=BucketOut)
def get_bucket(bucket_id: uuid.UUID, session: Session = Depends(get_session)) -> BucketOut:
    bucket = _load_one(session, bucket_id)
    return _to_out(bucket, *_stats_for(session, bucket_id))


@router.patch("/{bucket_id}", response_model=BucketOut)
def update_bucket(
    bucket_id: uuid.UUID, payload: BucketUpdate, session: Session = Depends(get_session)
) -> BucketOut:
    bucket = _load_one(session, bucket_id)

    if payload.name is not None and payload.name != bucket.name:
        clash = session.execute(
            select(Bucket).where(Bucket.name == payload.name)
        ).scalar_one_or_none()
        if clash is not None:
            raise ApiError(409, "name_taken", f"A bucket named {payload.name!r} already exists.")
        bucket.name = payload.name

    if payload.description is not None:
        bucket.description = payload.description

    session.commit()
    documents, parsed, chars = _stats_for(session, bucket_id)
    return _to_out(bucket, documents, parsed, chars)


@router.delete("/{bucket_id}", status_code=204)
def delete_bucket(bucket_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    bucket = _load_one(session, bucket_id)
    documents, _, _ = _stats_for(session, bucket_id)
    if documents:
        raise ApiError(
            409,
            "bucket_not_empty",
            f"This bucket still has {documents} document(s). Move or delete them first.",
        )
    session.delete(bucket)
    session.commit()
    return Response(status_code=204)
