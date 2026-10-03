from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.documents import _to_out
from forge.errors import ApiError
from forge.models import Bucket, Dataset, Document
from forge.schemas import OverviewStats, Page, RecentUpload

router = APIRouter(prefix="/overview", tags=["overview"])


@router.get("/stats", response_model=OverviewStats)
def get_stats(session: Session = Depends(get_session)) -> OverviewStats:
    try:
        row = session.execute(
            select(
                func.count().label("documents"),
                func.count().filter(Document.status == "parsed").label("parsed"),
                func.count().filter(Document.status == "failed").label("failed"),
                func.count().filter(Document.status == "rejected").label("rejected"),
                func.count().filter(Document.status == "pending").label("pending"),
                func.count()
                .filter(
                    (Document.status == "parsed")
                    & (func.json_array_length(Document.quality, "$.flags") > 0)
                )
                .label("flagged"),
                func.coalesce(
                    func.sum(Document.char_count).filter(Document.status == "parsed"), 0
                ).label("chars"),
                func.coalesce(
                    func.sum(Document.word_count).filter(Document.status == "parsed"), 0
                ).label("words"),
            ).select_from(Document)
        ).one()
        buckets = session.scalar(select(func.count()).select_from(Bucket)) or 0
        snapshots, ready = session.execute(
            select(
                func.count(),
                func.count().filter(Dataset.status == "ready"),
            ).select_from(Dataset)
        ).one()
    except SQLAlchemyError as problem:
        raise ApiError(503, "unavailable", "Could not load library statistics.") from problem
    return OverviewStats(
        buckets=buckets,
        documents=row.documents,
        parsed_documents=row.parsed,
        failed_documents=row.failed,
        rejected_documents=row.rejected,
        pending_documents=row.pending,
        flagged_documents=row.flagged,
        chars=row.chars,
        words=row.words,
        est_tokens=row.chars // 4,
        snapshots=snapshots,
        ready_snapshots=ready,
    )


@router.get("/recent-uploads", response_model=Page[RecentUpload])
def recent_uploads(
    session: Session = Depends(get_session),
    limit: int = Query(default=8, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
) -> Page[RecentUpload]:
    try:
        total = session.scalar(select(func.count()).select_from(Document)) or 0
        rows = session.execute(
            select(Document, Bucket.name)
            .join(Bucket, Document.bucket_id == Bucket.id)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return Page[RecentUpload](
            items=[
                RecentUpload(**_to_out(document).model_dump(), bucket_name=name)
                for document, name in rows
            ],
            total=total,
        )
    except SQLAlchemyError as problem:
        raise ApiError(503, "unavailable", "Could not load recent uploads.") from problem
