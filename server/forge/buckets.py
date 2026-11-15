import hashlib
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from forge import jobs
from forge.dataset_import import inspect_dataset, kind_problem, validate_import_params
from forge.db import get_session
from forge.errors import ApiError
from forge.models import Bucket, Document, Job
from forge.schemas import (
    BucketActiveJob,
    BucketCreate,
    BucketDeletionSummary,
    BucketOut,
    BucketStats,
    BucketUpdate,
    DatasetImportCreate,
    HfInspectRequest,
    HfInspectResponse,
    JobOut,
    JobPayload,
    Page,
)
from forge.storage import (
    FileAccounting,
    InvalidStoragePathError,
    account_storage_files,
    delete_storage_files,
    require_bucket_lock,
)

router = APIRouter(prefix="/buckets", tags=["buckets"])

CHARS_PER_TOKEN = 4
EXTERNAL_URI_CHUNK = 500


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
        source_dataset=bucket.source_dataset,
        resolved_revision=bucket.resolved_revision,
        dataset_config=bucket.dataset_config,
        dataset_split=bucket.dataset_split,
        text_column=bucket.text_column,
        import_limit=bucket.import_limit,
        kind=bucket.kind,
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
    row = session.execute(
        select(
            func.count(),
            func.count().filter(Document.status == "parsed"),
            func.coalesce(func.sum(Document.char_count), 0),
        ).where(Document.bucket_id == bucket_id)
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
    with require_bucket_lock(bucket_id):
        session.rollback()
        bucket = _load_one(session, bucket_id)

        if payload.name is not None and payload.name != bucket.name:
            clash = session.execute(
                select(Bucket).where(Bucket.name == payload.name)
            ).scalar_one_or_none()
            if clash is not None:
                raise ApiError(
                    409,
                    "name_taken",
                    f"A bucket named {payload.name!r} already exists.",
                )
            bucket.name = payload.name

        if payload.description is not None:
            bucket.description = payload.description

        session.commit()
        documents, parsed, chars = _stats_for(session, bucket_id)
        return _to_out(bucket, documents, parsed, chars)


@dataclass(frozen=True)
class _DeletionPlan:
    summary: BucketDeletionSummary
    accounting: FileAccounting


def _matches_bucket_id(raw: object, bucket_id: uuid.UUID) -> bool:
    if raw is None:
        return False
    try:
        return uuid.UUID(str(raw)) == bucket_id
    except ValueError:
        return False


def _active_jobs_for(session: Session, bucket_id: uuid.UUID) -> list[BucketActiveJob]:
    active: list[BucketActiveJob] = []
    rows = session.scalars(
        select(Job)
        .where(
            Job.job_type.in_(("dataset_import", "snapshot_create")),
            Job.status.in_(("pending", "running")),
        )
        .order_by(Job.created_at)
    )
    for job in rows:
        payload = job.payload or {}
        owner_id = (
            payload.get("target_id")
            if job.job_type == "dataset_import"
            else payload.get("bucket_id")
        )
        if _matches_bucket_id(owner_id, bucket_id):
            active.append(BucketActiveJob(id=job.id, job_type=job.job_type, status=job.status))
    return active


def _preserved_external_uris(
    session: Session, bucket_id: uuid.UUID, candidate_uris: list[str]
) -> set[str]:
    preserved: set[str] = set()
    for start in range(0, len(candidate_uris), EXTERNAL_URI_CHUNK):
        chunk = candidate_uris[start : start + EXTERNAL_URI_CHUNK]
        for uri in session.scalars(
            select(Document.raw_uri)
            .where(Document.bucket_id != bucket_id, Document.raw_uri.in_(chunk))
            .distinct()
        ):
            if uri:
                preserved.add(uri)
        for uri in session.scalars(
            select(Document.text_uri)
            .where(Document.bucket_id != bucket_id, Document.text_uri.in_(chunk))
            .distinct()
        ):
            if uri:
                preserved.add(uri)
    return preserved


def _build_deletion_plan(session: Session, bucket: Bucket) -> _DeletionPlan:
    doc_count = 0
    raw_uris: set[str] = set()
    text_uris: set[str] = set()
    doc_digest = hashlib.sha256()

    rows = session.execute(
        select(Document.id, Document.raw_uri, Document.text_uri)
        .where(Document.bucket_id == bucket.id)
        .order_by(Document.id)
    ).yield_per(1000)

    for doc_id, raw_uri, text_uri in rows:
        doc_count += 1
        if raw_uri:
            raw_uris.add(raw_uri)
        if text_uri:
            text_uris.add(text_uri)
        doc_digest.update(f"{doc_id}|{raw_uri}|{text_uri or ''}\n".encode())

    preserved_uris = _preserved_external_uris(session, bucket.id, sorted(raw_uris | text_uris))
    try:
        accounting = account_storage_files(
            sorted(raw_uris),
            sorted(text_uris),
            preserved_uris=preserved_uris,
        )
    except InvalidStoragePathError as problem:
        raise ApiError(409, "invalid_storage_path", str(problem)) from problem

    active_jobs = _active_jobs_for(session, bucket.id)

    scope_hash = hashlib.sha256()
    scope_hash.update(f"{bucket.id}\n{bucket.name}\n{doc_count}\n".encode())
    scope_hash.update(doc_digest.digest())
    for fp in accounting.file_fingerprints:
        scope_hash.update(f"{fp}\n".encode())

    summary = BucketDeletionSummary(
        bucket_id=bucket.id,
        bucket_name=bucket.name,
        document_count=doc_count,
        raw_file_count=accounting.raw_file_count,
        text_file_count=accounting.text_file_count,
        physical_file_count=accounting.physical_file_count,
        raw_bytes=accounting.raw_bytes,
        text_bytes=accounting.text_bytes,
        total_bytes=accounting.total_bytes,
        shared_file_count=accounting.shared_file_count,
        missing_file_count=accounting.missing_file_count,
        active_jobs=active_jobs,
        scope_token=scope_hash.hexdigest(),
    )
    return _DeletionPlan(summary=summary, accounting=accounting)


@router.get("/{bucket_id}/deletion-summary", response_model=BucketDeletionSummary)
def get_bucket_deletion_summary(
    bucket_id: uuid.UUID, session: Session = Depends(get_session)
) -> BucketDeletionSummary:
    """Compute a read-only summary of documents and exclusively owned files to be removed."""
    bucket = _load_one(session, bucket_id)
    return _build_deletion_plan(session, bucket).summary


@router.delete("/{bucket_id}", status_code=204)
def delete_bucket(
    bucket_id: uuid.UUID,
    confirm: bool = Query(default=False),
    scope_token: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> Response:
    """Delete a collection, its document records, and exclusively owned files after confirmation."""
    if not confirm:
        _load_one(session, bucket_id)
        raise ApiError(
            400,
            "confirmation_required",
            "Explicit confirmation is required to delete a collection.",
        )

    with require_bucket_lock(
        bucket_id,
        message=(
            "This collection is currently busy with an active operation. "
            "Wait for it to finish and try again."
        ),
    ):
        # Reset any earlier transaction snapshot before reading under the collection lock.
        session.rollback()
        bucket = _load_one(session, bucket_id)

        if _active_jobs_for(session, bucket_id):
            raise ApiError(
                409,
                "bucket_busy",
                "This collection has an active import or snapshot job. "
                "Wait for it to finish or cancel it before deleting.",
            )

        plan = _build_deletion_plan(session, bucket)
        if scope_token is not None and scope_token != plan.summary.scope_token:
            raise ApiError(
                409,
                "scope_changed",
                "This collection changed after the deletion summary was loaded. "
                "Review the updated summary and confirm again.",
            )

        try:
            result = delete_storage_files(
                plan.accounting.owned_raw_paths + plan.accounting.owned_text_paths
            )
        except InvalidStoragePathError as problem:
            raise ApiError(409, "invalid_storage_path", str(problem)) from problem

        if result.failed:
            first = result.failed[0]
            raise ApiError(
                500,
                "cleanup_failed",
                f"Incomplete cleanup: deleted {result.deleted_count} file(s), "
                f"{result.missing_count} already missing, {len(result.failed)} failed "
                f"({first.uri}: {first.error}). The collection record was kept so you can retry.",
            )

        session.execute(delete(Document).where(Document.bucket_id == bucket.id))
        session.delete(bucket)
        session.commit()
    return Response(status_code=204)


@router.post("/huggingface/inspect", response_model=HfInspectResponse)
def inspect_hf_dataset(payload: HfInspectRequest) -> HfInspectResponse:
    """Inspect a public Hugging Face dataset, returning configurations, splits, and columns."""
    return inspect_dataset(payload.dataset)


@router.post("/{bucket_id}/import", response_model=JobOut, status_code=201)
def start_dataset_import(
    bucket_id: uuid.UUID,
    payload: DatasetImportCreate,
    session: Session = Depends(get_session),
) -> JobOut:
    """Queue a background import job to stream Hugging Face records into this collection."""
    with require_bucket_lock(bucket_id):
        session.rollback()
        _load_one(session, bucket_id)

    use = validate_import_params(payload.dataset, payload)

    with require_bucket_lock(bucket_id):
        session.rollback()
        bucket = _load_one(session, bucket_id)
        problem = kind_problem(session, bucket, use)
        if problem:
            raise ApiError(409, "wrong_kind", problem)
        job = Job(
            job_type="dataset_import",
            payload={
                **JobPayload(target_id=bucket.id, target_name=bucket.name).model_dump(mode="json"),
                **payload.model_dump(mode="json"),
                "use": use,
            },
        )
        session.add(job)
        session.commit()
    jobs.start_runner()
    return JobOut.model_validate(job)
