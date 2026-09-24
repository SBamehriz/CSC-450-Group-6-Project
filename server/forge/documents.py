import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.ingest import (
    IMAGE_EXTENSIONS,
    ParseError,
    clean,
    looks_mis_decoded,
    parse,
    split_extension,
)
from forge.models import Bucket, Document
from forge.schemas import (
    DocumentOut,
    DocumentText,
    MoveDocument,
    Page,
    RejectDocument,
    UploadOutcome,
    UploadResult,
)

router = APIRouter(tags=["documents"])

MAX_FILES_PER_REQUEST = 200
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TEXT_WINDOW = 200_000


def _storage_root() -> Path:
    return get_settings().data_dir / "files"


def _write(relative: str, data: bytes) -> str:
    path = _storage_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return relative


def _read_text(relative: str, offset: int, length: int) -> tuple[str, int]:
    path = _storage_root() / relative
    if not path.is_file():
        raise ApiError(404, "not_found", "The cleaned text for this document is missing.")
    text = path.read_text(encoding="utf-8")
    return text[offset : offset + length], len(text)


def _delete(relative: str | None) -> None:
    if not relative:
        return
    (_storage_root() / relative).unlink(missing_ok=True)


def _to_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        bucket_id=document.bucket_id,
        filename=document.filename,
        source_format=document.source_format,
        source_note=document.source_note,
        status=document.status,
        error=document.error,
        content_hash=document.content_hash,
        char_count=document.char_count,
        word_count=document.word_count,
        quality=document.quality,
        created_at=document.created_at,
    )


def _load(session: Session, document_id: uuid.UUID) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise ApiError(404, "not_found", "No document with that id.")
    return document


def _require_bucket(session: Session, bucket_id: uuid.UUID) -> Bucket:
    bucket = session.get(Bucket, bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "No bucket with that id.")
    return bucket


@router.post("/buckets/{bucket_id}/documents", response_model=UploadResult, status_code=201)
def upload_documents(
    bucket_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    source_note: str = Form(default=""),
    session: Session = Depends(get_session),
) -> UploadResult:
    _require_bucket(session, bucket_id)

    if len(files) > MAX_FILES_PER_REQUEST:
        raise ApiError(
            400,
            "too_many_files",
            f"{len(files)} files in one go is too many. The limit is "
            f"{MAX_FILES_PER_REQUEST}. Split it into smaller batches.",
        )

    outcomes: list[UploadOutcome] = []
    for upload in files:
        name = upload.filename or "unnamed"
        raw = upload.file.read()
        if len(raw) > MAX_FILE_BYTES:
            outcomes.append(
                UploadOutcome(
                    filename=name,
                    status="failed",
                    error=f"{len(raw) / 1e6:.1f} MB is over the "
                    f"{MAX_FILE_BYTES // (1024 * 1024)} MB limit for one file.",
                )
            )
            continue
        outcomes.extend(_ingest_one(session, bucket_id, name, raw, source_note))

    session.commit()
    parsed = sum(1 for outcome in outcomes if outcome.status == "parsed")
    return UploadResult(parsed=parsed, failed=len(outcomes) - parsed, outcomes=outcomes)


def _ingest_one(
    session: Session, bucket_id: uuid.UUID, name: str, raw: bytes, source_note: str
) -> list[UploadOutcome]:
    try:
        pieces = parse(name, raw)
    except ParseError as problem:
        document = Document(
            bucket_id=bucket_id,
            filename=name,
            source_format=_stored_format(name),
            source_note=source_note,
            raw_uri=_write(f"raw/{uuid.uuid4()}", raw),
            status="failed",
            error=str(problem),
            quality={},
        )
        session.add(document)
        session.flush()
        return [UploadOutcome(filename=name, status="failed", error=str(problem))]

    suspect = looks_mis_decoded(raw)
    raw_uri = _write(f"raw/{uuid.uuid4()}", raw)
    outcomes: list[UploadOutcome] = []

    for piece in pieces:
        cleaned = clean(piece.text, encoding_suspect=suspect)

        fmt = _stored_format(piece.filename)
        if fmt == "other":
            fmt = _stored_format(name)

        doc_quality = dict(cleaned.quality)
        if getattr(piece, "metadata", None):
            doc_quality["extraction"] = piece.metadata

        # so we can name the duplicate
        twin = session.execute(
            select(Document).where(
                Document.bucket_id == bucket_id,
                Document.content_hash == cleaned.content_hash,
            )
        ).scalar_one_or_none()
        if twin is not None:
            message = f"Duplicate of {twin.filename}. Identical text after cleaning."
            session.add(
                Document(
                    bucket_id=bucket_id,
                    filename=piece.filename,
                    source_format=fmt,
                    source_note=source_note,
                    raw_uri=raw_uri,
                    status="failed",
                    error=message,
                    quality=doc_quality,
                )
            )
            outcomes.append(UploadOutcome(filename=piece.filename, status="failed", error=message))
            continue

        document = Document(
            bucket_id=bucket_id,
            filename=piece.filename,
            source_format=fmt,
            source_note=source_note,
            raw_uri=raw_uri,
            status="parsed",
            content_hash=cleaned.content_hash,
            char_count=cleaned.char_count,
            word_count=cleaned.word_count,
            quality=doc_quality,
        )
        session.add(document)
        session.flush()  # need the id for the text path
        document.text_uri = _write(f"text/{document.id}.txt", cleaned.text.encode("utf-8"))
        outcomes.append(
            UploadOutcome(filename=piece.filename, status="parsed", document_id=document.id)
        )

    return outcomes


def _stored_format(filename: str) -> str:
    extension, _ = split_extension(filename)
    extension = "html" if extension == "htm" else extension  # treat htm like html
    if extension in ("txt", "md", "html", "pdf", "docx", "jsonl"):
        return extension
    if extension in IMAGE_EXTENSIONS:
        return "image"
    return "other"


@router.get("/documents", response_model=Page[DocumentOut])
def list_documents(
    session: Session = Depends(get_session),
    bucket_id: uuid.UUID | None = None,
    status: str | None = None,
    q: str | None = None,
    flagged: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[DocumentOut]:
    filters = []
    if bucket_id is not None:
        filters.append(Document.bucket_id == bucket_id)
    if status is not None:
        filters.append(Document.status == status)
    if q:
        filters.append(Document.filename.ilike(f"%{q}%"))
    if flagged:
        # flagged means parsed but questionable
        filters.append(Document.status == "parsed")
        filters.append(func.json_array_length(Document.quality, "$.flags") > 0)

    total = session.execute(select(func.count()).select_from(Document).where(*filters)).scalar_one()
    rows = session.execute(
        select(Document)
        .where(*filters)
        .order_by(Document.created_at.desc(), Document.filename)
        .limit(limit)
        .offset(offset)
    ).scalars()
    return Page[DocumentOut](items=[_to_out(row) for row in rows], total=total)


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> DocumentOut:
    return _to_out(_load(session, document_id))


@router.get("/documents/{document_id}/text", response_model=DocumentText)
def get_document_text(
    document_id: uuid.UUID,
    session: Session = Depends(get_session),
    offset: int = Query(default=0, ge=0),
    length: int = Query(default=4000, ge=1, le=MAX_TEXT_WINDOW),
) -> DocumentText:
    document = _load(session, document_id)
    if not document.text_uri:
        raise ApiError(
            409,
            "no_text",
            f"This document has no cleaned text. It {document.status} at upload.",
        )
    window, total = _read_text(document.text_uri, offset, length)
    return DocumentText(text=window, offset=offset, length=len(window), total=total)


@router.post("/documents/{document_id}/move", response_model=DocumentOut)
def move_document(
    document_id: uuid.UUID, payload: MoveDocument, session: Session = Depends(get_session)
) -> DocumentOut:
    document = _load(session, document_id)
    _require_bucket(session, payload.bucket_id)

    if payload.bucket_id != document.bucket_id and document.content_hash:
        # dedup is per bucket
        twin = session.execute(
            select(Document).where(
                Document.bucket_id == payload.bucket_id,
                Document.content_hash == document.content_hash,
            )
        ).scalar_one_or_none()
        if twin is not None:
            raise ApiError(
                409,
                "duplicate_in_target",
                f"That bucket already has this exact text, as {twin.filename}.",
            )

    document.bucket_id = payload.bucket_id
    session.commit()
    return _to_out(document)


@router.post("/documents/{document_id}/reject", response_model=DocumentOut)
def reject_document(
    document_id: uuid.UUID, payload: RejectDocument, session: Session = Depends(get_session)
) -> DocumentOut:
    document = _load(session, document_id)
    document.status = "rejected"
    document.error = payload.reason or "Rejected by hand."
    # free the hash for a better copy
    document.content_hash = None
    session.commit()
    return _to_out(document)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    document = _load(session, document_id)
    text_uri, raw_uri = document.text_uri, document.raw_uri
    session.delete(document)
    session.commit()

    _delete(text_uri)
    # jsonl documents share one raw file
    still_used = session.execute(
        select(func.count()).select_from(Document).where(Document.raw_uri == raw_uri)
    ).scalar_one()
    if not still_used:
        _delete(raw_uri)
    return Response(status_code=204)
