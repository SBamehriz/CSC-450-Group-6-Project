import hashlib
import json
import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.models import Bucket, Dataset, Document, Run
from forge.schemas import DatasetCreate, DatasetOut, Page

router = APIRouter(prefix="/datasets", tags=["datasets"])
MAX_DOCUMENTS = 1000
MAX_TEXT_BYTES = 20 * 1024 * 1024


def _out(row: Dataset) -> DatasetOut:
    return DatasetOut.model_validate(row)


def _archive(row: Dataset) -> Path:
    return get_settings().data_dir / "datasets" / f"{row.id}.zip"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(row: Dataset, documents: list[Document], fraction: float, seed: int) -> None:
    ranked = sorted(
        documents,
        key=lambda doc: (
            hashlib.sha256(f"{seed}:{doc.content_hash}:{doc.id}".encode()).digest(),
            str(doc.id),
        ),
    )
    validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * fraction)))
    validation_ids = {doc.id for doc in ranked[:validation_count]}
    counts = {"train": 0, "validation": 0}
    chars = 0
    bytes_read = 0
    sources: list[dict[str, str]] = []
    destination = _archive(row)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot.zip"
        with ZipFile(staged, "w", compression=ZIP_DEFLATED) as archive:
            for split in ("train", "validation"):
                with archive.open(f"{split}.jsonl", "w") as output:
                    for doc in documents:
                        if ("validation" if doc.id in validation_ids else "train") != split:
                            continue
                        if not doc.text_uri or not doc.content_hash:
                            raise ApiError(
                                409, "source_changed", "A selected document has no cleaned text."
                            )
                        source = get_settings().data_dir / "files" / doc.text_uri
                        try:
                            data = source.read_bytes()
                        except OSError as problem:
                            raise ApiError(
                                409,
                                "source_changed",
                                f"Cleaned text for {doc.filename} is missing.",
                            ) from problem
                        if hashlib.sha256(data).hexdigest() != doc.content_hash:
                            raise ApiError(
                                409, "source_changed", f"Cleaned text for {doc.filename} changed."
                            )
                        bytes_read += len(data)
                        if bytes_read > MAX_TEXT_BYTES:
                            raise ApiError(
                                422, "too_large", "This bucket exceeds the 20 MB snapshot limit."
                            )
                        text = data.decode("utf-8")
                        chars += len(text)
                        counts[split] += 1
                        sources.append(
                            {"id": str(doc.id), "hash": doc.content_hash, "split": split}
                        )
                        record = {
                            "id": str(doc.id),
                            "filename": doc.filename,
                            "source_format": doc.source_format,
                            "source_note": doc.source_note,
                            "text": text,
                        }
                        output.write(
                            (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                        )
            manifest = {
                "format": "forge-text-snapshot-v1",
                "dataset_id": str(row.id),
                "name": row.name,
                "bucket_id": str(row.bucket_id),
                "seed": seed,
                "validation_fraction": fraction,
                "documents": counts,
                "characters": chars,
                "token_count": 0,
                "sources": sources,
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        os.replace(staged, destination)

    row.status = "ready"
    row.doc_count = len(documents)
    row.char_count = chars
    row.token_count = 0
    row.artifact_uri = f"datasets/{row.id}.zip"
    row.config = {
        "format": "forge-text-snapshot-v1",
        "seed": seed,
        "validation_fraction": fraction,
        "train_documents": counts["train"],
        "validation_documents": counts["validation"],
        "sha256": _sha256(destination),
    }


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, session: Session = Depends(get_session)) -> DatasetOut:
    if session.get(Bucket, payload.bucket_id) is None:
        raise ApiError(404, "not_found", "No bucket with that id.")
    if session.scalar(select(Dataset.id).where(Dataset.name == payload.name)):
        raise ApiError(409, "name_taken", "That dataset name is already in use.")
    documents = list(
        session.scalars(
            select(Document)
            .where(Document.bucket_id == payload.bucket_id, Document.status == "parsed")
            .order_by(Document.content_hash, Document.id)
            .limit(MAX_DOCUMENTS + 1)
        )
    )
    if len(documents) < 2:
        raise ApiError(422, "not_enough_data", "Add at least two parsed documents first.")
    if len(documents) > MAX_DOCUMENTS:
        raise ApiError(422, "too_large", "This bucket exceeds the 1000 document snapshot limit.")
    row = Dataset(
        id=uuid.uuid4(),
        name=payload.name,
        description=payload.description,
        bucket_id=payload.bucket_id,
    )
    try:
        _snapshot(row, documents, payload.validation_fraction, payload.seed)
        session.add(row)
        session.commit()
    except IntegrityError as problem:
        session.rollback()
        _archive(row).unlink(missing_ok=True)
        raise ApiError(409, "name_taken", "That dataset name is already in use.") from problem
    except Exception:
        session.rollback()
        _archive(row).unlink(missing_ok=True)
        raise
    return _out(row)


@router.get("", response_model=Page[DatasetOut])
def list_datasets(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[DatasetOut]:
    rows = session.scalars(
        select(Dataset).order_by(Dataset.created_at.desc(), Dataset.id).limit(limit).offset(offset)
    )
    total = session.scalar(select(func.count()).select_from(Dataset)) or 0
    return Page(items=[_out(row) for row in rows], total=total)


def _get(session: Session, dataset_id: uuid.UUID) -> Dataset:
    row = session.get(Dataset, dataset_id)
    if row is None:
        raise ApiError(404, "not_found", "No dataset with that id.")
    return row


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> DatasetOut:
    return _out(_get(session, dataset_id))


@router.get("/{dataset_id}/download", response_class=FileResponse)
def download_dataset(
    dataset_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    row = _get(session, dataset_id)
    path = _archive(row)
    if row.status != "ready" or not row.artifact_uri or not path.is_file():
        raise ApiError(409, "not_ready", "The dataset archive is not available.")
    if _sha256(path) != row.config.get("sha256"):
        raise ApiError(409, "artifact_changed", "The dataset archive changed on disk.")
    return FileResponse(path, media_type="application/zip", filename=f"{row.name}.zip")


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    row = _get(session, dataset_id)
    if session.scalar(select(Run.id).where(Run.dataset_id == dataset_id).limit(1)):
        raise ApiError(409, "in_use", "A run uses this dataset. Keep it with the run.")
    path = _archive(row)
    session.delete(row)
    session.commit()
    path.unlink(missing_ok=True)
    return Response(status_code=204)
