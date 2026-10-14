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

CHARS_PER_TOKEN = 4
MAX_DOCUMENTS = 1000
MAX_TEXT_BYTES = 20 * 1024 * 1024


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


def _archive(row: Dataset) -> Path:
    return get_settings().data_dir / "datasets" / f"{row.id}.zip"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(
    row: Dataset,
    documents: list[Document],
    fraction: float,
    seed: int,
    extra_config: dict | None = None,
) -> None:
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
                        source = (
                            (get_settings().data_dir / "files" / doc.text_uri)
                            if doc.text_uri
                            else None
                        )
                        if source is not None and source.is_file():
                            try:
                                data = source.read_bytes()
                            except OSError as problem:
                                raise ApiError(
                                    409,
                                    "source_changed",
                                    f"Cleaned text for {doc.filename} is missing.",
                                ) from problem
                            if (
                                doc.content_hash
                                and hashlib.sha256(data).hexdigest() != doc.content_hash
                            ):
                                raise ApiError(
                                    409,
                                    "source_changed",
                                    f"Cleaned text for {doc.filename} changed.",
                                )
                        elif doc.text_uri and (source is None or not source.is_file()):
                            raise ApiError(
                                409,
                                "source_changed",
                                f"Cleaned text for {doc.filename} is missing.",
                            )
                        else:
                            data = (
                                f"Document content for {doc.filename}\n"
                                * max(1, doc.char_count // 30)
                            )[: doc.char_count].encode("utf-8")
                            if not doc.content_hash:
                                doc.content_hash = hashlib.sha256(data).hexdigest()

                        bytes_read += len(data)
                        if bytes_read > MAX_TEXT_BYTES:
                            raise ApiError(
                                422,
                                "too_large",
                                "This bucket exceeds the 20 MB snapshot limit.",
                            )
                        text = data.decode("utf-8", errors="replace")
                        chars += len(text)
                        counts[split] += 1
                        sources.append(
                            {"id": str(doc.id), "hash": doc.content_hash or "", "split": split}
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
        **(extra_config or {}),
        "format": "forge-text-snapshot-v1",
        "seed": seed,
        "validation_fraction": fraction,
        "train_documents": counts["train"],
        "validation_documents": counts["validation"],
        "sha256": _sha256(destination),
    }


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, session: Session = Depends(get_session)) -> DatasetOut:
    if session.scalar(select(Dataset.id).where(Dataset.name == payload.name)):
        raise ApiError(409, "name_taken", f"A dataset named {payload.name!r} already exists.")

    if payload.bucket_id is None:
        row = Dataset(
            id=uuid.uuid4(),
            name=payload.name,
            description=payload.description,
            bucket_id=None,
            status="draft",
            doc_count=0,
            char_count=0,
            token_count=0,
            config=payload.config,
        )
        session.add(row)
        session.commit()
        return _to_out(row)

    bucket = session.get(Bucket, payload.bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "No bucket with that id.")

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
        raise ApiError(
            422,
            "too_large",
            f"This bucket exceeds the {MAX_DOCUMENTS} document snapshot limit.",
        )

    row = Dataset(
        id=uuid.uuid4(),
        name=payload.name,
        description=payload.description,
        bucket_id=payload.bucket_id,
        status="draft",
    )
    try:
        _snapshot(
            row,
            documents,
            fraction=payload.validation_fraction,
            seed=payload.seed,
            extra_config=payload.config,
        )
        session.add(row)
        session.commit()
    except IntegrityError as problem:
        session.rollback()
        _archive(row).unlink(missing_ok=True)
        raise ApiError(
            409, "name_taken", f"A dataset named {payload.name!r} already exists."
        ) from problem
    except Exception:
        session.rollback()
        _archive(row).unlink(missing_ok=True)
        raise
    return _to_out(row)


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

    total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()

    datasets = (
        session.execute(
            query.order_by(Dataset.created_at.desc(), Dataset.id).limit(limit).offset(offset)
        )
        .scalars()
        .all()
    )

    return Page[DatasetOut](items=[_to_out(d) for d in datasets], total=total)


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> DatasetOut:
    dataset = _load_one(session, dataset_id)
    return _to_out(dataset)


@router.get("/{dataset_id}/download", response_class=FileResponse)
def download_dataset(
    dataset_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    row = _load_one(session, dataset_id)
    path = _archive(row)
    if row.status != "ready" or not row.artifact_uri or not path.is_file():
        raise ApiError(409, "not_ready", "The dataset archive is not available.")
    if _sha256(path) != row.config.get("sha256"):
        raise ApiError(409, "artifact_changed", "The dataset archive changed on disk.")
    return FileResponse(path, media_type="application/zip", filename=f"{row.name}.zip")


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    row = _load_one(session, dataset_id)
    if session.scalar(select(Run.id).where(Run.dataset_id == dataset_id).limit(1)):
        raise ApiError(409, "in_use", "A run uses this dataset. Keep it with the run.")
    path = _archive(row)
    session.delete(row)
    session.commit()
    path.unlink(missing_ok=True)
    return Response(status_code=204)
