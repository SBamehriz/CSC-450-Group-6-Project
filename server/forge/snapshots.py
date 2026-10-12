import hashlib
import json
import math
import random
import uuid
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.errors import ApiError
from forge.models import Bucket, Dataset, Document

MAX_DOCUMENTS = 1000
MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def archive_path(dataset: Dataset) -> Path:
    expected = f"datasets/{dataset.id}.zip"
    if dataset.artifact_uri != expected:
        raise ApiError(409, "not_ready", "This record has no text snapshot. Build a new one.")
    return get_settings().data_dir / expected


def source_rows(session: Session, bucket_id: uuid.UUID) -> tuple[Bucket, list[dict]]:
    bucket = session.get(Bucket, bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "No collection with that id.")
    pending = session.scalar(
        select(Document.id).where(Document.bucket_id == bucket_id, Document.status == "pending")
    )
    if pending is not None:
        raise ApiError(
            409, "not_ready", "Some files are still being read. Try again once they finish."
        )
    documents = session.scalars(
        select(Document)
        .where(Document.bucket_id == bucket_id, Document.status == "parsed")
        .order_by(Document.content_hash, Document.id)
        .limit(MAX_DOCUMENTS + 1)
    ).all()
    if not 2 <= len(documents) <= MAX_DOCUMENTS:
        raise ApiError(422, "invalid_source", "Choose a collection with 2 to 1,000 parsed files.")
    root = (get_settings().data_dir / "files").resolve()
    rows, total_bytes = [], 0
    for document in documents:
        path = (root / (document.text_uri or "")).resolve()
        if not document.text_uri or not path.is_relative_to(root) or not path.is_file():
            raise ApiError(
                409, "missing_text", f"The cleaned text for {document.filename!r} is missing."
            )
        total_bytes += path.stat().st_size
        if total_bytes > MAX_SNAPSHOT_BYTES:
            raise ApiError(
                413, "too_large", "This snapshot is over 50 MB. Use a smaller collection."
            )
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as problem:
            raise ApiError(
                409, "missing_text", f"Could not read {document.filename!r}."
            ) from problem
        digest = hashlib.sha256(raw).hexdigest()
        if not text.strip() or digest != document.content_hash:
            raise ApiError(
                409,
                "changed_text",
                f"The saved text for {document.filename!r} has changed. Upload it again.",
            )
        rows.append(
            {
                "id": str(document.id),
                "filename": document.filename,
                "source_note": document.source_note,
                "content_hash": digest,
                "text": text,
                "quality": document.quality,
            }
        )
    return bucket, rows


def split_rows(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * fraction)))
    return ordered[validation_count:], ordered[:validation_count]


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_snapshot(
    dataset: Dataset, bucket: Bucket, rows: list[dict], fraction: float, seed: int
) -> None:
    train, validation = split_rows(rows, fraction, seed)
    files = {}
    for split, records in (("train", train), ("validation", validation)):
        files[f"{split}.jsonl"] = b"".join(
            (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            for record in records
        )
    manifest = {
        "format": "forge-text-snapshot-v1",
        "dataset_id": str(dataset.id),
        "name": dataset.name,
        "bucket_id": str(bucket.id),
        "bucket_name": bucket.name,
        "seed": seed,
        "validation_fraction": fraction,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        "documents": [
            {
                "id": row["id"],
                "filename": row["filename"],
                "content_hash": row["content_hash"],
                "split": split,
            }
            for split, records in (("train", train), ("validation", validation))
            for row in records
        ],
    }
    destination = get_settings().data_dir / "datasets" / f"{dataset.id}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with ZipFile(temporary, "w") as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
            archive.writestr("manifest.json", json_bytes(manifest))
        temporary.replace(destination)
        dataset.artifact_uri = f"datasets/{dataset.id}.zip"
        dataset.doc_count = len(rows)
        dataset.char_count = sum(len(row["text"]) for row in rows)
        dataset.token_count = 0
        dataset.config = {
            "format": manifest["format"],
            "seed": seed,
            "validation_fraction": fraction,
            "train_documents": len(train),
            "validation_documents": len(validation),
            "bucket_name": bucket.name,
            "sha256": file_hash(destination),
            "size_bytes": destination.stat().st_size,
        }
        dataset.status = "ready"
    except OSError as problem:
        destination.unlink(missing_ok=True)
        raise ApiError(
            503, "storage_unavailable", "Could not save the snapshot. Check free disk space."
        ) from problem
    finally:
        temporary.unlink(missing_ok=True)


def read_snapshot(dataset: Dataset) -> tuple[dict, dict[str, list[dict]]]:
    path = archive_path(dataset)
    try:
        if not path.is_file() or file_hash(path) != dataset.config.get("sha256"):
            raise ValueError("The snapshot is missing or its checksum changed. Build a new one.")
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "forge-text-snapshot-v1" or manifest.get(
                "dataset_id"
            ) != str(dataset.id):
                raise ValueError("The snapshot manifest does not match this dataset.")
            splits = {}
            for split in ("train", "validation"):
                filename = f"{split}.jsonl"
                raw = archive.read(filename)
                if hashlib.sha256(raw).hexdigest() != manifest["files"][filename]:
                    raise ValueError("A text file inside the snapshot has changed.")
                records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
                if not records:
                    raise ValueError("Each split needs at least one document.")
                for record in records:
                    if (
                        hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                        != record["content_hash"]
                    ):
                        raise ValueError("A document inside the snapshot has changed.")
                splits[split] = records
            train_ids = {row["id"] for row in splits["train"]}
            validation_ids = {row["id"] for row in splits["validation"]}
            if train_ids & validation_ids:
                raise ValueError("A document appears in both splits.")
            return manifest, splits
    except (OSError, BadZipFile, ValueError, KeyError, TypeError) as problem:
        raise ApiError(
            409,
            "invalid_snapshot",
            str(problem)
            if isinstance(problem, ValueError)
            else "Could not read the saved snapshot. Build a new one.",
        ) from problem
