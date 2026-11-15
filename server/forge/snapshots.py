import hashlib
import json
import math
import random
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.errors import ApiError
from forge.jobs import Canceled, report
from forge.models import Bucket, Dataset, Document, Job
from forge.storage import acquire_bucket_lock


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def archive_path(dataset: Dataset, zip_path: Path | None = None) -> Path:
    # an old saved zip, or one packed now from the folder
    zip_path = zip_path or get_settings().data_dir / "datasets" / f"{dataset.id}.zip"
    if zip_path.is_file():
        return zip_path
    folder = get_settings().data_dir / "datasets" / str(dataset.id)
    if (folder / "manifest.json").is_file():
        # Package zip on demand if separate files exist but zip was removed
        temporary = zip_path.with_name(f"{zip_path.name}.tmp")
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for item in ("train.jsonl", "validation.jsonl", "manifest.json"):
                fpath = folder / item
                if fpath.is_file():
                    archive.write(fpath, item)
        temporary.replace(zip_path)
        return zip_path
    raise ApiError(409, "not_ready", "This record has no text snapshot. Build a new one.")


def stored_ok(dataset: Dataset) -> bool:
    # the checksum is the manifest's, or the zip's for an older snapshot
    expected = dataset.config.get("sha256")
    folder = get_settings().data_dir / "datasets" / str(dataset.id)
    zip_path = folder.with_name(f"{dataset.id}.zip")
    manifest = folder / "manifest.json"
    if manifest.is_file() and (not expected or file_hash(manifest) == expected):
        return True
    return zip_path.is_file() and (not expected or file_hash(zip_path) == expected)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_snapshot_streaming(
    dataset_id: uuid.UUID,
    bucket_id: uuid.UUID,
    fraction: float,
    seed: int,
    *,
    job_id: uuid.UUID | None = None,
) -> None:
    """Stream snapshot documents in bounded batches, writing separate disk files and zip archive."""
    with Session(get_engine()) as session:
        bucket = session.get(Bucket, bucket_id)
        dataset = session.get(Dataset, dataset_id)
        if bucket is None or dataset is None:
            raise ApiError(404, "not_found", "Dataset or collection was removed.")

        free_bytes = shutil.disk_usage(get_settings().data_dir).free
        if free_bytes < 10 * 1024 * 1024:
            raise ApiError(
                503, "storage_unavailable", "Could not save the snapshot. Check free disk space."
            )

        # Freeze document membership and content hash versions at start
        frozen_entries = session.execute(
            select(Document.id, Document.content_hash)
            .where(Document.bucket_id == bucket_id, Document.status == "parsed")
            .order_by(Document.content_hash, Document.id)
        ).all()
        if len(frozen_entries) < 2:
            raise ApiError(
                422, "invalid_source", "Choose a collection with at least two parsed files."
            )

        bucket_name = bucket.name
        dataset_name = dataset.name

    # Deterministic split: shuffle frozen entries using fixed seed
    ordered = list(frozen_entries)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * fraction)))
    validation_entries = ordered[:validation_count]
    train_entries = ordered[validation_count:]
    total_docs = len(ordered)

    data_dir = get_settings().data_dir
    root = (data_dir / "files").resolve()
    dest_folder = data_dir / "datasets" / str(dataset_id)
    dest_folder.parent.mkdir(parents=True, exist_ok=True)

    temp_folder = dest_folder.with_name(f"{dataset_id}-dir.tmp")
    shutil.rmtree(temp_folder, ignore_errors=True)
    temp_folder.mkdir(parents=True, exist_ok=True)

    train_path = temp_folder / "train.jsonl"
    val_path = temp_folder / "validation.jsonl"
    manifest_path = temp_folder / "manifest.json"

    train_hasher = hashlib.sha256()
    val_hasher = hashlib.sha256()
    manifest_documents = []
    total_chars = 0
    completed = 0
    batch_size = 100

    try:
        # Stream train documents in bounded batches
        with train_path.open("wb") as train_out, Session(get_engine()) as session:
            for i in range(0, len(train_entries), batch_size):
                chunk = train_entries[i : i + batch_size]
                chunk_ids = [entry[0] for entry in chunk]
                docs = session.scalars(select(Document).where(Document.id.in_(chunk_ids))).all()
                doc_map = {doc.id: doc for doc in docs}
                for doc_id, expected_hash in chunk:
                    doc = doc_map.get(doc_id)
                    if doc is None or not doc.text_uri:
                        raise ApiError(409, "missing_text", f"Document {doc_id} text is missing.")
                    fpath = (root / doc.text_uri).resolve()
                    if not fpath.is_relative_to(root) or not fpath.is_file():
                        raise ApiError(
                            409,
                            "missing_text",
                            f"The cleaned text for {doc.filename!r} is missing.",
                        )
                    try:
                        raw = fpath.read_bytes()
                        text = raw.decode("utf-8")
                    except (OSError, UnicodeError) as exc:
                        raise ApiError(
                            409, "missing_text", f"Could not read {doc.filename!r}."
                        ) from exc
                    digest = hashlib.sha256(raw).hexdigest()
                    if not text.strip() or digest != expected_hash:
                        raise ApiError(
                            409,
                            "changed_text",
                            f"The saved text for {doc.filename!r} has changed. Upload it again.",
                        )
                    record = {
                        "id": str(doc.id),
                        "filename": doc.filename,
                        "source_note": doc.source_note,
                        "content_hash": digest,
                        "text": text,
                        "quality": doc.quality,
                    }
                    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                    train_out.write(line)
                    train_hasher.update(line)
                    total_chars += len(text)
                    manifest_documents.append(
                        {
                            "id": str(doc.id),
                            "filename": doc.filename,
                            "content_hash": digest,
                            "split": "train",
                        }
                    )
                    completed += 1
                if job_id:
                    report(job_id, completed, total_docs, "Writing training text")

        # Stream validation documents in bounded batches
        with val_path.open("wb") as val_out, Session(get_engine()) as session:
            for i in range(0, len(validation_entries), batch_size):
                chunk = validation_entries[i : i + batch_size]
                chunk_ids = [entry[0] for entry in chunk]
                docs = session.scalars(select(Document).where(Document.id.in_(chunk_ids))).all()
                doc_map = {doc.id: doc for doc in docs}
                for doc_id, expected_hash in chunk:
                    doc = doc_map.get(doc_id)
                    if doc is None or not doc.text_uri:
                        raise ApiError(409, "missing_text", f"Document {doc_id} text is missing.")
                    fpath = (root / doc.text_uri).resolve()
                    if not fpath.is_relative_to(root) or not fpath.is_file():
                        raise ApiError(
                            409,
                            "missing_text",
                            f"The cleaned text for {doc.filename!r} is missing.",
                        )
                    try:
                        raw = fpath.read_bytes()
                        text = raw.decode("utf-8")
                    except (OSError, UnicodeError) as exc:
                        raise ApiError(
                            409, "missing_text", f"Could not read {doc.filename!r}."
                        ) from exc
                    digest = hashlib.sha256(raw).hexdigest()
                    if not text.strip() or digest != expected_hash:
                        raise ApiError(
                            409,
                            "changed_text",
                            f"The saved text for {doc.filename!r} has changed. Upload it again.",
                        )
                    record = {
                        "id": str(doc.id),
                        "filename": doc.filename,
                        "source_note": doc.source_note,
                        "content_hash": digest,
                        "text": text,
                        "quality": doc.quality,
                    }
                    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                    val_out.write(line)
                    val_hasher.update(line)
                    total_chars += len(text)
                    manifest_documents.append(
                        {
                            "id": str(doc.id),
                            "filename": doc.filename,
                            "content_hash": digest,
                            "split": "validation",
                        }
                    )
                    completed += 1
                if job_id:
                    report(job_id, completed, total_docs, "Writing validation text")

        manifest = {
            "format": "forge-text-snapshot-v1",
            "dataset_id": str(dataset_id),
            "name": dataset_name,
            "bucket_id": str(bucket_id),
            "bucket_name": bucket_name,
            "seed": seed,
            "validation_fraction": fraction,
            "files": {
                "train.jsonl": train_hasher.hexdigest(),
                "validation.jsonl": val_hasher.hexdigest(),
            },
            "documents": manifest_documents,
        }
        manifest_path.write_bytes(json_bytes(manifest))

        # the folder is the snapshot; a zip is packed only when someone downloads it
        zip_hash = file_hash(manifest_path)
        zip_size = sum(path.stat().st_size for path in (train_path, val_path, manifest_path))

        # Atomic publication: move temporary directory
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        temp_folder.rename(dest_folder)

        with Session(get_engine()) as session:
            ds = session.get(Dataset, dataset_id)
            if ds is not None:
                ds.artifact_uri = f"datasets/{dataset_id}"
                ds.doc_count = total_docs
                ds.char_count = total_chars
                ds.token_count = 0
                ds.config = {
                    "format": manifest["format"],
                    "seed": seed,
                    "validation_fraction": fraction,
                    "train_documents": len(train_entries),
                    "validation_documents": len(validation_entries),
                    "bucket_name": bucket_name,
                    "sha256": zip_hash,
                    "size_bytes": zip_size,
                }
                ds.status = "ready"
                session.commit()
    except Exception:
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise


def execute_snapshot_job(job_id: uuid.UUID) -> None:
    """Run snapshot creation job from the job queue."""
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        target_id = uuid.UUID(str(job.payload["target_id"]))
        bucket_id = uuid.UUID(str(job.payload["bucket_id"]))
        fraction = float(job.payload.get("validation_fraction", 0.1))
        seed = int(job.payload.get("seed", 42))

    try:
        with acquire_bucket_lock(bucket_id, timeout=5), Session(get_engine()) as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            dataset = session.get(Dataset, target_id)
            bucket = session.get(Bucket, bucket_id)
            if dataset is None or bucket is None:
                job.status = "failed"
                job.error = "The dataset or bucket no longer exists."
                job.completed_at = datetime.now(UTC)
                session.commit()
                return
            if job.cancel_requested:
                job.status = "cancelled"
                dataset.status = "failed"
                job.completed_at = datetime.now(UTC)
                session.commit()
                return
            job.status = "running"
            dataset.status = "processing"
            job.started_at = datetime.now(UTC)
            session.commit()

        write_snapshot_streaming(target_id, bucket_id, fraction, seed, job_id=job_id)
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            dataset = session.get(Dataset, target_id)
            if job and job.cancel_requested:
                raise Canceled()
            if job:
                job.status = "completed"
                job.pid = None
                job.completed_at = datetime.now(UTC)
            if dataset:
                dataset.status = "ready"
            session.commit()
    except Exception as problem:
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            dataset = session.get(Dataset, target_id)
            canceled = isinstance(problem, Canceled) or (job is not None and job.cancel_requested)
            err_msg = "Canceled before the snapshot was finished." if canceled else str(problem)
            if job:
                job.status = "cancelled" if canceled else "failed"
                job.error = err_msg
                job.pid = None
                job.completed_at = datetime.now(UTC)
            if dataset:
                dataset.status = "failed"
            session.commit()
        if not canceled:
            raise


def read_snapshot(dataset: Dataset) -> tuple[dict, dict[str, list[dict]]]:
    """Read snapshot manifest and document records from separate disk files or legacy zip."""
    if not stored_ok(dataset):
        raise ApiError(
            409,
            "invalid_snapshot",
            "The snapshot is missing or its checksum changed. Build a new one.",
        )

    folder = get_settings().data_dir / "datasets" / str(dataset.id)
    manifest_file = folder / "manifest.json"
    train_file = folder / "train.jsonl"
    val_file = folder / "validation.jsonl"

    if manifest_file.is_file() and train_file.is_file() and val_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("format") != "forge-text-snapshot-v1" or manifest.get(
                "dataset_id"
            ) != str(dataset.id):
                raise ValueError("The snapshot manifest does not match this dataset.")
            splits = {}
            for split, fpath in (("train", train_file), ("validation", val_file)):
                filename = f"{split}.jsonl"
                digest = file_hash(fpath)
                if digest != manifest["files"][filename]:
                    raise ValueError("A text file inside the snapshot has changed.")
                raw_lines = fpath.read_text(encoding="utf-8").splitlines()
                records = [json.loads(line) for line in raw_lines if line]
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
        except (OSError, ValueError, KeyError, TypeError) as problem:
            raise ApiError(
                409,
                "invalid_snapshot",
                str(problem)
                if isinstance(problem, ValueError)
                else "Could not read the saved snapshot. Build a new one.",
            ) from problem

    # Legacy ZIP archive path
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
