import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.errors import ApiError
from forge.ingest import clean
from forge.jobs import Canceled, report
from forge.models import Bucket, Document, Job
from forge.schemas import (
    DatasetImportCreate,
    HfConfigInfo,
    HfInspectResponse,
    HfSplitInfo,
)

log = logging.getLogger(__name__)

HF_HUB_API = "https://huggingface.co/api/datasets"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
BATCH_SIZE = 100


def get_hf_headers() -> dict[str, str]:
    """Build request headers with optional credentials from server environment."""
    headers = {
        "User-Agent": "Forge-CSC450/1.0",
        "Accept": "application/json",
    }
    # Keep credentials securely on the server
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            pass

    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_dataset_identifier(dataset_input: str) -> str:
    """Normalize public Hugging Face dataset name or URL to repository ID."""
    cleaned = dataset_input.strip()
    if not cleaned:
        raise ApiError(400, "invalid_dataset", "Please enter a dataset name or URL.")

    # Match URLs such as https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
    match = re.search(r"huggingface\.co/datasets/([^/?#]+(?:/[^/?#]+)?)", cleaned)
    if match:
        return match.group(1).strip()

    # Match repository identifier like "imdb" or "stanfordnlp/imdb"
    cleaned = cleaned.rstrip("/")
    if re.match(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$", cleaned):
        return cleaned

    raise ApiError(
        400,
        "invalid_dataset",
        f"'{dataset_input}' is not a recognized Hugging Face dataset name or URL.",
    )


def resolve_dataset(dataset_name: str) -> tuple[str, str]:
    """Resolve the dataset to its canonical repository ID and pinned commit revision (sha)."""
    normalized = parse_dataset_identifier(dataset_name)
    url = f"{HF_HUB_API}/{normalized}"
    req = urllib.request.Request(url, headers=get_hf_headers())

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise ApiError(
                404,
                "dataset_not_found",
                f"Dataset '{normalized}' was not found on Hugging Face.",
            ) from err
        if err.code in (401, 403):
            raise ApiError(
                403,
                "dataset_forbidden",
                f"Dataset '{normalized}' is private or gated. Check server credentials.",
            ) from err
        raise ApiError(
            502,
            "hf_network_error",
            f"Hugging Face returned status {err.code} while resolving '{normalized}'.",
        ) from err
    except urllib.error.URLError as err:
        raise ApiError(
            502,
            "hf_network_error",
            f"Network failure connecting to Hugging Face: {err.reason}",
        ) from err
    except Exception as err:
        raise ApiError(
            502,
            "hf_network_error",
            f"Unexpected error resolving dataset on Hugging Face: {err}",
        ) from err

    canonical_id = data.get("id", normalized)
    sha = data.get("sha")
    if not sha or not isinstance(sha, str):
        raise ApiError(
            502,
            "revision_missing",
            f"Could not resolve an immutable commit revision for '{canonical_id}'.",
        )

    return canonical_id, sha


def inspect_dataset(dataset_input: str) -> HfInspectResponse:
    """Inspect dataset metadata, discovering configurations, splits, and string columns."""
    canonical_id, sha = resolve_dataset(dataset_input)
    headers = get_hf_headers()

    # 1. Query info endpoint for comprehensive schema across all configs
    info_url = f"{HF_DATASETS_SERVER}/info?dataset={urllib.parse.quote(canonical_id)}"
    info_req = urllib.request.Request(info_url, headers=headers)
    dataset_info: dict = {}
    try:
        with urllib.request.urlopen(info_req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dataset_info = data.get("dataset_info", {})
    except Exception as problem:
        log.warning("Could not fetch info endpoint for %s: %s", canonical_id, problem)

    configs_list: list[HfConfigInfo] = []

    if dataset_info and isinstance(dataset_info, dict):
        for cfg_name, details in dataset_info.items():
            if not isinstance(details, dict):
                continue
            splits_data = details.get("splits", {})
            splits: list[HfSplitInfo] = []
            if isinstance(splits_data, dict):
                for s_name, s_info in splits_data.items():
                    num = s_info.get("num_examples") if isinstance(s_info, dict) else None
                    splits.append(HfSplitInfo(name=s_name, num_examples=num))

            features = details.get("features", {})
            columns: list[str] = []
            text_columns: list[str] = []
            if isinstance(features, dict):
                for f_name, f_type in features.items():
                    columns.append(f_name)
                    if isinstance(f_type, dict) and f_type.get("dtype") == "string":
                        text_columns.append(f_name)
                    elif f_type == "string":
                        text_columns.append(f_name)

            if not text_columns:
                # If no explicit string dtype detected, include all columns as candidates
                text_columns = list(columns)

            configs_list.append(
                HfConfigInfo(
                    name=cfg_name,
                    splits=splits,
                    columns=columns,
                    text_columns=text_columns,
                )
            )

    # 2. Fallback to /splits and /rows if /info did not return configs
    if not configs_list:
        splits_url = f"{HF_DATASETS_SERVER}/splits?dataset={urllib.parse.quote(canonical_id)}"
        splits_req = urllib.request.Request(splits_url, headers=headers)
        try:
            with urllib.request.urlopen(splits_req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_splits = data.get("splits", [])
        except Exception as problem:
            raise ApiError(
                502,
                "hf_network_error",
                f"Could not fetch dataset splits from Hugging Face: {problem}",
            ) from problem

        # Group by config
        grouped: dict[str, list[str]] = {}
        for s in raw_splits:
            cfg = s.get("config", "default")
            s_name = s.get("split")
            if s_name:
                grouped.setdefault(cfg, []).append(s_name)

        for cfg_name, split_names in grouped.items():
            splits = [HfSplitInfo(name=sn) for sn in split_names]
            columns: list[str] = []
            text_columns: list[str] = []

            # Sample first split to extract column names
            if split_names:
                sample_split = split_names[0]
                rows_url = (
                    f"{HF_DATASETS_SERVER}/rows?dataset={urllib.parse.quote(canonical_id)}"
                    f"&config={urllib.parse.quote(cfg_name)}"
                    f"&split={urllib.parse.quote(sample_split)}&offset=0&length=1"
                )
                try:
                    sample_req = urllib.request.Request(rows_url, headers=headers)
                    with urllib.request.urlopen(sample_req, timeout=15) as s_resp:
                        rows_resp = json.loads(s_resp.read().decode("utf-8"))
                        for feat in rows_resp.get("features", []):
                            fname = feat.get("name")
                            if fname:
                                columns.append(fname)
                                ftype = feat.get("type")
                                if isinstance(ftype, dict) and ftype.get("dtype") == "string":
                                    text_columns.append(fname)
                except Exception as problem:
                    log.warning("Could not sample rows for columns: %s", problem)

            if not text_columns and columns:
                text_columns = list(columns)

            configs_list.append(
                HfConfigInfo(
                    name=cfg_name,
                    splits=splits,
                    columns=columns,
                    text_columns=text_columns,
                )
            )

    if not configs_list:
        raise ApiError(
            400,
            "no_configs_found",
            f"No usable configurations or splits were found for '{canonical_id}'.",
        )

    return HfInspectResponse(
        dataset=dataset_input,
        canonical_id=canonical_id,
        sha=sha,
        configs=configs_list,
    )


def _storage_root() -> Path:
    return get_settings().data_dir / "files"


def _write_file(relative: str, data: bytes) -> str:
    path = _storage_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return relative


def validate_import_params(
    canonical_id: str,
    payload: DatasetImportCreate,
) -> None:
    """Validate requested config, split, and text_column against dataset schema."""
    schema = inspect_dataset(canonical_id)
    matched_config = next((c for c in schema.configs if c.name == payload.config), None)
    if matched_config is None:
        valid_configs = ", ".join(repr(c.name) for c in schema.configs)
        raise ApiError(
            400,
            "invalid_config",
            f"Configuration '{payload.config}' does not exist for this dataset. "
            f"Available: {valid_configs}.",
        )

    matched_split = next((s for s in matched_config.splits if s.name == payload.split), None)
    if matched_split is None:
        valid_splits = ", ".join(repr(s.name) for s in matched_config.splits)
        raise ApiError(
            400,
            "invalid_split",
            f"Split '{payload.split}' does not exist in config '{payload.config}'. "
            f"Available: {valid_splits}.",
        )

    if payload.text_column not in matched_config.columns:
        valid_cols = ", ".join(repr(col) for col in matched_config.columns)
        raise ApiError(
            400,
            "invalid_column",
            f"Column '{payload.text_column}' was not found. Available: {valid_cols}.",
        )


def import_dataset(job_id: uuid.UUID) -> None:
    """Execute the background dataset import job in bounded batches."""
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.status != "running":
            return
        payload = job.payload
        bucket_id = uuid.UUID(payload["target_id"])
        bucket = session.get(Bucket, bucket_id)
        if bucket is None:
            raise ValueError("The collection this job was importing into no longer exists.")

        dataset_input = payload["dataset"]
        config = payload["config"]
        split = payload["split"]
        text_column = payload["text_column"]
        import_limit = int(payload.get("limit", 500))

    # Resolve and pin to immutable commit revision
    canonical_id, resolved_revision = resolve_dataset(dataset_input)

    # Save pinned source metadata onto collection
    with Session(get_engine()) as session:
        bucket = session.get(Bucket, bucket_id)
        if bucket is not None:
            bucket.source_dataset = canonical_id
            bucket.resolved_revision = resolved_revision
            bucket.dataset_config = config
            bucket.dataset_split = split
            bucket.text_column = text_column
            bucket.import_limit = import_limit
            session.commit()

    # Discover total if known
    total = 0
    try:
        schema = inspect_dataset(canonical_id)
        for cfg in schema.configs:
            if cfg.name == config:
                for sp in cfg.splits:
                    if sp.name == split and sp.num_examples is not None:
                        total = min(import_limit, sp.num_examples)
                        break
    except Exception as problem:
        log.warning("Could not determine total examples for %s: %s", canonical_id, problem)

    headers = get_hf_headers()
    processed = 0
    imported = 0
    skipped = 0
    failed = 0

    report(
        job_id,
        completed=0,
        total=total,
        message=f"Starting import from {canonical_id}...",
        processed=0,
        imported=0,
        skipped=0,
        failed=0,
    )

    offset = 0
    while offset < import_limit:
        # Check cancellation
        with Session(get_engine()) as session:
            chk_job = session.get(Job, job_id)
            if chk_job and chk_job.cancel_requested:
                raise Canceled()

        current_batch_size = min(BATCH_SIZE, import_limit - offset)
        url = (
            f"{HF_DATASETS_SERVER}/rows?dataset={urllib.parse.quote(canonical_id)}"
            f"&config={urllib.parse.quote(config)}"
            f"&split={urllib.parse.quote(split)}"
            f"&offset={offset}&length={current_batch_size}"
        )

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"Hugging Face API returned error {err.code}: {err.reason}") from err
        except Exception as err:
            raise RuntimeError(
                f"Network failure while streaming rows from Hugging Face: {err}"
            ) from err

        rows = rows_data.get("rows", [])
        if not rows:
            break

        with Session(get_engine()) as session:
            for item in rows:
                row_dict = item.get("row", {})
                row_idx = item.get("row_idx", offset)
                processed += 1

                if text_column not in row_dict:
                    failed += 1
                    continue

                raw_text = row_dict[text_column]
                if raw_text is None or not isinstance(raw_text, str) or not raw_text.strip():
                    skipped += 1
                    continue

                # Pass through standard cleaning
                cleaned = clean(raw_text, encoding_suspect=False)
                if not cleaned.text.strip():
                    skipped += 1
                    continue

                # Duplicate check against existing documents in this collection
                twin = session.execute(
                    select(Document).where(
                        Document.bucket_id == bucket_id,
                        Document.content_hash == cleaned.content_hash,
                    )
                ).scalar_one_or_none()
                if twin is not None:
                    skipped += 1
                    continue

                # Persist raw text and cleaned text
                raw_bytes = raw_text.encode("utf-8")
                raw_uri = _write_file(f"raw/{uuid.uuid4()}", raw_bytes)

                filename = f"hf_{config}_{split}_{row_idx:06d}.txt"
                source_note = (
                    f"Imported from Hugging Face: {canonical_id} "
                    f"(revision: {resolved_revision[:8]}, config: {config}, "
                    f"split: {split}, row: {row_idx})"
                )

                doc = Document(
                    bucket_id=bucket_id,
                    filename=filename,
                    source_format="txt",
                    source_note=source_note,
                    raw_uri=raw_uri,
                    status="parsed",
                    content_hash=cleaned.content_hash,
                    char_count=cleaned.char_count,
                    word_count=cleaned.word_count,
                    quality=cleaned.quality,
                )
                session.add(doc)
                session.flush()
                doc.text_uri = _write_file(f"text/{doc.id}.txt", cleaned.text.encode("utf-8"))
                imported += 1

            session.commit()

        offset += len(rows)

        counts = f"{imported} imported, {skipped} skipped, {failed} failed"
        if total > 0:
            msg = f"{processed} / {total} records ({counts})"
        else:
            msg = f"{processed} records processed ({counts})"

        report(
            job_id,
            completed=processed,
            total=total,
            message=msg,
            processed=processed,
            imported=imported,
            skipped=skipped,
            failed=failed,
        )

    if imported == 0 and failed > 0:
        raise RuntimeError(
            f"Import failed: all {failed} records failed to read column '{text_column}'."
        )

    if processed == 0:
        raise RuntimeError(
            f"No records found in dataset '{canonical_id}' config '{config}' split '{split}'."
        )

    summary_msg = (
        f"Import complete: {imported} imported, {skipped} skipped, {failed} failed "
        f"({processed} processed)"
    )
    report(
        job_id,
        completed=processed,
        total=total,
        message=summary_msg,
        processed=processed,
        imported=imported,
        skipped=skipped,
        failed=failed,
    )
