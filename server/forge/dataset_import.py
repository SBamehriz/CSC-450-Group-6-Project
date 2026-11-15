import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge import chatformat
from forge.config import get_settings
from forge.db import get_engine
from forge.errors import ApiError
from forge.ingest import clean, normalise
from forge.jobs import Canceled, report
from forge.models import Bucket, Document, Job
from forge.schemas import (
    DatasetImportCreate,
    HfConfigInfo,
    HfInspectResponse,
    HfSplitInfo,
)
from forge.storage import acquire_bucket_lock

log = logging.getLogger(__name__)

HF_HUB_API = "https://huggingface.co/api/datasets"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
BATCH_SIZE = 100
TARGET_FILES = 2_000
DOWNLOAD_CHUNK = 1 << 20
ROWS_RETRIES = 6
# rows per file when the size of the split is unknown, and the most text one file holds
PACK_ROWS = 20_000
PACK_CHARS = 32_000_000
# after this many values, stop if most of them could not be used
CHECK_ROWS = 2_000
TRIES = 4
WAIT = 2.0
READABLE = (".parquet", ".jsonl", ".jsonl.gz")
# the words a file's path is sorted into splits by when the card names none
SPLIT_WORDS = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "eval": "test",
}
RESOLVE = re.compile(r"https://huggingface\.co/datasets/(.+?)/resolve/([^/]+)/(.+)")


class NoParquetCopy(RuntimeError):
    """The hub has no Parquet copy of this split, so rows come 100 at a time."""


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
            chat_columns: list[str] = []
            if isinstance(features, dict):
                for f_name, f_type in features.items():
                    columns.append(f_name)
                    use = column_use(f_type)
                    if use == "text":
                        text_columns.append(f_name)
                    elif use == "chat":
                        chat_columns.append(f_name)

            configs_list.append(
                HfConfigInfo(
                    name=cfg_name,
                    splits=splits,
                    columns=columns,
                    text_columns=text_columns,
                    chat_columns=chat_columns,
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
            chat_columns: list[str] = []

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
                                use = column_use(feat.get("type"))
                                if use == "text":
                                    text_columns.append(fname)
                                elif use == "chat":
                                    chat_columns.append(fname)
                except Exception as problem:
                    log.warning("Could not sample rows for columns: %s", problem)

            configs_list.append(
                HfConfigInfo(
                    name=cfg_name,
                    splits=splits,
                    columns=columns,
                    text_columns=text_columns,
                    chat_columns=chat_columns,
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


def column_use(feature: object) -> str | None:
    """Says "text" for a column of strings, "chat" for a list of role and content turns."""
    if feature in ("string", "large_string"):
        return "text"
    if isinstance(feature, dict):
        if feature.get("dtype") in ("string", "large_string"):
            return "text"
        if feature.get("_type") not in ("List", "LargeList", "Sequence"):
            return None
        feature = [feature.get("feature")]
    if isinstance(feature, list) and len(feature) == 1 and isinstance(feature[0], dict):
        return "chat" if {"role", "content"} <= set(feature[0]) else None
    return None


def kind_problem(session: Session, bucket: Bucket, use: str | None) -> str | None:
    """Why rows of this kind cannot go into this collection, or None. An empty one takes any."""
    if use is None or bucket.kind == use:
        return None
    # a file that failed holds no text, so it does not count
    kept = select(Document.id).where(Document.bucket_id == bucket.id, Document.status == "parsed")
    if session.scalar(kept.limit(1)) is None:
        return None
    if use == "chat":
        return (
            "This collection already holds plain text, so conversations cannot go in it. "
            "Put them in a new or empty collection."
        )
    return (
        "This collection holds conversations, so plain text cannot go in it. "
        "Put it in a new or empty collection."
    )


def validate_import_params(
    canonical_id: str,
    payload: DatasetImportCreate,
) -> str:
    """Validate requested config, split, and text_column; say whether it holds text or chat."""
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

    usable = matched_config.text_columns + matched_config.chat_columns
    if payload.text_column not in usable:
        choice = (
            f"Pick one of: {', '.join(repr(col) for col in usable)}."
            if usable
            else "This part has no column of text or conversations."
        )
        raise ApiError(
            400,
            "column_not_text",
            f"Column '{payload.text_column}' holds neither text nor conversations. {choice}",
        )
    return "chat" if payload.text_column in matched_config.chat_columns else "text"


def import_dataset(job_id: uuid.UUID) -> None:
    """Execute the background dataset import job in bounded batches."""
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.status != "running":
            return
        payload = job.payload
        bucket_id = uuid.UUID(payload["target_id"])
        dataset_input = payload["dataset"]
        config = payload["config"]
        split = payload["split"]
        text_column = payload["text_column"]
        # None reads the whole split; "use" is "text" or "chat", None decides from the rows
        import_limit = int(payload["limit"]) if payload.get("limit") else None
        use = payload.get("use")
        pinned = job.result.get("revision")

    with acquire_bucket_lock(bucket_id, timeout=5), Session(get_engine()) as session:
        bucket = session.get(Bucket, bucket_id)
        if bucket is None:
            raise ValueError("The collection this job was importing into no longer exists.")

    # Resolve and pin to immutable commit revision; a retry keeps the first one
    canonical_id, resolved_revision = resolve_dataset(dataset_input)
    resolved_revision = pinned or resolved_revision

    # Save pinned source metadata onto collection
    with acquire_bucket_lock(bucket_id, timeout=5), Session(get_engine()) as session:
        bucket = session.get(Bucket, bucket_id)
        if bucket is None:
            raise ValueError("The collection this job was importing into no longer exists.")
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
                        total = min(import_limit or sp.num_examples, sp.num_examples)
                        break
    except Exception as problem:
        log.warning("Could not determine total examples for %s: %s", canonical_id, problem)

    headers = get_hf_headers()
    processed = 0
    imported = 0
    skipped = 0
    failed = 0
    repeats = 0

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

    # the dataset's own files; the hub's converted copy only when it has none Forge reads
    partial = False
    try:
        urls = data_file_urls(canonical_id, resolved_revision, config, split)
        if not urls:
            urls = parquet_urls(canonical_id, config, split)
            partial = any("/partial-" in url for url in urls)
    except NoParquetCopy:
        log.info("%s has no Parquet copy, reading rows one request at a time", canonical_id)
    else:
        _import_from_parquet(
            job_id,
            bucket_id,
            canonical_id,
            resolved_revision,
            config,
            split,
            text_column,
            import_limit,
            total,
            urls,
            use,
            partial,
        )
        return

    offset = 0
    while import_limit is None or offset < import_limit:
        # Check cancellation
        with Session(get_engine()) as session:
            chk_job = session.get(Job, job_id)
            if chk_job and chk_job.cancel_requested:
                raise Canceled()

        current_batch_size = min(BATCH_SIZE, (import_limit or offset + BATCH_SIZE) - offset)
        url = (
            f"{HF_DATASETS_SERVER}/rows?dataset={urllib.parse.quote(canonical_id)}"
            f"&config={urllib.parse.quote(config)}"
            f"&split={urllib.parse.quote(split)}"
            f"&offset={offset}&length={current_batch_size}"
        )

        rows_data = _rows(url, headers)

        rows = rows_data.get("rows", [])
        if not rows:
            break

        with acquire_bucket_lock(bucket_id, timeout=5), Session(get_engine()) as session:
            if session.get(Bucket, bucket_id) is None:
                raise ValueError("The collection this job was importing into no longer exists.")
            for item in rows:
                row_dict = item.get("row", {})
                row_idx = item.get("row_idx", offset)
                processed += 1

                if text_column not in row_dict:
                    failed += 1
                    continue

                value = row_dict[text_column]
                if use is None and value is not None:
                    use = "chat" if isinstance(value, list) else "text"
                raw_text, _why = _row_text(value, use)
                if raw_text is None:
                    skipped += 1
                    continue

                # imported rows get the light normalising only
                cleaned = clean(normalise(raw_text), mode="none")
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
                    repeats += 1
                    continue

                bucket = session.get(Bucket, bucket_id)
                problem = kind_problem(session, bucket, use) if bucket is not None else None
                if problem:
                    raise RuntimeError(problem)
                if bucket is not None and use:
                    bucket.kind = use

                # one file: the raw copy is the cleaned text
                doc_id = uuid.uuid4()
                raw_uri = _write_file(f"text/{doc_id}.txt", cleaned.text.encode("utf-8"))

                filename = f"hf_{config}_{split}_{row_idx:06d}.txt"
                source_note = (
                    f"Imported from Hugging Face: {canonical_id} "
                    f"(revision: {resolved_revision[:8]}, config: {config}, "
                    f"split: {split}, row: {row_idx})"
                )

                doc = Document(
                    id=doc_id,
                    bucket_id=bucket_id,
                    filename=filename,
                    source_format="txt",
                    source_note=source_note,
                    raw_uri=raw_uri,
                    text_uri=raw_uri,
                    status="parsed",
                    content_hash=cleaned.content_hash,
                    char_count=cleaned.char_count,
                    word_count=cleaned.word_count,
                    quality=cleaned.quality,
                )
                session.add(doc)
                session.flush()
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
    if imported == 0:
        raise RuntimeError(_nothing_kept(text_column, repeats > 0))

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


def _rows(url: str, headers: dict[str, str]) -> dict:
    for attempt in range(ROWS_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < ROWS_RETRIES:
                time.sleep(min(60, 2**attempt))
                continue
            raise RuntimeError(f"Hugging Face API returned error {err.code}: {err.reason}") from err
        except Exception as err:
            raise RuntimeError(
                f"Network failure while streaming rows from Hugging Face: {err}"
            ) from err
    raise RuntimeError("Hugging Face kept refusing requests. Try again later.")


def parquet_urls(canonical_id: str, config: str, split: str) -> list[str]:
    """The hub's Parquet copy of one split, as download links."""
    url = (
        f"{HF_DATASETS_SERVER}/parquet?dataset={urllib.parse.quote(canonical_id)}"
        f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
    )
    try:
        request = urllib.request.Request(url, headers=get_hf_headers())
        with urllib.request.urlopen(request, timeout=30) as response:
            listing = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise NoParquetCopy(canonical_id) from err
        raise RuntimeError(f"Hugging Face API returned error {err.code}: {err.reason}") from err
    except Exception as err:
        raise RuntimeError(f"Network failure while listing the Parquet files: {err}") from err
    files = [
        row["url"]
        for row in listing.get("parquet_files", [])
        if row.get("config") == config and row.get("split") == split and row.get("url")
    ]
    if not files:
        raise NoParquetCopy(canonical_id)
    return files


# tests swap these two out
def _hub_info(repo: str, revision: str):
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo, revision=revision)


def _hub_download(repo: str, revision: str, path: str, folder: Path) -> Path:
    from huggingface_hub import hf_hub_download

    found = hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=folder)
    return Path(found)


def _pattern(glob: str) -> re.Pattern[str]:
    # a card's data_files pattern: * and ? within a folder, ** across folders
    found, at = [], 0
    glob = glob.removeprefix("./")
    while at < len(glob):
        if glob.startswith("**/", at):
            found.append("(?:.*/)?")
            at += 3
        elif glob.startswith("**", at):
            found.append(".*")
            at += 2
        elif glob[at] in "*?":
            found.append("[^/]*" if glob[at] == "*" else "[^/]")
            at += 1
        else:
            found.append(re.escape(glob[at]))
            at += 1
    return re.compile("".join(found))


def _globs(data_files: object) -> list[tuple[str, list[str]]]:
    """(split, patterns) in any of the ways a card writes data_files. No split means train."""

    def patterns(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        return [one for one in value if isinstance(one, str)] if isinstance(value, list) else []

    if isinstance(data_files, dict):
        return [(str(split), patterns(value)) for split, value in data_files.items()]
    if not isinstance(data_files, list):
        return [("train", patterns(data_files))] if isinstance(data_files, str) else []
    found = [("train", patterns(data_files))] if patterns(data_files) else []
    for one in data_files:
        if isinstance(one, dict):
            found.append((str(one.get("split") or "train"), patterns(one.get("path"))))
    return found


def _by_name(files: list[str]) -> dict[str, list[str]]:
    # the first split word in a path decides; anything else is train
    splits: dict[str, list[str]] = {}
    for path in files:
        words = re.split(r"[-_./]", path.lower())
        split = next((SPLIT_WORDS[word] for word in words if word in SPLIT_WORDS), "train")
        splits.setdefault(split, []).append(path)
    return splits


def _layout(info) -> dict[str, dict[str, list[str]]]:
    """Config, then split, then its readable files in name order."""
    files = sorted(
        one.rfilename for one in info.siblings or [] if one.rfilename.lower().endswith(READABLE)
    )
    card = info.card_data
    configs = card.get("configs") if card is not None else None
    if not isinstance(configs, list) or not configs:
        return {"default": _by_name(files)} if files else {}
    found: dict[str, dict[str, list[str]]] = {}
    for config in configs:
        if not isinstance(config, dict):
            continue
        name = str(config.get("config_name") or "default")
        globs = _globs(config.get("data_files"))
        if not globs:
            found[name] = _by_name(files)
            continue
        splits = {}
        for split, patterns in globs:
            compiled = [_pattern(one) for one in patterns]
            matched = [path for path in files if any(p.fullmatch(path) for p in compiled)]
            if matched:
                splits[split] = matched
        found[name] = splits
    return found


def data_file_urls(canonical_id: str, revision: str, config: str, split: str) -> list[str]:
    """Links to the dataset's own Parquet or JSON lines files for one split, in name order."""
    from huggingface_hub.errors import GatedRepoError

    try:
        info = _hub_info(canonical_id, revision)
    except GatedRepoError as err:
        raise RuntimeError(
            "This dataset asks you to accept its terms first. Accept them on its Hugging Face "
            "page, sign in to Hugging Face on this computer, then try again."
        ) from err
    except Exception as err:
        raise RuntimeError(f"Could not list the dataset's files on Hugging Face: {err}") from err
    paths = _layout(info).get(config, {}).get(split, [])
    base = f"https://huggingface.co/datasets/{canonical_id}/resolve/{revision}"
    return [f"{base}/{urllib.parse.quote(path)}" for path in paths]


def fetch(url: str, path: Path) -> None:
    """Download one data file, trying a few times; a stopped download carries on from its part."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    hub = RESOLVE.fullmatch(url)
    problem: Exception | None = None
    for attempt in range(TRIES):
        try:
            if hub is None:
                download(url, path)
                return
            repo, revision, name = (urllib.parse.unquote(part) for part in hub.groups())
            # the hub keeps a half done file in this folder and resumes it
            os.replace(_hub_download(repo, revision, name, path.parent / "hub"), path)
            return
        except Exception as err:
            # a local "not found" only means there was no connection
            gone = isinstance(err, EntryNotFoundError | GatedRepoError | RepositoryNotFoundError)
            if gone and not isinstance(err, LocalEntryNotFoundError):
                raise RuntimeError(
                    "Hugging Face no longer gives one of this dataset's files. Check the dataset's "
                    "page, then start a new import."
                ) from err
            problem = err
        if attempt + 1 < TRIES:
            time.sleep(WAIT * 2**attempt)
    raise RuntimeError(
        f"Could not download a data file from Hugging Face after {TRIES} tries ({problem}). "
        "Check the internet connection, then retry this job."
    )


def _suffix(url: str) -> str:
    name = url.lower().split("?")[0]
    return next((one for one in sorted(READABLE, key=len, reverse=True) if name.endswith(one)), "")


def _values(path: Path, column: str, skip: int) -> Generator[list, None, None]:
    """The column's values in batches, after the first `skip` rows of the file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow import json as pa_json

    try:
        if path.suffix != ".parquet":
            options = pa_json.ReadOptions(block_size=1 << 24)
            with pa.input_stream(str(path), compression="detect") as stream:
                for batch in pa_json.open_json(stream, read_options=options):
                    index = batch.schema.get_field_index(column)
                    values = batch.column(index).to_pylist() if index >= 0 else [None] * len(batch)
                    yield values[skip:]
                    skip = max(0, skip - len(values))
            return
        with path.open("rb") as handle:
            data = pq.ParquetFile(handle)
            if column not in data.schema_arrow.names:
                raise RuntimeError(f"Column '{column}' is not in the dataset's data files.")
            first = 0
            while first < data.num_row_groups and data.metadata.row_group(first).num_rows <= skip:
                skip -= data.metadata.row_group(first).num_rows
                first += 1
            groups = list(range(first, data.num_row_groups))
            if not groups:
                return
            for batch in data.iter_batches(10_000, row_groups=groups, columns=[column]):
                values = batch.column(0).to_pylist()
                yield values[skip:]
                skip = max(0, skip - len(values))
    except (pa.ArrowException, OSError, ValueError) as err:
        raise RuntimeError(f"The data file {path.name} could not be read ({err}).") from err


def _row_text(value: object, use: str | None) -> tuple[str | None, str]:
    """A row's text, or None and "skipped" (empty) or "failed" (not text or not a conversation)."""
    if value is None:
        return None, "skipped"
    if use == "chat":
        if chatformat.problem(value) is not None:
            return None, "failed"
        return chatformat.render(value), ""  # type: ignore[arg-type]
    if not isinstance(value, str):
        return None, "failed"
    text = value.strip()
    return (text, "") if text else (None, "skipped")


def _nothing_kept(column: str, repeats: bool) -> str:
    if repeats:
        return "Every row read is already in this collection, so nothing new was added."
    return (
        f"Nothing usable was found in the column '{column}'. Pick another column, then try again."
    )


def download(url: str, path: Path) -> None:
    """Stream one file to disk, so a large file never sits in memory."""
    request = urllib.request.Request(url, headers=get_hf_headers())
    try:
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out:
            shutil.copyfileobj(response, out, DOWNLOAD_CHUNK)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"Hugging Face returned error {err.code} for a data file.") from err
    except OSError as err:
        raise RuntimeError(f"The download of a data file stopped: {err}") from err


def _import_from_parquet(
    job_id: uuid.UUID,
    bucket_id: uuid.UUID,
    canonical_id: str,
    revision: str,
    config: str,
    split: str,
    text_column: str,
    limit: int | None,
    total: int,
    urls: list[str],
    use: str | None = None,
    partial: bool = False,
) -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as problem:
        raise RuntimeError(
            "Importing from Hugging Face needs pyarrow. Install the prepare extra from the "
            "README first."
        ) from problem

    # a small import keeps one row per file; a big one groups rows evenly
    size = limit or total
    rows_per_file = max(1, -(-size // TARGET_FILES)) if size else PACK_ROWS
    folder = get_settings().data_dir / "imports" / str(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    total = total or limit or 0
    # a retry carries on from the place saved with the last file written
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        saved = dict(job.result) if job is not None else {}
    counts = {"processed": 0, "imported": 0, "skipped": 0, "failed": 0, **saved.get("counts", {})}
    cursor = saved.get("cursor") or {"file": 0, "row": 0}
    unusable = int(saved.get("unusable", 0))
    # repeats are kept out within a file; a whole repeated file by its content hash
    seen: set[bytes] = set()
    pack: list[str] = []
    chars = 0
    first_row = counts["processed"]
    parts = int(saved.get("parts", 0))

    def say(message: str) -> None:
        report(job_id, completed=counts["processed"], total=total, message=message, **counts)

    def full() -> bool:
        return limit is not None and counts["processed"] >= limit

    def take(value: object) -> None:
        nonlocal use, chars, unusable
        counts["processed"] += 1
        if use is None and value is not None:
            use = "chat" if isinstance(value, list) else "text"
        text, why = _row_text(value, use)
        if text is None:
            counts[why] += 1
            unusable += 1
        else:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if digest in seen:
                counts["skipped"] += 1
            else:
                seen.add(digest)
                pack.append(text)
                chars += len(text)
        if counts["processed"] == CHECK_ROWS and unusable * 2 > CHECK_ROWS:
            raise RuntimeError(
                f"Most of the first {CHECK_ROWS:,} values in the column '{text_column}' could "
                f"not be used ({unusable:,} were empty, not text or not a usable conversation). "
                "Pick another column, then try again."
            )

    def write_pack(file_index: int, row: int) -> None:
        """Save the pack as one document and the place to carry on from, in one commit."""
        nonlocal pack, first_row, parts, seen, chars
        last_row = counts["processed"] - 1
        joiner = f"{chatformat.EOS}\n" if use == "chat" else "\n\n"
        # imported rows get the light normalising only, never drop_repeated_lines
        cleaned = clean(normalise(joiner.join(pack)), mode="none")
        with acquire_bucket_lock(bucket_id, timeout=5), Session(get_engine()) as session:
            bucket = session.get(Bucket, bucket_id)
            if bucket is None:
                raise RuntimeError("The collection was deleted.")
            problem = kind_problem(session, bucket, use) if pack else None
            if problem:
                raise RuntimeError(problem)
            twin = session.execute(
                select(Document.id).where(
                    Document.bucket_id == bucket_id,
                    Document.content_hash == cleaned.content_hash,
                )
            ).first()
            if twin is None and cleaned.text.strip():
                # the same file gets the same name on a retry, so a file left by a stop is replaced
                doc_id = uuid.uuid5(job_id, f"{file_index}:{row}")
                uri = _write_file(f"text/{doc_id}.txt", cleaned.text.encode("utf-8"))
                doc = Document(
                    id=doc_id,
                    bucket_id=bucket_id,
                    filename=f"hf_{config}_{split}_part{parts:05d}.txt",
                    source_format="txt",
                    source_note=(
                        f"Imported from Hugging Face: {canonical_id} (revision: {revision[:8]}, "
                        f"config: {config}, split: {split}, rows {first_row} to {last_row}, "
                        f"{len(pack):,} rows in this file)"
                    ),
                    raw_uri=uri,
                    text_uri=uri,
                    status="parsed",
                    content_hash=cleaned.content_hash,
                    char_count=cleaned.char_count,
                    word_count=cleaned.word_count,
                    quality=cleaned.quality,
                )
                session.add(doc)
                bucket.kind = use or bucket.kind
                counts["imported"] += len(pack)
                parts += 1
            else:
                counts["skipped"] += len(pack)
            job = session.get(Job, job_id)
            if job is not None:
                job.result = {
                    "revision": revision,
                    "cursor": {"file": file_index, "row": row},
                    "counts": dict(counts),
                    "unusable": unusable,
                    "parts": parts,
                }
            session.commit()
        pack, seen, chars, first_row = [], set(), 0, counts["processed"]

    try:
        for index in range(cursor["file"], len(urls)):
            if full():
                break
            say(f"Downloading data file {index + 1} of {len(urls)}")
            path = folder / f"{index:05d}{_suffix(urls[index]) or '.parquet'}"
            if not path.is_file():
                fetch(urls[index], path)
            consumed = cursor["row"] if index == cursor["file"] else 0
            with contextlib.closing(_values(path, text_column, consumed)) as batches:
                for values in batches:
                    for value in values:
                        if full():
                            break
                        consumed += 1
                        take(value)
                        if len(pack) >= rows_per_file or chars >= PACK_CHARS:
                            write_pack(index, consumed)
                    say(
                        f"{counts['processed']:,} / {total:,} records ({counts['imported']:,} "
                        f"imported in {parts:,} files, {counts['skipped']:,} skipped, "
                        f"{counts['failed']:,} failed)"
                    )
                    if full():
                        break
            if full():
                write_pack(index, consumed)
            else:
                write_pack(index + 1, 0)
            path.unlink(missing_ok=True)
    except Canceled:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    # a failed import keeps its half downloaded file, so a retry carries on
    shutil.rmtree(folder, ignore_errors=True)

    if counts["processed"] == 0:
        raise RuntimeError(
            f"No records found in dataset '{canonical_id}' config '{config}' split '{split}'."
        )
    if counts["imported"] == 0:
        raise RuntimeError(_nothing_kept(text_column, unusable < counts["processed"]))
    message = (
        f"Import complete: {counts['imported']:,} rows imported in {parts:,} files, "
        f"{counts['skipped']:,} skipped, {counts['failed']:,} failed "
        f"({counts['processed']:,} processed)"
    )
    if partial:
        message += (
            ". Hugging Face only has a partial copy of this split, so only its first part was read."
        )
    say(message)
