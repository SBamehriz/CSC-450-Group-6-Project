"""Safe, deduplicated file accounting and bulk deletion within Forge storage."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from filelock import FileLock, Timeout

from forge.config import get_settings
from forge.errors import ApiError


class InvalidStoragePathError(ValueError):
    """Raised when a relative storage URI resolves outside the managed files directory."""


class BucketBusyError(Exception):
    """Raised when a collection lock cannot be acquired immediately."""


@dataclass(frozen=True)
class FileAccounting:
    """Physical storage summary after deduplicating paths and excluding shared files."""

    raw_file_count: int
    text_file_count: int
    physical_file_count: int
    raw_bytes: int
    text_bytes: int
    total_bytes: int
    shared_file_count: int
    missing_file_count: int
    owned_raw_paths: tuple[tuple[str, Path], ...]
    owned_text_paths: tuple[tuple[str, Path], ...]
    file_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class FileDeleteFailure:
    """Describes a single file that could not be removed from disk."""

    uri: str
    error: str


@dataclass(frozen=True)
class FileDeleteResult:
    """Outcome summary returned after attempting to unlink storage files."""

    deleted_count: int
    missing_count: int
    failed: tuple[FileDeleteFailure, ...]


def files_root() -> Path:
    """Return the resolved directory under which document source and text files live."""
    return (get_settings().data_dir / "files").resolve()


def resolve_storage_file(relative: str | None) -> Path:
    """Resolve a relative storage URI inside ``files_root()`` and reject unsafe escapes."""
    if not relative or not relative.strip():
        raise InvalidStoragePathError("Storage path cannot be empty.")
    if "\x00" in relative:
        raise InvalidStoragePathError("Storage path contains invalid characters.")
    if relative.startswith(("/", "\\")):
        raise InvalidStoragePathError(f"Storage path must be relative: {relative!r}")

    normalized = relative.replace("\\", "/")
    win = PureWindowsPath(relative)
    if win.drive or win.root or PurePosixPath(normalized).root:
        raise InvalidStoragePathError(f"Storage path must not be anchored: {relative!r}")
    if ".." in PurePosixPath(normalized).parts:
        raise InvalidStoragePathError(f"Storage path must not traverse directories: {relative!r}")

    root = files_root()
    raw_target = root / normalized
    if raw_target.is_symlink():
        raise InvalidStoragePathError(f"Refusing symlink storage path: {relative!r}")

    resolved = raw_target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise InvalidStoragePathError(f"Storage path escapes files directory: {relative!r}")
    if resolved.exists() and resolved.is_dir():
        raise InvalidStoragePathError(
            f"Storage path must refer to a file, not a directory: {relative!r}"
        )
    return resolved


def account_storage_files(
    raw_uris: Iterable[str | None],
    text_uris: Iterable[str | None],
    *,
    preserved_uris: Iterable[str | None] = (),
) -> FileAccounting:
    """Compute deduplicated file counts and removable byte sizes for storage URIs."""
    preserved_paths: set[Path] = set()
    for uri in preserved_uris:
        if not uri:
            continue
        try:
            preserved_paths.add(resolve_storage_file(uri))
        except InvalidStoragePathError:
            # Malformed external URIs cannot match validated target files, so skip safely.
            continue

    shared_paths: set[Path] = set()
    seen_raw: set[Path] = set()
    seen_text: set[Path] = set()
    owned_raw: list[tuple[str, Path]] = []
    owned_text: list[tuple[str, Path]] = []

    for uri in raw_uris:
        if not uri:
            continue
        resolved = resolve_storage_file(uri)
        if resolved in preserved_paths:
            shared_paths.add(resolved)
            continue
        if resolved not in seen_raw:
            seen_raw.add(resolved)
            owned_raw.append((uri, resolved))

    for uri in text_uris:
        if not uri:
            continue
        resolved = resolve_storage_file(uri)
        if resolved in preserved_paths:
            shared_paths.add(resolved)
            continue
        if resolved not in seen_text:
            seen_text.add(resolved)
            owned_text.append((uri, resolved))

    missing_paths: set[Path] = set()
    counted_physical: set[Path] = set()
    raw_file_count = 0
    text_file_count = 0
    raw_bytes = 0
    text_bytes = 0
    total_bytes = 0
    fingerprints: list[str] = []

    def _stat_group(entries: list[tuple[str, Path]], *, kind: str) -> tuple[int, int]:
        nonlocal total_bytes
        count = 0
        subtotal = 0
        for uri, path in entries:
            try:
                st = path.stat()
            except FileNotFoundError:
                missing_paths.add(path)
                fingerprints.append(f"{kind}:{uri}:missing")
                continue
            count += 1
            subtotal += st.st_size
            fingerprints.append(f"{kind}:{uri}:{st.st_size}:{st.st_mtime_ns}")
            if path not in counted_physical:
                counted_physical.add(path)
                total_bytes += st.st_size
        return count, subtotal

    raw_file_count, raw_bytes = _stat_group(owned_raw, kind="raw")
    text_file_count, text_bytes = _stat_group(owned_text, kind="text")
    fingerprints.extend(f"shared:{path.as_posix()}" for path in sorted(shared_paths))
    fingerprints.sort()

    return FileAccounting(
        raw_file_count=raw_file_count,
        text_file_count=text_file_count,
        physical_file_count=len(counted_physical),
        raw_bytes=raw_bytes,
        text_bytes=text_bytes,
        total_bytes=total_bytes,
        shared_file_count=len(shared_paths),
        missing_file_count=len(missing_paths),
        owned_raw_paths=tuple(owned_raw),
        owned_text_paths=tuple(owned_text),
        file_fingerprints=tuple(fingerprints),
    )


def delete_storage_files(entries: Iterable[tuple[str, Path]]) -> FileDeleteResult:
    """Safely delete deduplicated files inside ``files_root()`` and collect any failures."""
    root = files_root()
    seen: set[Path] = set()
    deleted_count = 0
    missing_count = 0
    failed: list[FileDeleteFailure] = []

    for uri, path in entries:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved == root or not resolved.is_relative_to(root):
            raise InvalidStoragePathError(f"Storage path escapes files directory: {uri!r}")
        if resolved.exists() and resolved.is_dir():
            raise InvalidStoragePathError(f"Refusing to remove directory: {uri!r}")
        try:
            resolved.unlink()
            deleted_count += 1
        except FileNotFoundError:
            missing_count += 1
        except OSError as exc:
            failed.append(FileDeleteFailure(uri=uri, error=str(exc)))

    return FileDeleteResult(
        deleted_count=deleted_count,
        missing_count=missing_count,
        failed=tuple(failed),
    )


def bucket_lock_path(bucket_id: uuid.UUID) -> Path:
    """Return the lock file path protecting a collection against conflicting mutations."""
    lock_dir = get_settings().data_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"bucket-{bucket_id}.lock"


@contextmanager
def acquire_bucket_lock(bucket_id: uuid.UUID, *, timeout: float = 0) -> Iterator[None]:
    """Acquire an exclusive per-collection file lock or raise BucketBusyError."""
    lock = FileLock(bucket_lock_path(bucket_id))
    try:
        with lock.acquire(timeout=timeout):
            yield
    except Timeout as problem:
        raise BucketBusyError(str(bucket_id)) from problem


@contextmanager
def require_bucket_lock(
    bucket_id: uuid.UUID,
    *,
    timeout: float = 0,
    message: str = (
        "This collection is currently locked by another operation or deletion. "
        "Wait for it to finish and try again."
    ),
) -> Iterator[None]:
    """Acquire a single collection lock for an HTTP handler or raise 409 bucket_busy."""
    try:
        with acquire_bucket_lock(bucket_id, timeout=timeout):
            yield
    except BucketBusyError as problem:
        raise ApiError(409, "bucket_busy", message) from problem
