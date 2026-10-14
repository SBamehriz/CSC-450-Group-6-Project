import threading
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from forge.datasets import _load_one
from forge.db import get_session
from forge.errors import ApiError
from forge.models import Job
from forge.preparation import build_prepared, prepared_path, read_prepared, tools_available
from forge.schemas import PrepareCreate, PreparedOut, PrepareTools, TokenPreview, TokenPreviewCreate
from forge.snapshots import read_snapshot

router = APIRouter(tags=["preparation"])
_prepare_lock = threading.Lock()


@router.get("/preparation/tools", response_model=PrepareTools)
def preparation_tools() -> PrepareTools:
    return PrepareTools(available=tools_available(), max_text_mb=10)


@router.get("/datasets/{dataset_id}/prepared", response_model=PreparedOut)
def get_prepared(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> PreparedOut:
    dataset = _load_one(session, dataset_id)
    stats = dataset.config.get("prepared", {})
    if not stats:
        return PreparedOut(ready=False)
    try:
        read_prepared(dataset)
    except ApiError as problem:
        return PreparedOut(**{**stats, "ready": False, "message": problem.message})
    return PreparedOut(**stats)


@router.post("/datasets/{dataset_id}/prepare", response_model=PreparedOut, status_code=201)
def prepare_dataset(
    dataset_id: uuid.UUID, payload: PrepareCreate, session: Session = Depends(get_session)
) -> PreparedOut:
    if not tools_available():
        raise ApiError(
            503,
            "tools_missing",
            "Tokenizer tools are not installed. Add the prepare extra from the README.",
        )
    if not _prepare_lock.acquire(blocking=False):
        raise ApiError(
            409, "busy", "Another snapshot is being prepared. Try again when it finishes."
        )
    job = None
    dataset = None
    saved = False
    try:
        dataset = _load_one(session, dataset_id)
        if dataset.config.get("prepared", {}).get("ready"):
            raise ApiError(
                409,
                "already_prepared",
                "This snapshot already has prepared files. Remove them before preparing again.",
            )
        read_snapshot(dataset)
        job = Job(
            job_type="tokenization",
            status="running",
            payload={"dataset_id": str(dataset_id), "vocab_size": payload.vocab_size},
            started_at=datetime.now(UTC),
        )
        session.add(job)
        session.commit()
        stats = build_prepared(dataset, payload.vocab_size)
        saved = True
        dataset.config = {**dataset.config, "prepared": stats}
        dataset.token_count = stats["train_tokens"] + stats["validation_tokens"]
        job.status = "completed"
        job.progress = {"documents": dataset.doc_count, "tokens": dataset.token_count}
        job.completed_at = datetime.now(UTC)
        session.commit()
        return PreparedOut(**stats)
    except Exception as problem:
        session.rollback()
        if saved and dataset is not None:
            prepared_path(dataset).unlink(missing_ok=True)
        if job is not None:
            job.status = "failed"
            job.error = (
                problem.message
                if isinstance(problem, ApiError)
                else "Could not prepare the snapshot. Try again."
            )
            job.completed_at = datetime.now(UTC)
            session.add(job)
            session.commit()
        if isinstance(problem, ApiError):
            raise
        raise ApiError(
            503, "prepare_failed", "Could not prepare the snapshot. Try again."
        ) from problem
    finally:
        _prepare_lock.release()


@router.get("/datasets/{dataset_id}/prepared/download", response_class=FileResponse)
def download_prepared(
    dataset_id: uuid.UUID, session: Session = Depends(get_session)
) -> FileResponse:
    dataset = _load_one(session, dataset_id)
    read_prepared(dataset)
    return FileResponse(
        prepared_path(dataset), media_type="application/zip", filename=f"prepared-{dataset.id}.zip"
    )


@router.post("/datasets/{dataset_id}/prepared/preview", response_model=TokenPreview)
def preview_tokens(
    dataset_id: uuid.UUID, payload: TokenPreviewCreate, session: Session = Depends(get_session)
) -> TokenPreview:
    if not tools_available():
        raise ApiError(503, "tools_missing", "Tokenizer tools are not installed.")
    from tokenizers import Tokenizer

    _manifest, files = read_prepared(_load_one(session, dataset_id))
    tokenizer = Tokenizer.from_str(files["tokenizer.json"].decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoded = tokenizer.encode(payload.text, add_special_tokens=False)
    return TokenPreview(token_count=len(encoded.ids), ids=encoded.ids, pieces=encoded.tokens)


@router.delete("/datasets/{dataset_id}/prepared", status_code=204)
def delete_prepared(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    if not _prepare_lock.acquire(blocking=False):
        raise ApiError(409, "busy", "Wait for the current preparation to finish.")
    try:
        dataset = _load_one(session, dataset_id)
        dataset.config = {key: value for key, value in dataset.config.items() if key != "prepared"}
        dataset.token_count = 0
        session.commit()
        prepared_path(dataset).unlink(missing_ok=True)
        return Response(status_code=204)
    finally:
        _prepare_lock.release()
