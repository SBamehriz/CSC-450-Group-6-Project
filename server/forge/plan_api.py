import json
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge import runs_api
from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.jobs_api import _tools
from forge.measuring import pick_checkpoint
from forge.models import Bucket, Dataset, DatasetBuild, Document, Job, Model, Run, Tokenizer
from forge.schemas import JobPayload, JobProgress, PlanOut, PlanRequest, RunOut, TrainCreate
from forge.train import plan as planner
from forge.train.presets import MAX_VOCAB, ModelConfig, count_params, validate

router = APIRouter(tags=["training"])

MIN_VOCAB = 300
# the snapshot settings the Data page uses by default
VALIDATION_FRACTION = 0.1
SNAPSHOT_SEED = 42


@router.post("/plan", response_model=PlanOut)
def plan(payload: PlanRequest, session: Session = Depends(get_session)) -> PlanOut:
    return planner.plan(session, payload)


@router.post("/train", response_model=RunOut, status_code=201)
def train(payload: TrainCreate, session: Session = Depends(get_session)) -> RunOut:
    """Makes the model, gets the text ready if needed, and starts training."""
    return runs_api._run_to_out(start_training(session, payload), session)


def start_training(session: Session, payload: TrainCreate) -> Run:
    """The model row, then jobs for whatever is missing, then the training job."""
    name = payload.name.strip()
    bucket = session.get(Bucket, payload.bucket_id)
    if bucket is None:
        raise ApiError(404, "not_found", "That collection no longer exists.")
    if not name or session.scalar(select(Model.id).where(Model.name == name)):
        raise ApiError(
            409, "name_taken", f"A model called {name!r} already exists. Choose another name."
        )
    _check_text(session, bucket)
    if payload.base_model_id:
        base, start, vocabulary = _base(session, payload.base_model_id)
        model = Model(
            name=name,
            description=f"Fine-tuned from {base.name} on {bucket.name}.",
            n_layer=base.n_layer,
            d_model=base.d_model,
            n_head=base.n_head,
            ctx_len=base.ctx_len,
            vocab_size=base.vocab_size,
            dropout=payload.dropout,
            param_count=base.param_count,
            parent_id=base.id,
            chat=bucket.kind == "chat",
        )
    else:
        config = _config(payload)
        start, vocabulary = None, None
        model = Model(
            name=name,
            n_layer=config.n_layer,
            d_model=config.d_model,
            n_head=config.n_head,
            ctx_len=config.ctx_len,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            param_count=count_params(config),
        )
    _tools()

    # the queue runs jobs one at a time, oldest first, so each finds what it needs
    queued: list[Job] = []
    first = datetime.now(UTC) - timedelta(milliseconds=1)

    def line_up(kind: str, target, extra: dict | None = None) -> None:
        job = Job(
            job_type=kind,
            payload=JobPayload(target_id=target.id, target_name=target.name).model_dump(mode="json")
            | (extra or {}),
            progress=JobProgress().model_dump(),
            created_at=first + timedelta(microseconds=len(queued)),
        )
        session.add(job)
        queued.append(job)

    snapshot = _ready_snapshot(session, bucket)
    if snapshot is None:
        snapshot = Dataset(
            name=_free_name(session, Dataset, f"{name} text"),
            bucket_id=bucket.id,
            status="processing",
            config={
                "seed": SNAPSHOT_SEED,
                "validation_fraction": VALIDATION_FRACTION,
                "bucket_name": bucket.name,
            },
        )
        session.add(snapshot)
        session.flush()
        line_up(
            "snapshot_create",
            snapshot,
            {
                "bucket_id": str(bucket.id),
                "validation_fraction": VALIDATION_FRACTION,
                "seed": SNAPSHOT_SEED,
            },
        )
    if vocabulary is None:
        vocabulary = session.scalar(
            select(Tokenizer)
            .where(
                Tokenizer.dataset_id == snapshot.id,
                Tokenizer.status == "ready",
                Tokenizer.requested_vocab_size == model.vocab_size,
            )
            .order_by(Tokenizer.created_at.desc())
        )
    if vocabulary is None:
        vocabulary = Tokenizer(
            name=_free_name(session, Tokenizer, f"{name} vocabulary"),
            dataset_id=snapshot.id,
            dataset_name=snapshot.name,
            requested_vocab_size=model.vocab_size,
        )
        session.add(vocabulary)
        session.flush()
        line_up("tokenizer_train", vocabulary)
    build = session.scalar(
        select(DatasetBuild)
        .where(
            DatasetBuild.dataset_id == snapshot.id,
            DatasetBuild.tokenizer_id == vocabulary.id,
            DatasetBuild.status == "ready",
        )
        .order_by(DatasetBuild.created_at.desc())
    )
    if build is None:
        build = DatasetBuild(
            name=_free_name(session, DatasetBuild, f"{name} training set"),
            tokenizer_id=vocabulary.id,
            dataset_id=snapshot.id,
            dataset_name=snapshot.name,
        )
        session.add(build)
        session.flush()
        line_up("dataset_build", build)

    model.tokenizer_id = vocabulary.id
    session.add(model)
    session.flush()
    settings = payload.settings
    hyperparams: dict = {
        "settings": settings.model_dump(),
        # to tell what the run is waiting on
        "prepare_jobs": [str(job.id) for job in queued],
    }
    if payload.base_model_id and bucket.kind == "chat":
        # the trainer learns only the assistant's words
        hyperparams["chat"] = True
    run = Run(
        name=name,
        model_id=model.id,
        dataset_id=snapshot.id,
        build_id=build.id,
        init_checkpoint_id=start,
        seed=payload.seed,
        device=payload.device,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        max_steps=settings.max_steps,
        status="pending",
        hyperparams=hyperparams,
    )
    session.add(run)
    session.commit()
    runs_api._queue(session, run, resume=False)
    return run


def _config(payload: TrainCreate) -> ModelConfig:
    if payload.shape is None:
        raise ApiError(422, "shape_missing", "Give the shape of the new model.")
    try:
        config = ModelConfig(**payload.shape.model_dump(), dropout=payload.dropout)
        validate(config)
    except ValueError as problem:
        raise ApiError(422, "invalid_shape", str(problem)) from problem
    if not MIN_VOCAB <= config.vocab_size <= MAX_VOCAB:
        raise ApiError(
            422, "invalid_shape", f"A vocabulary has {MIN_VOCAB} to {MAX_VOCAB:,} word pieces."
        )
    return config


def _check_text(session: Session, bucket: Bucket) -> None:
    # the snapshot job would stop on these, so say it now
    counts: dict[str, int] = {}
    for status, count in session.execute(
        select(Document.status, func.count())
        .where(Document.bucket_id == bucket.id)
        .group_by(Document.status)
    ):
        counts[status] = count
    if counts.get("pending"):
        raise ApiError(
            409, "not_ready", "Some files are still being read. Try again once they finish."
        )
    if counts.get("parsed", 0) < 2:
        raise ApiError(
            409, "no_text", f"{bucket.name} needs at least two files with text. Add some first."
        )


def _base(session: Session, model_id: uuid.UUID) -> tuple[Model, uuid.UUID, Tokenizer]:
    """The model to fine-tune, the save point to start from, and its vocabulary."""
    base = session.get(Model, model_id)
    if base is None:
        raise ApiError(404, "not_found", "The model to fine-tune no longer exists.")
    if base.source != "forge":
        raise ApiError(
            409,
            "hub_base",
            f"{base.name} came from the Hugging Face hub. Fine-tuning starts from a model "
            "trained in Forge.",
        )
    start = pick_checkpoint(session, base.id)
    vocabulary_id = base.tokenizer_id
    if vocabulary_id is None and start is not None:
        # a model made before it was given a vocabulary: the one it was trained with
        trained = session.get(Run, start.run_id)
        build = (
            session.get(DatasetBuild, trained.build_id) if trained and trained.build_id else None
        )
        vocabulary_id = build.tokenizer_id if build else None
    vocabulary = session.get(Tokenizer, vocabulary_id) if vocabulary_id else None
    if start is None or vocabulary is None or vocabulary.status != "ready":
        raise ApiError(409, "base_untrained", f"{base.name} has no save point yet. Train it first.")
    return base, start.id, vocabulary


def _ready_snapshot(session: Session, bucket: Bucket) -> Dataset | None:
    """The newest ready snapshot, if the collection's text has not changed since."""
    newest = session.scalar(
        select(Dataset)
        .where(Dataset.bucket_id == bucket.id, Dataset.status == "ready")
        .order_by(Dataset.created_at.desc())
    )
    if newest is None:
        return None
    path = get_settings().data_dir / "datasets" / str(newest.id) / "manifest.json"
    try:
        listed = json.loads(path.read_text(encoding="utf-8"))["documents"]
        taken = {(row["id"], row["content_hash"]) for row in listed}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = {
        (str(doc_id), digest)
        for doc_id, digest in session.execute(
            select(Document.id, Document.content_hash).where(
                Document.bucket_id == bucket.id, Document.status == "parsed"
            )
        )
    }
    return newest if taken == now else None


def _free_name(session: Session, table, wanted: str) -> str:
    name, number = wanted, 1
    while session.scalar(select(table.id).where(table.name == name)) is not None:
        number += 1
        name = f"{wanted} {number}"
    return name
