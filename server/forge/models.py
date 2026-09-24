import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


class UtcDateTime(TypeDecorator):
    # sqlite drops the timezone
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        UtcDateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


DOCUMENT_STATUSES = ("pending", "parsed", "failed", "rejected")
SOURCE_FORMATS = ("txt", "md", "html", "pdf", "docx", "image", "jsonl", "other")
DATASET_STATUSES = ("draft", "processing", "ready", "failed")
JOB_STATUSES = ("pending", "running", "completed", "failed", "cancelled")
JOB_TYPES = ("tokenization", "training", "eval", "ingest", "export")
RUN_STATUSES = ("pending", "running", "completed", "failed", "stopped")
MODEL_PRESETS = ("nano", "micro", "small", "custom")


def _sql_in(column: str, allowed: tuple[str, ...]) -> str:
    values = ", ".join(f"'{value}'" for value in allowed)
    return f"{column} IN ({values})"


class Bucket(Base):
    __tablename__ = "buckets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = _created_at()


class Document(Base):
    # one file, or one jsonl line
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    bucket_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("buckets.id", ondelete="RESTRICT"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_format: Mapped[str] = mapped_column(Text, nullable=False)
    source_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_uri: Mapped[str] = mapped_column(Text, nullable=False)
    text_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    word_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    quality: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_sql_in("status", DOCUMENT_STATUSES), name="ck_documents_status"),
        CheckConstraint(
            _sql_in("source_format", SOURCE_FORMATS), name="ck_documents_source_format"
        ),
        UniqueConstraint("bucket_id", "content_hash", name="uq_documents_bucket_hash"),
        Index("ix_documents_bucket_id", "bucket_id"),
        Index("ix_documents_status", "status"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    bucket_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("buckets.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_sql_in("status", DATASET_STATUSES), name="ck_datasets_status"),
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_bucket_id", "bucket_id"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    progress: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_sql_in("job_type", JOB_TYPES), name="ck_jobs_type"),
        CheckConstraint(_sql_in("status", JOB_STATUSES), name="ck_jobs_status"),
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_type", "job_type"),
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    preset: Mapped[str | None] = mapped_column(Text, nullable=True)
    n_layer: Mapped[int] = mapped_column(Integer, nullable=False)
    d_model: Mapped[int] = mapped_column(Integer, nullable=False)
    n_head: Mapped[int] = mapped_column(Integer, nullable=False)
    ctx_len: Mapped[int] = mapped_column(Integer, nullable=False)
    vocab_size: Mapped[int] = mapped_column(Integer, nullable=False, default=16384)
    dropout: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    param_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_models_preset", "preset"),
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    device: Mapped[str] = mapped_column(Text, nullable=False, default="cpu")
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.001)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_sql_in("status", RUN_STATUSES), name="ck_runs_status"),
        Index("ix_runs_model_id", "model_id"),
        Index("ix_runs_dataset_id", "dataset_id"),
        Index("ix_runs_status", "status"),
    )


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    is_best: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        UniqueConstraint("run_id", "step", name="uq_checkpoints_run_step"),
        Index("ix_checkpoints_run_id", "run_id"),
        Index("ix_checkpoints_is_best", "is_best"),
    )

