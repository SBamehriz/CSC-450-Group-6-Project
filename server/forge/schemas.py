import uuid
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int


class BucketStats(BaseModel):
    documents: int
    parsed_documents: int
    chars: int
    est_tokens: int


class BucketCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("cannot be blank")
        return value


class BucketUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("cannot be blank")
        return value


class BucketOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    created_at: datetime
    stats: BucketStats


class DocumentOut(BaseModel):
    id: uuid.UUID
    bucket_id: uuid.UUID
    filename: str
    source_format: str
    source_note: str
    status: str
    error: str | None
    content_hash: str | None
    char_count: int
    word_count: int
    quality: dict
    created_at: datetime


class UploadOutcome(BaseModel):
    """One line per file the user handed us, in the order they sent them."""

    filename: str
    status: str  # parsed | failed
    document_id: uuid.UUID | None = None
    error: str | None = None


class UploadResult(BaseModel):
    parsed: int
    failed: int
    outcomes: list[UploadOutcome]


class DocumentText(BaseModel):
    """A window into the cleaned text, never the whole 50MB document."""

    text: str
    offset: int
    length: int
    total: int


class MoveDocument(BaseModel):
    bucket_id: uuid.UUID


class RejectDocument(BaseModel):
    reason: str = Field(default="", max_length=500)

    @field_validator("reason")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class Health(BaseModel):
    status: str
    database: str


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)
    bucket_id: uuid.UUID | None = None
    config: dict = Field(default_factory=dict)


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    bucket_id: uuid.UUID | None
    status: str
    doc_count: int
    char_count: int
    token_count: int
    config: dict
    artifact_uri: str | None
    created_at: datetime


class JobCreate(BaseModel):
    job_type: str
    payload: dict = Field(default_factory=dict)


class JobOut(BaseModel):
    id: uuid.UUID
    job_type: str
    status: str
    payload: dict
    progress: dict
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)
    preset: str | None = None
    n_layer: int = Field(ge=1, le=12)
    d_model: int = Field(ge=1, le=768)
    n_head: int = Field(ge=1)
    ctx_len: int = Field(ge=128, le=2048)
    vocab_size: int = 16384
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    config: dict = Field(default_factory=dict)


class ModelOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    preset: str | None
    n_layer: int
    d_model: int
    n_head: int
    ctx_len: int
    vocab_size: int
    dropout: float
    param_count: int
    config: dict
    created_at: datetime


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    model_id: uuid.UUID
    dataset_id: uuid.UUID | None = None
    device: str = "cpu"
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    max_steps: int = Field(default=1000, ge=1)
    hyperparams: dict = Field(default_factory=dict)


class RunOut(BaseModel):
    id: uuid.UUID
    name: str
    model_id: uuid.UUID
    dataset_id: uuid.UUID | None
    status: str
    device: str
    batch_size: int
    learning_rate: float
    max_steps: int
    current_step: int
    loss: float | None
    eval_loss: float | None
    tokens_per_sec: float | None
    metrics: dict
    hyperparams: dict
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class CheckpointOut(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    step: int
    epoch: int
    loss: float | None
    val_loss: float | None
    uri: str
    size_bytes: int
    is_best: bool
    metrics: dict
    created_at: datetime

