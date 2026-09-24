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
