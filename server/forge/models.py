import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
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
