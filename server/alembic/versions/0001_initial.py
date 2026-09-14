"""buckets and documents

Revision ID: 0001
Revises:
Created: 2026-08-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "buckets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "bucket_id",
            sa.Uuid(),
            sa.ForeignKey("buckets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("source_format", sa.Text(), nullable=False),
        sa.Column("source_note", sa.Text(), nullable=False, server_default=""),
        sa.Column("raw_uri", sa.Text(), nullable=False),
        sa.Column("text_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("char_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("word_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("quality", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'parsed', 'failed', 'rejected')",
            name="ck_documents_status",
        ),
        sa.CheckConstraint(
            "source_format IN ('txt', 'md', 'html', 'pdf', 'jsonl', 'other')",
            name="ck_documents_source_format",
        ),
        sa.UniqueConstraint("bucket_id", "content_hash", name="uq_documents_bucket_hash"),
    )
    op.create_index("ix_documents_bucket_id", "documents", ["bucket_id"])
    op.create_index("ix_documents_status", "documents", ["status"])


def downgrade() -> None:
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_bucket_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("buckets")
