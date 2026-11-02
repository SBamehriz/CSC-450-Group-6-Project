"""add bpb_evaluations table

Revision ID: 0008
Revises: 0007
Created: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bpb_evaluations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "model_id",
            sa.Uuid(),
            sa.ForeignKey("models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_source", sa.Text(), nullable=False, server_default="native"),
        sa.Column(
            "checkpoint_id",
            sa.Uuid(),
            sa.ForeignKey("checkpoints.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "dataset_id",
            sa.Uuid(),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("validation_file_hash", sa.Text(), nullable=False),
        sa.Column("tokenizer_identity", sa.Text(), nullable=False),
        sa.Column("bpb", sa.Float(), nullable=True),
        sa.Column("total_nll", sa.Float(), nullable=True),
        sa.Column("scored_bytes", sa.BigInteger(), nullable=True),
        sa.Column("scored_tokens", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_bpb_evaluations_status",
        ),
    )
    op.create_index("ix_bpb_evaluations_model_id", "bpb_evaluations", ["model_id"])
    op.create_index("ix_bpb_evaluations_dataset_id", "bpb_evaluations", ["dataset_id"])
    op.create_index("ix_bpb_evaluations_status", "bpb_evaluations", ["status"])


def downgrade() -> None:
    op.drop_index("ix_bpb_evaluations_status", table_name="bpb_evaluations")
    op.drop_index("ix_bpb_evaluations_dataset_id", table_name="bpb_evaluations")
    op.drop_index("ix_bpb_evaluations_model_id", table_name="bpb_evaluations")
    op.drop_table("bpb_evaluations")
