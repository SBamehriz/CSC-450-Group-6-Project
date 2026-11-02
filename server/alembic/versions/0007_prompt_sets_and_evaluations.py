"""add prompt_sets and evaluations tables

Revision ID: 0007
Revises: 0006
Created: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prompt_sets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("prompts", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_prompt_sets_name", "prompt_sets", ["name"])

    op.create_table(
        "evaluations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "model_id",
            sa.Uuid(),
            sa.ForeignKey("models.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column(
            "prompt_set_id",
            sa.Uuid(),
            sa.ForeignKey("prompt_sets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("prompt_set_name", sa.Text(), nullable=True),
        sa.Column("captured_prompts", sa.JSON(), nullable=False),
        sa.Column("results", sa.JSON(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_evaluations_status",
        ),
    )
    op.create_index("ix_evaluations_model_id", "evaluations", ["model_id"])
    op.create_index("ix_evaluations_prompt_set_id", "evaluations", ["prompt_set_id"])
    op.create_index("ix_evaluations_status", "evaluations", ["status"])


def downgrade() -> None:
    op.drop_index("ix_evaluations_status", table_name="evaluations")
    op.drop_index("ix_evaluations_prompt_set_id", table_name="evaluations")
    op.drop_index("ix_evaluations_model_id", table_name="evaluations")
    op.drop_table("evaluations")
    op.drop_index("ix_prompt_sets_name", table_name="prompt_sets")
    op.drop_table("prompt_sets")
