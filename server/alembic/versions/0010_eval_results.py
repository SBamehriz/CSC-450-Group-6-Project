"""add eval_results table

Revision ID: 0010
Revises: 0009
Created: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eval_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("result_type", sa.Text(), nullable=False),
        sa.Column(
            "model_id",
            sa.Uuid(),
            sa.ForeignKey("models.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column(
            "checkpoint_id",
            sa.Uuid(),
            sa.ForeignKey("checkpoints.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("evaluation_id", sa.Uuid(), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("code_version", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("bpb", sa.Float(), nullable=True),
        sa.Column("total_nll", sa.Float(), nullable=True),
        sa.Column("scored_bytes", sa.BigInteger(), nullable=True),
        sa.Column("scored_tokens", sa.BigInteger(), nullable=True),
        sa.Column("benchmark_name", sa.Text(), nullable=True),
        sa.Column("benchmark_score", sa.Float(), nullable=True),
        sa.Column("benchmark_scores", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "result_type IN ('prompt', 'bpb', 'benchmark')",
            name="ck_eval_results_type",
        ),
    )
    op.create_index("ix_eval_results_model_id", "eval_results", ["model_id"])
    op.create_index("ix_eval_results_evaluation_id", "eval_results", ["evaluation_id"])
    op.create_index("ix_eval_results_result_type", "eval_results", ["result_type"])
    op.create_index("ix_eval_results_created_at", "eval_results", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_eval_results_created_at", table_name="eval_results")
    op.drop_index("ix_eval_results_result_type", table_name="eval_results")
    op.drop_index("ix_eval_results_evaluation_id", table_name="eval_results")
    op.drop_index("ix_eval_results_model_id", table_name="eval_results")
    op.drop_table("eval_results")
