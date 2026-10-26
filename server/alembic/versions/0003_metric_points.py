"""add metric_points table

Revision ID: 0003
Revises: 0002
Created: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metric_points",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("speed", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "step", name="uq_metric_points_run_step"),
    )
    op.create_index("ix_metric_points_run_id", "metric_points", ["run_id"])
    op.create_index("ix_metric_points_run_step", "metric_points", ["run_id", "step"])


def downgrade() -> None:
    op.drop_index("ix_metric_points_run_step", table_name="metric_points")
    op.drop_index("ix_metric_points_run_id", table_name="metric_points")
    op.drop_table("metric_points")
