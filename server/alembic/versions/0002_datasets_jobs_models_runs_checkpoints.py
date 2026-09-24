"""datasets, jobs, models, runs, and checkpoints

Revision ID: 0002
Revises: 0001
Created: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "bucket_id",
            sa.Uuid(),
            sa.ForeignKey("buckets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("token_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('draft', 'processing', 'ready', 'failed')",
            name="ck_datasets_status",
        ),
    )
    op.create_index("ix_datasets_status", "datasets", ["status"])
    op.create_index("ix_datasets_bucket_id", "datasets", ["bucket_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("progress", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export')",
            name="ck_jobs_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_type", "jobs", ["job_type"])

    op.create_table(
        "models",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("preset", sa.Text(), nullable=True),
        sa.Column("n_layer", sa.Integer(), nullable=False),
        sa.Column("d_model", sa.Integer(), nullable=False),
        sa.Column("n_head", sa.Integer(), nullable=False),
        sa.Column("ctx_len", sa.Integer(), nullable=False),
        sa.Column("vocab_size", sa.Integer(), nullable=False, server_default="16384"),
        sa.Column("dropout", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("param_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_models_preset", "models", ["preset"])

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "model_id",
            sa.Uuid(),
            sa.ForeignKey("models.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "dataset_id",
            sa.Uuid(),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("device", sa.Text(), nullable=False, server_default="cpu"),
        sa.Column("batch_size", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("learning_rate", sa.Float(), nullable=False, server_default="0.001"),
        sa.Column("max_steps", sa.Integer(), nullable=False, server_default="1000"),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("eval_loss", sa.Float(), nullable=True),
        sa.Column("tokens_per_sec", sa.Float(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("hyperparams", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'stopped')",
            name="ck_runs_status",
        ),
    )
    op.create_index("ix_runs_model_id", "runs", ["model_id"])
    op.create_index("ix_runs_dataset_id", "runs", ["dataset_id"])
    op.create_index("ix_runs_status", "runs", ["status"])

    op.create_table(
        "checkpoints",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("is_best", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "step", name="uq_checkpoints_run_step"),
    )
    op.create_index("ix_checkpoints_run_id", "checkpoints", ["run_id"])
    op.create_index("ix_checkpoints_is_best", "checkpoints", ["is_best"])


def downgrade() -> None:
    op.drop_index("ix_checkpoints_is_best", table_name="checkpoints")
    op.drop_index("ix_checkpoints_run_id", table_name="checkpoints")
    op.drop_table("checkpoints")

    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_dataset_id", table_name="runs")
    op.drop_index("ix_runs_model_id", table_name="runs")
    op.drop_table("runs")

    op.drop_index("ix_models_preset", table_name="models")
    op.drop_table("models")

    op.drop_index("ix_jobs_type", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")

    op.drop_index("ix_datasets_bucket_id", table_name="datasets")
    op.drop_index("ix_datasets_status", table_name="datasets")
    op.drop_table("datasets")
