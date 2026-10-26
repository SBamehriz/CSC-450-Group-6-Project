from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export')"
NEW_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build')"
)


def upgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.add_column(sa.Column("pid", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="0")
        )
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", NEW_TYPES)

    op.create_table(
        "tokenizers",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "dataset_id",
            sa.Uuid(),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("requested_vocab_size", sa.Integer(), nullable=False),
        sa.Column("vocab_size", sa.Integer(), nullable=True),
        sa.Column("trained_on", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'building', 'ready', 'failed', 'cancelled')",
            name="ck_tokenizers_status",
        ),
        sa.CheckConstraint(
            "requested_vocab_size BETWEEN 257 AND 65535", name="ck_tokenizers_vocab"
        ),
    )

    op.create_table(
        "dataset_builds",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "tokenizer_id",
            sa.Uuid(),
            sa.ForeignKey("tokenizers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "dataset_id",
            sa.Uuid(),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("manifest_uri", sa.Text(), nullable=True),
        sa.Column("stats", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'building', 'ready', 'failed', 'cancelled')",
            name="ck_builds_status",
        ),
    )
    op.create_index("ix_dataset_builds_tokenizer_id", "dataset_builds", ["tokenizer_id"])


def downgrade() -> None:
    op.drop_index("ix_dataset_builds_tokenizer_id", table_name="dataset_builds")
    op.drop_table("dataset_builds")
    op.drop_table("tokenizers")

    op.execute("DELETE FROM jobs WHERE job_type IN ('tokenizer_train', 'dataset_build')")
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", OLD_TYPES)
        batch.drop_column("cancel_requested")
        batch.drop_column("attempts")
        batch.drop_column("pid")
