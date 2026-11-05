from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build', 'dataset_import', 'snapshot_create')"
)
NEW_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build', 'dataset_import', 'snapshot_create', "
    "'train_run', 'speed_test', 'model_import', 'model_export', 'benchmark', "
    "'prompt_compare')"
)


def upgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.add_column(sa.Column("result", sa.JSON(), nullable=False, server_default="{}"))
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", NEW_TYPES)

    with op.batch_alter_table("runs", recreate="always") as batch:
        batch.add_column(
            sa.Column(
                "build_id",
                sa.Uuid(),
                sa.ForeignKey("dataset_builds.id", name="fk_runs_build_id", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch.add_column(sa.Column("seed", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("tokens_seen", sa.BigInteger(), nullable=False, server_default="0")
        )
        batch.create_index("ix_runs_build_id", ["build_id"])

    with op.batch_alter_table("models", recreate="always") as batch:
        batch.add_column(
            sa.Column(
                "tokenizer_id",
                sa.Uuid(),
                sa.ForeignKey("tokenizers.id", name="fk_models_tokenizer_id", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch.add_column(sa.Column("source", sa.Text(), nullable=False, server_default="forge"))
        batch.add_column(sa.Column("artifact_uri", sa.Text(), nullable=True))
        batch.create_check_constraint("ck_models_source", "source IN ('forge', 'hub')")

    with op.batch_alter_table("checkpoints", recreate="always") as batch:
        batch.add_column(sa.Column("sha256", sa.Text(), nullable=True))

    op.create_table(
        "speed_tests",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("preset", sa.Text(), nullable=True),
        sa.Column("shape", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("sequences_per_step", sa.Integer(), nullable=False),
        sa.Column("device", sa.Text(), nullable=False),
        sa.Column("device_name", sa.Text(), nullable=False),
        sa.Column("tokens_per_sec", sa.Float(), nullable=False),
        sa.Column("peak_memory_bytes", sa.BigInteger(), nullable=True),
        sa.Column("steps", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("speed_tests")

    with op.batch_alter_table("checkpoints", recreate="always") as batch:
        batch.drop_column("sha256")

    op.execute("DELETE FROM models WHERE source = 'hub'")
    with op.batch_alter_table("models", recreate="always") as batch:
        batch.drop_constraint("ck_models_source", type_="check")
        batch.drop_column("artifact_uri")
        batch.drop_column("source")
        batch.drop_column("tokenizer_id")

    with op.batch_alter_table("runs", recreate="always") as batch:
        batch.drop_index("ix_runs_build_id")
        batch.drop_column("tokens_seen")
        batch.drop_column("seed")
        batch.drop_column("build_id")

    op.execute(
        "DELETE FROM jobs WHERE job_type IN ('train_run', 'speed_test', 'model_import', "
        "'model_export', 'benchmark', 'prompt_compare')"
    )
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", OLD_TYPES)
        batch.drop_column("result")
