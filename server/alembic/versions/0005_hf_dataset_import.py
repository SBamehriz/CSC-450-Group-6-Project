from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build')"
)
NEW_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build', 'dataset_import')"
)


def upgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", NEW_TYPES)

    op.add_column("buckets", sa.Column("source_dataset", sa.Text(), nullable=True))
    op.add_column("buckets", sa.Column("resolved_revision", sa.Text(), nullable=True))
    op.add_column("buckets", sa.Column("dataset_config", sa.Text(), nullable=True))
    op.add_column("buckets", sa.Column("dataset_split", sa.Text(), nullable=True))
    op.add_column("buckets", sa.Column("text_column", sa.Text(), nullable=True))
    op.add_column("buckets", sa.Column("import_limit", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("buckets") as batch:
        batch.drop_column("import_limit")
        batch.drop_column("text_column")
        batch.drop_column("dataset_split")
        batch.drop_column("dataset_config")
        batch.drop_column("resolved_revision")
        batch.drop_column("source_dataset")

    op.execute("DELETE FROM jobs WHERE job_type = 'dataset_import'")
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", OLD_TYPES)
