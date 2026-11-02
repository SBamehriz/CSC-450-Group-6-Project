from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build', 'dataset_import')"
)
NEW_TYPES = (
    "job_type IN ('tokenization', 'training', 'eval', 'ingest', 'export', "
    "'tokenizer_train', 'dataset_build', 'dataset_import', 'snapshot_create')"
)


def upgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", NEW_TYPES)

    op.create_index("ix_documents_bucket_status", "documents", ["bucket_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_documents_bucket_status", table_name="documents")
    op.execute("DELETE FROM jobs WHERE job_type = 'snapshot_create'")
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_constraint("ck_jobs_type", type_="check")
        batch.create_check_constraint("ck_jobs_type", OLD_TYPES)
