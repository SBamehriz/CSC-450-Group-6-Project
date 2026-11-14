"""fine-tuning and chat

Revision ID: 0011
Revises: 0010
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("buckets") as batch:
        batch.add_column(sa.Column("kind", sa.Text(), nullable=False, server_default="text"))
    with op.batch_alter_table("models") as batch:
        batch.add_column(sa.Column("parent_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("chat", sa.Boolean(), nullable=False, server_default="0"))
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("init_checkpoint_id", sa.Uuid(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("init_checkpoint_id")
    with op.batch_alter_table("models") as batch:
        batch.drop_column("chat")
        batch.drop_column("parent_id")
    with op.batch_alter_table("buckets") as batch:
        batch.drop_column("kind")
