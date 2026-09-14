from logging.config import fileConfig

from sqlalchemy import create_engine

from alembic import context
from forge.config import ensure_data_dir, load_settings
from forge.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    # -x db_url=... overrides
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    if override:
        return override
    ensure_data_dir()
    return load_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), future=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # sqlite cant ALTER, so batch mode
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
