import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# repo root
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path

    @property
    def database_path(self) -> Path | None:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        return Path(self.database_url[len(prefix) :])


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("FORGE_DATA_DIR", ROOT / "forge-data"))
    default_db = f"sqlite:///{data_dir / 'forge.db'}"
    return Settings(
        database_url=os.environ.get("FORGE_DATABASE_URL", default_db),
        data_dir=data_dir,
    )


def ensure_data_dir() -> Path:
    # sqlite wont make the folder
    directory = get_settings().data_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@lru_cache
def get_settings() -> Settings:
    return load_settings()
