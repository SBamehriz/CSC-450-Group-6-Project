"""A stand-in for Hugging Face: local files listed, described and downloaded as if remote."""

import json
import shutil
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import DatasetCardData

from forge import dataset_import, jobs

SHA = "c0ffee" * 7
CONVERTED = "refs/convert/parquet"


def _feature(kind: pa.DataType) -> object:
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return {"dtype": "string", "_type": "Value"}
    if pa.types.is_list(kind) and pa.types.is_struct(kind.value_type):
        return [{field.name: _feature(field.type) for field in kind.value_type}]
    return {"dtype": str(kind), "_type": "Value"}


class _Answer:
    def __init__(self, value: dict) -> None:
        self.raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.raw


class Hub:
    def __init__(self, root: Path) -> None:
        self.root = root
        # repo -> (sha, card, {path: local file}); the converted copy is kept apart
        self.repos: dict[str, tuple[str, dict | None, dict[str, Path]]] = {}
        self.converted: dict[str, dict[str, Path]] = {}
        self.downloads: list[str] = []
        # paths that fail this many more times before they download
        self.flaky: dict[str, int] = {}

    def _save(self, folder: str, files: dict[str, pa.Table | str]) -> dict[str, Path]:
        found = {}
        for path, content in files.items():
            local = self.root / folder / path
            local.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                local.write_text(content, encoding="utf-8")
            else:
                pq.write_table(content, local, row_group_size=700)
            found[path] = local
        return found

    def add(self, repo: str, files: dict, card: dict | None = None, sha: str = SHA) -> None:
        self.repos[repo] = (sha, card, self._save(sha, files))

    def add_converted(self, repo: str, files: dict) -> None:
        self.converted[repo] = self._save(f"{repo}-converted", files)

    def info(self, repo: str, revision: str):
        if revision == CONVERTED:
            files, card, sha = self.converted[repo], None, CONVERTED
        else:
            sha, card, files = self.repos[repo]
        siblings = [SimpleNamespace(rfilename=path) for path in files]
        card_data = DatasetCardData(**card) if card is not None else None
        return SimpleNamespace(sha=sha, siblings=siblings, card_data=card_data)

    def download(self, repo: str, revision: str, path: str, folder: Path) -> Path:
        if self.flaky.get(path):
            self.flaky[path] -= 1
            raise OSError("the connection was reset")
        files = self.converted[repo] if revision == CONVERTED else self.repos[repo][2]
        self.downloads.append(path)
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[path], target)
        return target

    def _describe(self, repo: str) -> dict:
        """What the datasets server's /info says: each config's splits, rows and columns."""
        found = {}
        layout = dataset_import._layout(self.info(repo, "main"))
        files = self.repos[repo][2]
        if not layout and repo in self.converted:
            files = self.converted[repo]
            for path in files:
                config, split = path.split("/")[:2]
                layout.setdefault(config, {}).setdefault(split.removeprefix("partial-"), [])
                layout[config][split.removeprefix("partial-")].append(path)
        for config, splits in layout.items():
            first = files[next(iter(splits.values()))[0]]
            if first.suffix == ".parquet":
                features = {f.name: _feature(f.type) for f in pq.read_schema(first)}
            else:
                record = json.loads(first.read_text(encoding="utf-8").splitlines()[0])
                features = {k: _feature(pa.array([v]).type) for k, v in record.items()}
            counts = {
                split: sum(_rows(files[path]) for path in paths) for split, paths in splits.items()
            }
            found[config] = {
                "features": features,
                "splits": {split: {"num_examples": n} for split, n in counts.items()},
            }
        return {"dataset_info": found}

    def urlopen(self, request, timeout=None):
        url = request.full_url
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        repo = query.get("dataset") or url.removeprefix(dataset_import.HF_HUB_API + "/")
        if repo not in self.repos:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        if url.startswith(dataset_import.HF_HUB_API):
            return _Answer({"id": repo, "sha": self.repos[repo][0]})
        if "/info?" in url:
            return _Answer(self._describe(repo))
        if "/parquet?" in url and repo in self.converted:
            revision = urllib.parse.quote(CONVERTED, safe="")
            base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
            listing = [
                {
                    "config": path.split("/")[0],
                    "split": path.split("/")[1].removeprefix("partial-"),
                    "url": f"{base}/{path}",
                }
                for path in self.converted[repo]
            ]
            return _Answer({"parquet_files": listing, "partial": True})
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)


def _rows(path: Path) -> int:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def install(monkeypatch, root: Path) -> Hub:
    hub = Hub(root)
    monkeypatch.setattr(jobs, "start_runner", lambda: None)
    monkeypatch.setattr(dataset_import, "_hub_info", hub.info)
    monkeypatch.setattr(dataset_import, "_hub_download", hub.download)
    monkeypatch.setattr(dataset_import.urllib.request, "urlopen", hub.urlopen)
    monkeypatch.setattr(dataset_import, "WAIT", 0.0)
    return hub


def start(client, bucket_id, **ask):
    body = {"config": "default", "split": "train", "text_column": "text", **ask}
    return client.post(f"/api/buckets/{bucket_id}/import", json=body)
