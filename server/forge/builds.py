import hashlib
import json
import subprocess
import sys
import uuid
from array import array
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.jobs import report
from forge.models import Dataset, DatasetBuild, Tokenizer
from forge.schemas import SplitStats
from forge.snapshots import read_snapshot

MAX_SPLIT_BYTES = 2_000_000_000
MAX_BUILD_TOKENS = 2_000_000_000
MIN_VAL_TOKENS = 100_000
SPLITS = ("train", "validation")


# text goes to the vocabulary trainer and the tokenizer in pieces this size:
# a 45 MB file in one go took over 5 GB of memory, in pieces it takes under 1 GB
PIECE_CHARS = 100_000


class Snapshot:
    """A snapshot read one document at a time, each checked as it is read."""

    def __init__(self, dataset_id: uuid.UUID | None) -> None:
        with Session(get_engine()) as session:
            dataset = session.get(Dataset, dataset_id) if dataset_id else None
            if dataset is None:
                raise ValueError("The snapshot was removed. Choose another snapshot.")
            session.expunge(dataset)
        self.dataset = dataset
        archive = get_settings().data_dir / "datasets" / f"{dataset.id}.zip"
        expected = dataset.config.get("sha256")
        if archive.is_file() and expected and _file_hash(archive) != expected:
            raise ValueError("The snapshot is missing or its checksum changed. Build a new one.")
        folder = get_settings().data_dir / "datasets" / str(dataset.id)
        self.folder = folder if (folder / "manifest.json").is_file() else None
        self.splits: dict[str, list[dict]] | None = None
        if self.folder is None:
            _manifest, self.splits = read_snapshot(dataset)
            self.counts = {split: len(rows) for split, rows in self.splits.items()}
            return
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "forge-text-snapshot-v1" or manifest.get("dataset_id") != str(
            dataset.id
        ):
            raise ValueError("The snapshot manifest does not match this dataset.")
        for split in SPLITS:
            if _file_hash(folder / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
                raise ValueError("A text file inside the snapshot has changed.")
        documents = manifest.get("documents", [])
        self.counts = {
            split: sum(1 for row in documents if row.get("split") == split) for split in SPLITS
        }

    def rows(self, split: str) -> Iterator[dict]:
        if self.splits is not None:
            yield from self.splits[split]
            return
        assert self.folder is not None
        seen = 0
        with (self.folder / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                if digest != record["content_hash"]:
                    raise ValueError("A document inside the snapshot has changed.")
                seen += 1
                yield record
        if not seen:
            raise ValueError("Each split needs at least one document.")


def pieces(text: str, size: int = PIECE_CHARS) -> Iterator[str]:
    """Cut text into pieces of about `size` characters, at line ends where there is one."""
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut > start:
                end = cut + 1
        yield text[start:end]
        start = end


def _file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json_file(path: Path, value: dict) -> None:
    # Sorting removes Rust/Python hash-map iteration from the artifact bytes.
    path.write_bytes(
        (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def train_tokenizer(job_id: uuid.UUID, target_id: uuid.UUID, destination: Path) -> dict:
    from tokenizers import Tokenizer as BpeTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers

    with Session(get_engine()) as session:
        target = session.get(Tokenizer, target_id)
        assert target is not None
        dataset_id, vocab_size = target.dataset_id, target.requested_vocab_size
    snapshot = Snapshot(dataset_id)
    dataset = snapshot.dataset
    # Only training files teach the vocabulary.
    total = snapshot.counts["train"]
    hashes: list[str] = []

    def texts():
        for index, document in enumerate(snapshot.rows("train")):
            report(job_id, index, total, "Reading training text")
            hashes.append(document["content_hash"])
            yield from pieces(document["text"])
        report(job_id, total, total, "Learning word pieces")

    tokenizer = BpeTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    report(job_id, total, total, "Saving vocabulary")
    if tokenizer.token_to_id("<|eos|>") != 0:
        raise ValueError("The vocabulary did not reserve token 0 for document boundaries.")
    _json_file(destination / "tokenizer.json", json.loads(tokenizer.to_str()))
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "trained_on": {
            "split": "train",
            "snapshot_sha256": dataset.config.get("sha256"),
            "documents": len(hashes),
            "hashes": hashes,
        },
    }


def build_dataset(job_id: uuid.UUID, target_id: uuid.UUID, destination: Path) -> dict:
    from tokenizers import Tokenizer as BpeTokenizer

    with Session(get_engine()) as session:
        build = session.get(DatasetBuild, target_id)
        assert build is not None
        tokenizer_row = session.get(Tokenizer, build.tokenizer_id)
        if (
            tokenizer_row is None
            or tokenizer_row.status != "ready"
            or not tokenizer_row.artifact_uri
        ):
            raise ValueError("Choose a ready vocabulary before making a training set.")
        tokenizer_path = get_settings().data_dir / tokenizer_row.artifact_uri
        dataset_id, tokenizer_id = build.dataset_id, build.tokenizer_id
    tokenizer = BpeTokenizer.from_file(str(tokenizer_path))
    snapshot = Snapshot(dataset_id)
    dataset = snapshot.dataset
    total_docs = sum(snapshot.counts.values())
    completed = total_tokens = 0
    stats, manifest_splits = {}, {}
    for split in SPLITS:
        counts = SplitStats()
        entries = []
        path = destination / f"{split}.bin"
        path.touch()
        for document in snapshot.rows(split):
            report(job_id, completed, total_docs, f"Turning {split} text into tokens")
            source = document["text"]
            ids = [
                token
                for encoded in tokenizer.encode_batch(list(pieces(source)))
                for token in encoded.ids
            ]
            ids.append(0)
            total_tokens += len(ids)
            if total_tokens > MAX_BUILD_TOKENS or (counts.tokens + len(ids)) * 2 > MAX_SPLIT_BYTES:
                raise ValueError("This training set is too large. Use a smaller snapshot.")
            tokens = array("H", ids)
            if sys.byteorder != "little":
                tokens.byteswap()
            with path.open("ab") as output:
                tokens.tofile(output)
            counts.documents += 1
            counts.tokens += len(ids)
            counts.bytes += len(source.encode("utf-8"))
            entries.append({"id": document["id"], "content_hash": document["content_hash"]})
            completed += 1
        stats[split] = counts.model_dump()
        manifest_splits[split] = {
            "file": path.name,
            **counts.model_dump(),
            "documents_in_order": entries,
            "sha256": _file_hash(path),
        }
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        # A copied app folder can run without the git command installed.
        revision = "unknown"
    warnings = []
    if manifest_splits["validation"]["tokens"] < MIN_VAL_TOKENS:
        warnings.append("Validation has fewer than 100,000 tokens; comparisons may be noisy.")
    manifest = {
        "format": "forge-uint16-le-v1",
        "tokenizer_id": str(tokenizer_id),
        "tokenizer_sha256": _file_hash(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": 0,
        "snapshot_id": str(dataset.id),
        "snapshot_sha256": dataset.config.get("sha256"),
        "forge_git_sha": revision,
        "splits": manifest_splits,
        "warnings": warnings,
    }
    _json_file(destination / "manifest.json", manifest)
    report(job_id, completed, total_docs, "Saving training set")
    return {"stats": stats}
