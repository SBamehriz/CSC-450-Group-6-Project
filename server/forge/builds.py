import hashlib
import json
import subprocess
import sys
import uuid
from array import array
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


def _snapshot(dataset_id: uuid.UUID | None) -> tuple[Dataset, dict[str, list[dict]]]:
    with Session(get_engine()) as session:
        dataset = session.get(Dataset, dataset_id) if dataset_id else None
        if dataset is None:
            raise ValueError("The snapshot was removed. Choose another snapshot.")
        _manifest, splits = read_snapshot(dataset)
        return dataset, splits


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
    dataset, splits = _snapshot(dataset_id)
    # Only training files teach the vocabulary.
    documents = splits["train"]

    def texts():
        for index, document in enumerate(documents):
            report(job_id, index, len(documents), "Reading training text")
            yield document["text"]
        report(job_id, len(documents), len(documents), "Learning word pieces")

    tokenizer = BpeTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts(), trainer=trainer, length=len(documents))
    report(job_id, len(documents), len(documents), "Saving vocabulary")
    if tokenizer.token_to_id("<|eos|>") != 0:
        raise ValueError("The vocabulary did not reserve token 0 for document boundaries.")
    _json_file(destination / "tokenizer.json", json.loads(tokenizer.to_str()))
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "trained_on": {
            "split": "train",
            "snapshot_sha256": dataset.config.get("sha256"),
            "documents": len(documents),
            "hashes": [document["content_hash"] for document in documents],
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
    dataset, splits = _snapshot(dataset_id)
    total_docs = sum(len(rows) for rows in splits.values())
    completed = total_tokens = 0
    stats, manifest_splits = {}, {}
    for split in SPLITS:
        counts = SplitStats()
        entries = []
        path = destination / f"{split}.bin"
        path.touch()
        for document in splits[split]:
            report(job_id, completed, total_docs, f"Turning {split} text into tokens")
            source = document["text"]
            ids = tokenizer.encode(source).ids + [0]
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
