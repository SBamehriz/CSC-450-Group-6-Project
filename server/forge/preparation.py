import hashlib
import io
import json
import sys
from array import array
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from forge.config import get_settings
from forge.errors import ApiError
from forge.models import Dataset
from forge.snapshots import file_hash, json_bytes, read_snapshot

MAX_PREPARE_BYTES = 10 * 1024 * 1024


def tools_available() -> bool:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def prepared_path(dataset: Dataset) -> Path:
    return get_settings().data_dir / "prepared" / f"{dataset.id}.zip"


def build_prepared(dataset: Dataset, requested_vocab_size: int) -> dict:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    _source_manifest, splits = read_snapshot(dataset)
    source_bytes = sum(len(row["text"].encode("utf-8")) for rows in splits.values() for row in rows)
    if source_bytes > MAX_PREPARE_BYTES:
        raise ApiError(
            413,
            "too_large",
            "Preparing data is limited to 10 MB of text for now. Make a smaller snapshot.",
        )
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=requested_vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|eos|>"],
        show_progress=False,
    )
    # Only training files teach the vocabulary.
    tokenizer.train_from_iterator((row["text"] for row in splits["train"]), trainer=trainer)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.token_to_id("<|eos|>") != 0:
        raise ApiError(503, "prepare_failed", "Could not reserve the document boundary token.")
    files = {"tokenizer.json": tokenizer.to_str(pretty=True).encode("utf-8")}
    records, counts = {}, {}
    for split, rows in splits.items():
        stream = io.BytesIO()
        records[split] = []
        offset = 0
        for row in rows:
            ids = tokenizer.encode(row["text"], add_special_tokens=False).ids + [0]
            tokens = array("H", ids)
            if tokens.itemsize != 2:
                raise ApiError(
                    503, "prepare_failed", "This computer cannot write the token format."
                )
            if sys.byteorder != "little":
                tokens.byteswap()
            stream.write(tokens.tobytes())
            records[split].append(
                {
                    "id": row["id"],
                    "content_hash": row["content_hash"],
                    "offset": offset,
                    "tokens": len(ids),
                }
            )
            offset += len(ids)
        files[f"{split}.bin"] = stream.getvalue()
        counts[split] = offset
    vocabulary = tokenizer.get_vocab_size()
    manifest = {
        "format": "forge-uint16-le-v1",
        "dataset_id": str(dataset.id),
        "snapshot_sha256": dataset.config["sha256"],
        "seed": dataset.config["seed"],
        "vocab_size": vocabulary,
        "requested_vocab_size": requested_vocab_size,
        "eos_id": 0,
        "trained_on": "train",
        "trained_on_documents": len(splits["train"]),
        "train_tokens": counts["train"],
        "validation_tokens": counts["validation"],
        "documents": records,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    }
    destination = prepared_path(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with ZipFile(temporary, "w") as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
            archive.writestr("manifest.json", json_bytes(manifest))
        temporary.replace(destination)
        return {
            "ready": True,
            "vocab_size": vocabulary,
            "requested_vocab_size": requested_vocab_size,
            "train_tokens": counts["train"],
            "validation_tokens": counts["validation"],
            "trained_on_documents": len(splits["train"]),
            "eos_id": 0,
            "sha256": file_hash(destination),
            "snapshot_sha256": dataset.config["sha256"],
        }
    except OSError as problem:
        destination.unlink(missing_ok=True)
        raise ApiError(
            503, "storage_unavailable", "Could not save the prepared files. Check free disk space."
        ) from problem
    finally:
        temporary.unlink(missing_ok=True)


def read_prepared(dataset: Dataset) -> tuple[dict, dict[str, bytes]]:
    stats = dataset.config.get("prepared", {})
    path = prepared_path(dataset)
    if not stats.get("ready"):
        raise ApiError(409, "not_ready", "Prepare this snapshot first.")
    try:
        if not path.is_file() or file_hash(path) != stats.get("sha256"):
            raise ValueError(
                "The prepared ZIP is missing or has changed. Remove it and prepare again."
            )
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if manifest["snapshot_sha256"] != dataset.config["sha256"] or manifest[
                "dataset_id"
            ] != str(dataset.id):
                raise ValueError("The prepared files belong to a different snapshot.")
            files = {
                name: archive.read(name)
                for name in ("tokenizer.json", "train.bin", "validation.bin")
            }
            for name, raw in files.items():
                if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                    raise ValueError("A file inside the prepared ZIP has changed.")
            return manifest, files
    except (OSError, BadZipFile, ValueError, KeyError, TypeError) as problem:
        raise ApiError(
            409,
            "invalid_prepared",
            str(problem)
            if isinstance(problem, ValueError)
            else "Could not read the prepared files. Prepare the snapshot again.",
        ) from problem
