import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge import gpu
from forge.config import ensure_data_dir, get_settings
from forge.db import get_engine
from forge.jobs import JobFailed, report
from forge.models import Job, Model

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")
PICKLES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")
EXTRAS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
# room left on the disk afterwards, so the computer keeps working
DISK_MARGIN = 1_000_000_000
# running a model takes its weights plus room for the numbers in between
RUN_OVERHEAD = 1.2
TIMEOUT = 10


class Refused(Exception):
    """A plain reason this model can't be imported."""


def models_dir() -> Path:
    return get_settings().data_dir / "models"


def listing(repo_id: str, revision: str) -> tuple[dict, dict]:
    """(files, facts) for one revision, straight from the hub."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True, timeout=TIMEOUT)
    except GatedRepoError as problem:
        raise Refused(
            f"{repo_id} is gated. Accept its terms on huggingface.co and sign in on this "
            "computer first."
        ) from problem
    except RepositoryNotFoundError as problem:
        raise Refused(f"There is no model called {repo_id} on the hub.") from problem
    except RevisionNotFoundError as problem:
        raise Refused(f"{repo_id} has no version called {revision!r}.") from problem
    except OSError as problem:
        raise Refused(
            f"Could not reach the Hugging Face hub ({problem}). Check the internet connection."
        ) from problem
    files = {
        one.rfilename: {"size": one.size or 0, "sha256": one.lfs.sha256 if one.lfs else None}
        for one in info.siblings or []
    }
    card = info.card_data
    license_name = getattr(card, "license", None) if card else None
    parameters = None
    if info.safetensors is not None:
        parameters = int(info.safetensors.total)
    return files, {
        "commit": info.sha,
        "license": str(license_name or "unknown"),
        "parameters": parameters,
    }


def read_small(repo_id: str, commit: str, name: str) -> bytes:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, name, revision=commit)).read_bytes()


def fetch(repo_id: str, commit: str, name: str, folder: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, name, revision=commit, local_dir=folder))


def shape_of(config: dict) -> dict:
    # gpt-2 style names, then llama style names
    def first(*names: str, default: int = 0) -> int:
        for name in names:
            if isinstance(config.get(name), int):
                return config[name]
        return default

    return {
        "n_layer": first("n_layer", "num_hidden_layers"),
        "d_model": first("n_embd", "hidden_size", "d_model"),
        "n_head": first("n_head", "num_attention_heads"),
        "ctx_len": first("n_positions", "max_position_embeddings", "n_ctx", default=2048),
        "vocab_size": first("vocab_size"),
    }


def _weights(files: dict, repo_id: str, commit: str) -> list[str]:
    if "model.safetensors" in files:
        return ["model.safetensors"]
    if "model.safetensors.index.json" in files:
        index = json.loads(read_small(repo_id, commit, "model.safetensors.index.json"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        for shard in shards:
            if not SAFE_FILE.match(shard) or not shard.endswith(".safetensors"):
                raise Refused(f"The shard list names {shard!r}, which is not a safetensors file.")
            if shard not in files:
                raise Refused(f"The shard list names {shard}, which is not in the model.")
        return ["model.safetensors.index.json", *shards]
    if any(name.endswith(PICKLES) for name in files):
        raise Refused(
            "This model only has pickle weights (like pytorch_model.bin), which can run code "
            "when loaded. Choose a model with safetensors weights."
        )
    raise Refused("This model has no safetensors weights.")


def plan(repo_id: str, revision: str, on_cpu: bool) -> dict:
    """Check everything that can be checked before the download."""
    if not REPO_ID.match(repo_id):
        raise Refused("Write the model as owner/name, like openai-community/gpt2.")
    files, facts = listing(repo_id, revision)
    if "config.json" not in files:
        raise Refused("This model has no config.json, so there is no way to tell its shape.")
    if "tokenizer.json" not in files:
        raise Refused("This model has no tokenizer.json, so its vocabulary can't be read.")
    weights = _weights(files, repo_id, facts["commit"])
    wanted = list(dict.fromkeys(weights + [name for name in EXTRAS if name in files]))
    for name in wanted:
        if name.endswith(".safetensors") and not files[name]["sha256"]:
            raise Refused(f"The hub lists no checksum for {name}, so it can't be checked.")
    config = json.loads(read_small(repo_id, facts["commit"], "config.json"))
    shape = shape_of(config)
    weight_bytes = sum(files[name]["size"] for name in weights if name.endswith(".safetensors"))
    needed_disk = sum(files[name]["size"] for name in wanted)
    free_disk = shutil.disk_usage(ensure_data_dir()).free - DISK_MARGIN
    if needed_disk > free_disk:
        raise Refused(
            f"This model needs {needed_disk / 1e9:.2f} GB on the disk and only "
            f"{max(free_disk, 0) / 1e9:.2f} GB is free (keeping 1 GB spare)."
        )
    run_bytes = int(weight_bytes * RUN_OVERHEAD)
    cards = gpu.read().cards
    free_card = max((card.free_bytes for card in cards), default=None)
    if free_card is not None and run_bytes > free_card and not on_cpu:
        raise Refused(
            f"This model needs about {run_bytes / 1e9:.1f} GB to run and the graphics card has "
            f"{free_card / 1e9:.1f} GB free. Close other programs that use the card, choose a "
            "smaller model, or tick Run it on the CPU (slower)."
        )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "commit": facts["commit"],
        "license": facts["license"],
        "files": {name: files[name] for name in wanted},
        "config": config,
        "shape": shape,
        "parameters": facts["parameters"] or weight_bytes // 2,
        "weight_bytes": weight_bytes,
        "run_bytes": run_bytes,
        "free_card_bytes": free_card,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _free_name(session: Session, wanted: str) -> str:
    name, number = wanted, 2
    while session.scalar(select(Model.id).where(Model.name == name)):
        name = f"{wanted} ({number})"
        number += 1
    return name


def run(job_id: uuid.UUID) -> dict:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        planned = dict(job.payload["plan"])
        wanted_name = job.payload.get("name") or planned["repo_id"].rsplit("/", 1)[-1]
    model_id = uuid.uuid4()
    destination = models_dir() / str(model_id)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        names = list(planned["files"])
        for index, name in enumerate(names):
            report(job_id, index, len(names), f"Downloading {name}")
            try:
                path = fetch(planned["repo_id"], planned["commit"], name, temporary)
            except OSError as problem:
                raise JobFailed(
                    f"The download of {name} stopped ({problem}). Check the internet "
                    "connection and Retry."
                ) from problem
            expected = planned["files"][name]["sha256"]
            if expected and _sha256(path) != expected:
                raise JobFailed(f"{name} did not match the hub's checksum, so it was thrown away.")
        # the hub's own cache notes are not part of the model
        shutil.rmtree(temporary / ".cache", ignore_errors=True)
        report(job_id, len(names), len(names), "Adding the model")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    shape = planned["shape"]
    with Session(get_engine()) as session:
        model = Model(
            id=model_id,
            name=_free_name(session, wanted_name),
            description=f"Imported from {planned['repo_id']}",
            preset=None,
            n_layer=shape["n_layer"],
            d_model=shape["d_model"],
            n_head=shape["n_head"],
            ctx_len=shape["ctx_len"],
            vocab_size=shape["vocab_size"],
            dropout=0.0,
            param_count=planned["parameters"],
            config={
                "hub": {
                    "repo_id": planned["repo_id"],
                    "revision": planned["revision"],
                    "commit": planned["commit"],
                    "license": planned["license"],
                    "files": planned["files"],
                },
                "hf_config": planned["config"],
            },
            source="hub",
            artifact_uri=destination.relative_to(get_settings().data_dir).as_posix(),
        )
        session.add(model)
        session.commit()
        name = model.name
    return {"model_id": str(model_id), "name": name, "commit": planned["commit"]}
