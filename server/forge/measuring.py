import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.jobs import JobFailed
from forge.models import Checkpoint, DatasetBuild, Model, Run, Tokenizer


@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    model_id: uuid.UUID
    name: str
    parameters: int
    checkpoint_id: uuid.UUID | None
    device: str


def pick_checkpoint(session: Session, model_id: uuid.UUID) -> Checkpoint | None:
    # the best save point across this model's runs, else the newest one
    runs = select(Run.id).where(Run.model_id == model_id)
    rows = list(session.scalars(select(Checkpoint).where(Checkpoint.run_id.in_(runs))))
    if not rows:
        return None
    scored = [row for row in rows if row.val_loss is not None]
    if scored:
        return min(scored, key=lambda row: (row.val_loss, -row.step))
    return max(rows, key=lambda row: row.created_at)


def _device(asked: str) -> str:
    import torch

    if asked == "cuda" and not torch.cuda.is_available():
        raise JobFailed("The graphics card is not available to PyTorch on this computer.")
    if asked == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_id: uuid.UUID, checkpoint_id: uuid.UUID | None, asked: str = "auto") -> Loaded:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

        from forge.train.export import gpt2_model
        from forge.train.trainer import load_checkpoint
    except ImportError as problem:
        raise JobFailed(
            "Measuring needs PyTorch and transformers. Install the train and hub extras first."
        ) from problem

    device = _device(asked)
    with Session(get_engine()) as session:
        row = session.get(Model, model_id)
        if row is None:
            raise JobFailed("This model was deleted.")
        name, parameters = row.name, row.param_count
        if row.source == "hub":
            if not row.artifact_uri:
                raise JobFailed(f"{name} has no files on this computer. Import it again.")
            folder = get_settings().data_dir / row.artifact_uri
            model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            chosen = None
        else:
            checkpoint = (
                session.get(Checkpoint, checkpoint_id) if checkpoint_id else None
            ) or pick_checkpoint(session, model_id)
            if checkpoint is None:
                raise JobFailed(f"{name} has not been trained yet, so there is nothing to measure.")
            run = session.get(Run, checkpoint.run_id)
            build = session.get(DatasetBuild, run.build_id) if run and run.build_id else None
            vocabulary = session.get(Tokenizer, build.tokenizer_id) if build else None
            if vocabulary is None or not vocabulary.artifact_uri:
                raise JobFailed(f"The vocabulary {name} was trained with is missing.")
            state = load_checkpoint(checkpoint)
            model = gpt2_model(state["model"], state["config"])
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(get_settings().data_dir / vocabulary.artifact_uri),
                eos_token="<|eos|>",
                bos_token="<|eos|>",
                model_max_length=state["config"]["ctx_len"],
            )
            chosen = checkpoint.id
    model, tokenizer = cast(Any, model), cast(Any, tokenizer)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return Loaded(model, tokenizer, model_id, name, parameters, chosen, device)
