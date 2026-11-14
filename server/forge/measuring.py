import secrets
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge.chatformat import CHAT_TEMPLATE, END, SPECIAL_TOKENS
from forge.config import get_settings
from forge.db import get_engine
from forge.jobs import JobFailed
from forge.models import Checkpoint, DatasetBuild, Model, Run, Tokenizer

# running a model takes its weights plus room for the numbers in between
RUN_OVERHEAD = 1.3


@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    model_id: uuid.UUID
    name: str
    parameters: int
    checkpoint_id: uuid.UUID | None
    device: str
    # it has learned to answer in turns
    chat: bool = False
    # word pieces it reads at once
    window: int = 2048
    # ids that end an answer
    ends: tuple[int, ...] = ()


class TooLong(JobFailed):
    """The last message alone does not fit in the model's window."""


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


def card_room() -> int | None:
    """Free bytes on the graphics card, or None without one."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.mem_get_info()[0])


def chat_marks(path: Path) -> bool:
    """The vocabulary keeps the chat marks at ids 0 to 4. New ones do, older ones do not."""
    from tokenizers import Tokenizer as Vocabulary

    words = Vocabulary.from_file(str(path))
    return all(words.token_to_id(mark) == index for index, mark in enumerate(SPECIAL_TOKENS))


def forge_tokenizer(path: Path, ctx_len: int):
    from transformers import PreTrainedTokenizerFast

    chat = (
        {"extra_special_tokens": list(SPECIAL_TOKENS[1:]), "chat_template": CHAT_TEMPLATE}
        if chat_marks(path)
        else {}
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        eos_token="<|eos|>",
        bos_token="<|eos|>",
        model_max_length=ctx_len,
        **chat,
    )


def _ends(model, tokenizer) -> tuple[int, ...]:
    found = getattr(model.generation_config, "eos_token_id", None)
    ids = set(found if isinstance(found, list) else [found])
    ids.add(tokenizer.eos_token_id)
    if END in tokenizer.all_special_tokens:
        ids.add(tokenizer.convert_tokens_to_ids(END))
    return tuple(sorted(one for one in ids if isinstance(one, int)))


def load(model_id: uuid.UUID, checkpoint_id: uuid.UUID | None, asked: str = "auto") -> Loaded:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        name, parameters, chat = row.name, row.param_count, row.chat
        if row.source == "hub":
            if not row.artifact_uri:
                raise JobFailed(f"{name} has no files on this computer. Import it again.")
            folder = get_settings().data_dir / row.artifact_uri
            model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            chat = chat or bool(tokenizer.chat_template)
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
            tokenizer = forge_tokenizer(
                get_settings().data_dir / vocabulary.artifact_uri, state["config"]["ctx_len"]
            )
            chosen = checkpoint.id
    model, tokenizer = cast(Any, model), cast(Any, tokenizer)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    config = model.config
    window = (
        getattr(config, "n_positions", None)
        or getattr(config, "max_position_embeddings", None)
        or 2048
    )
    ends = _ends(model, tokenizer)
    return Loaded(model, tokenizer, model_id, name, parameters, chosen, device, chat, window, ends)


def _ids(loaded: Loaded, text: str) -> list[int]:
    return loaded.tokenizer(text, add_special_tokens=False)["input_ids"]


def layout(loaded: Loaded, messages: list[dict]) -> str:
    """The conversation as the model reads it, ready for its answer."""
    if loaded.tokenizer.chat_template:
        return loaded.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    said = "".join(f"{turn['role'].title()}: {turn['content']}\n" for turn in messages)
    return said + "Assistant:"


def fit(loaded: Loaded, messages: list[dict], most: int) -> tuple[list[int], int]:
    """Word pieces of the conversation and room for the answer, old turns dropped first."""
    system = messages[:1] if messages[0]["role"] == "system" else []
    rest = messages[len(system) :]
    while True:
        ids = _ids(loaded, layout(loaded, system + rest))
        if len(ids) + most <= loaded.window or len(rest) <= 1:
            break
        rest = rest[1:]
    room = min(most, loaded.window - len(ids))
    if room < 1:
        raise TooLong(
            f"Your message is longer than {loaded.name} can read at once "
            f"({loaded.window:,} word pieces). Shorten it, or start a new conversation."
        )
    return ids, room


def generate(
    loaded: Loaded,
    ids: list[int],
    room: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    streamer=None,
    stop: threading.Event | None = None,
) -> list[int]:
    """New word pieces, the end mark included when the model wrote one."""
    import torch
    from transformers import (
        GenerationConfig,
        LogitsProcessor,
        LogitsProcessorList,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class Sample(LogitsProcessor):
        # its own random numbers, so two answers at once never share them
        def __init__(self) -> None:
            chosen = secrets.randbits(31) if seed is None else seed
            self.random = torch.Generator(device=loaded.device).manual_seed(chosen)

        def __call__(self, input_ids, scores):
            probs = torch.softmax(scores.float() / temperature, dim=-1)
            if top_p < 1.0:
                ordered, order = probs.sort(dim=-1, descending=True)
                drop = ordered.cumsum(dim=-1) - ordered > top_p
                probs = probs.scatter(-1, order, ordered.masked_fill(drop, 0.0))
            picked = torch.multinomial(probs, 1, generator=self.random)
            return torch.full_like(scores, float("-inf")).scatter(-1, picked, 0.0)

    class Stopped(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            asked = stop is not None and stop.is_set()
            return torch.full(
                (input_ids.shape[0],), asked, dtype=torch.bool, device=input_ids.device
            )

    prompt = torch.tensor([ids], device=loaded.device)
    settings = GenerationConfig(
        max_new_tokens=room,
        do_sample=False,
        eos_token_id=list(loaded.ends),
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    with torch.no_grad():
        output = loaded.model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            generation_config=settings,
            logits_processor=LogitsProcessorList([Sample()] if temperature > 0 else []),
            stopping_criteria=StoppingCriteriaList([Stopped()]),
            streamer=streamer,
        )
    return output[0, len(ids) :].tolist()
