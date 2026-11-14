"""A tiny real Forge model with random weights, saved the way the trainer saves one."""

import hashlib
import uuid

import torch
from tokenizers import Tokenizer as Vocabulary
from tokenizers import decoders, models, pre_tokenizers, trainers

from forge.chatformat import EOS, SPECIAL_TOKENS
from forge.config import get_settings
from forge.models import Checkpoint, DatasetBuild, Model, Run, Tokenizer

TEXT = "the river runs past the old mill and the children watch the boats go by. " * 40


def vocabulary(db, marks: bool = True) -> Tokenizer:
    # new vocabularies keep the chat marks at ids 0 to 4, older ones only the end of text
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=list(SPECIAL_TOKENS) if marks else [EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    learned = Vocabulary(models.BPE())
    learned.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    learned.decoder = decoders.ByteLevel()
    learned.train_from_iterator([TEXT], trainer=trainer)
    row = Tokenizer(
        name=f"tiny-{uuid.uuid4().hex[:6]}",
        dataset_name="tiny",
        requested_vocab_size=320,
        vocab_size=learned.get_vocab_size(),
    )
    db.add(row)
    db.flush()
    path = get_settings().data_dir / "tokenizers" / str(row.id) / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    learned.save(str(path))
    row.artifact_uri = path.relative_to(get_settings().data_dir).as_posix()
    row.status = "ready"
    db.commit()
    return row


def weights(config: dict, seed: int, always: int | None) -> dict:
    # `always` makes greedy decoding pick that id every time
    random = torch.Generator().manual_seed(seed)
    d, v = config["d_model"], config["vocab_size"]

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=random)

    # small enough that the next word is not just the last one again
    state = {
        "token_emb.weight": normal(v, d) * 0.3,
        "pos_emb.weight": normal(config["ctx_len"], d) * 0.3,
    }
    for index in range(config["n_layer"]):
        block = f"blocks.{index}."
        state |= {
            block + "norm1.weight": torch.ones(d),
            block + "norm1.bias": torch.zeros(d),
            block + "attn.qkv.weight": normal(3 * d, d) * 0.1,
            block + "attn.qkv.bias": torch.zeros(3 * d),
            block + "attn.proj.weight": normal(d, d) * 0.1,
            block + "attn.proj.bias": torch.zeros(d),
            block + "norm2.weight": torch.ones(d),
            block + "norm2.bias": torch.zeros(d),
            block + "mlp.fc.weight": normal(4 * d, d) * 0.1,
            block + "mlp.fc.bias": torch.zeros(4 * d),
            block + "mlp.proj.weight": normal(d, 4 * d) * 0.1,
            block + "mlp.proj.bias": torch.zeros(d),
        }
    state["norm_final.weight"] = torch.ones(d)
    state["norm_final.bias"] = torch.zeros(d)
    if always is not None:
        row = state["token_emb.weight"][always]
        state["norm_final.weight"] = torch.zeros(d)
        state["norm_final.bias"] = 10 * row / row.norm()
    return state


def make_forge_model(
    db,
    name: str | None = None,
    chat: bool = False,
    always: int | None = None,
    seed: int = 0,
    ctx_len: int = 64,
    marks: bool = True,
) -> Model:
    words = vocabulary(db, marks)
    build = DatasetBuild(
        name=f"tiny-{uuid.uuid4().hex[:6]}",
        tokenizer_id=words.id,
        dataset_name="tiny",
        status="ready",
    )
    db.add(build)
    config = {
        "n_layer": 2,
        "d_model": 32,
        "n_head": 2,
        "ctx_len": ctx_len,
        "vocab_size": words.vocab_size,
        "dropout": 0.0,
    }
    model = Model(
        name=name or f"tiny-{uuid.uuid4().hex[:6]}",
        param_count=30_000,
        tokenizer_id=words.id,
        chat=chat,
        config={},
        **config,
    )
    db.add(model)
    db.flush()
    run = Run(name="first", model_id=model.id, build_id=build.id, status="completed")
    db.add(run)
    db.flush()
    path = get_settings().data_dir / "runs" / str(run.id) / "step-00000010.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # the keys of the trainer's state_now
    torch.save(
        {
            "format": "forge-gpt-v1",
            "model": weights(config, seed, always),
            "optimizer": {},
            "step": 10,
            "tokens_seen": 5000,
            "config": config,
            "settings": {},
            "seed": 0,
            "tokenizer_id": str(words.id),
            "rng": {"torch": torch.get_rng_state(), "cuda": None},
        },
        path,
    )
    db.add(
        Checkpoint(
            run_id=run.id,
            step=10,
            loss=2.1,
            val_loss=2.0,
            uri=path.relative_to(get_settings().data_dir).as_posix(),
            size_bytes=path.stat().st_size,
            is_best=True,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
    db.commit()
    return model
