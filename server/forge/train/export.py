import json
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.jobs import JobFailed, report
from forge.models import Checkpoint, DatasetBuild, Job, Model, Run, Tokenizer

GPT2_NAMES = {
    "norm1": "ln_1",
    "attn.qkv": "attn.c_attn",
    "attn.proj": "attn.c_proj",
    "norm2": "ln_2",
    "mlp.fc": "mlp.c_fc",
    "mlp.proj": "mlp.c_proj",
}
EOS_ID = 0


def export_dir(checkpoint_id: uuid.UUID) -> Path:
    return get_settings().data_dir / "exports" / str(checkpoint_id)


def gpt2_state(state: dict, n_layer: int) -> dict:
    renamed = {
        "transformer.wte.weight": state["token_emb.weight"],
        "transformer.wpe.weight": state["pos_emb.weight"],
        "transformer.ln_f.weight": state["norm_final.weight"],
        "transformer.ln_f.bias": state["norm_final.bias"],
        "lm_head.weight": state["token_emb.weight"],
    }
    for index in range(n_layer):
        for ours, theirs in GPT2_NAMES.items():
            weight = state[f"blocks.{index}.{ours}.weight"]
            linear = not ours.startswith("norm")
            renamed[f"transformer.h.{index}.{theirs}.weight"] = (
                weight.t().contiguous() if linear else weight
            )
            renamed[f"transformer.h.{index}.{theirs}.bias"] = state[f"blocks.{index}.{ours}.bias"]
    return renamed


def gpt2_model(state: dict, config: dict):
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=config["vocab_size"],
            n_positions=config["ctx_len"],
            n_embd=config["d_model"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            activation_function="gelu",
            resid_pdrop=config.get("dropout", 0.0),
            embd_pdrop=config.get("dropout", 0.0),
            attn_pdrop=config.get("dropout", 0.0),
            layer_norm_epsilon=1e-5,
            bos_token_id=EOS_ID,
            eos_token_id=EOS_ID,
            tie_word_embeddings=True,
        )
    )
    model.load_state_dict(gpt2_state(state, config["n_layer"]), strict=True)
    return model


def run(job_id: uuid.UUID) -> dict:
    try:
        import transformers  # noqa: F401

        from forge.train.trainer import load_checkpoint
    except ImportError as problem:
        raise JobFailed(
            "Exporting needs PyTorch and transformers. Install the train and hub extras first."
        ) from problem

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        checkpoint = session.get(Checkpoint, uuid.UUID(job.payload["checkpoint_id"]))
        if checkpoint is None:
            raise JobFailed("This save point was deleted.")
        run_row = session.get(Run, checkpoint.run_id)
        model_row = session.get(Model, run_row.model_id) if run_row else None
        build = (
            session.get(DatasetBuild, run_row.build_id) if run_row and run_row.build_id else None
        )
        vocabulary = session.get(Tokenizer, build.tokenizer_id) if build else None
        vocab_path = (
            get_settings().data_dir / vocabulary.artifact_uri
            if vocabulary and vocabulary.artifact_uri
            else None
        )
        name = model_row.name if model_row else "model"
        report(job_id, 0, 3, "Reading the save point")
        state = load_checkpoint(checkpoint)
        checkpoint_id, step = checkpoint.id, checkpoint.step

    destination = export_dir(checkpoint_id)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        report(job_id, 1, 3, "Writing the Hugging Face folder")
        gpt2_model(state["model"], state["config"]).save_pretrained(
            temporary, safe_serialization=True
        )
        if vocab_path and vocab_path.is_file():
            shutil.copyfile(vocab_path, temporary / "tokenizer.json")
            (temporary / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "PreTrainedTokenizerFast",
                        "bos_token": "<|eos|>",
                        "eos_token": "<|eos|>",
                        "model_max_length": state["config"]["ctx_len"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        report(job_id, 2, 3, "Finishing")
        shutil.rmtree(destination, ignore_errors=True)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    files = sorted(path.name for path in destination.iterdir())
    return {
        "checkpoint_id": str(checkpoint_id),
        "model": name,
        "step": step,
        "folder": destination.relative_to(get_settings().data_dir).as_posix(),
        "files": files,
    }
