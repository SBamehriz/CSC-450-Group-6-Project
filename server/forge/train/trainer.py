import errno
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine, open_session
from forge.jobs import Canceled, JobFailed, report
from forge.models import Checkpoint, DatasetBuild, Job, MetricPoint, Model, Run
from forge.train.metrics import MetricBuffer
from forge.train.model import GPT
from forge.train.presets import ModelConfig, training_memory
from forge.train.settings import TrainSettings, learning_rate

CHECKPOINT_FORMAT = "forge-gpt-v1"
REPORT_EVERY = 5


class TokenFile:
    """Read-only view of one split's uint16 token file."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise JobFailed(
                f"The training set file {path.name} is missing. Make the training set again."
            )
        self.tokens = np.memmap(path, dtype="<u2", mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def window_for(train: TokenFile, ctx_len: int) -> int:
    window = min(ctx_len, len(train) - 1)
    if window < 2:
        raise JobFailed("This training set has too few tokens to train on. Add more text.")
    return window


def train_batch(
    train: TokenFile, window: int, seed: int, step: int, part: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, step, part])
    starts = rng.integers(0, len(train) - window, size=size)
    rows = np.stack([np.asarray(train.tokens[s : s + window + 1], dtype=np.int64) for s in starts])
    return rows[:, :-1], rows[:, 1:]


def validation_batches(validation: TokenFile, window: int, max_tokens: int, size: int):
    window = min(window, len(validation) - 1)
    if window < 2:
        return
    count = max(1, min(max_tokens // window, (len(validation) - 1) // window))
    starts = [i * window for i in range(count)]
    for first in range(0, len(starts), size):
        chunk = starts[first : first + size]
        rows = np.stack(
            [np.asarray(validation.tokens[s : s + window + 1], dtype=np.int64) for s in chunk]
        )
        yield rows[:, :-1], rows[:, 1:]


def free_card_memory() -> int | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def pick_micro_batch(config: ModelConfig, batch_size: int, device: str) -> int:
    if device != "cuda":
        return min(batch_size, 8) if batch_size % min(batch_size, 8) == 0 else 1
    free = free_card_memory() or 0
    for size in range(batch_size, 0, -1):
        if batch_size % size:
            continue
        if sum(training_memory(config, size).values()) <= free * 0.9:
            return size
    return 1


def pick_device(asked: str) -> str:
    if asked == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if asked == "cuda":
        raise JobFailed(
            "This run asks for the graphics card, but PyTorch can't see one on this computer. "
            "Choose The CPU, or install the CUDA build of PyTorch."
        )
    return "cpu"


def checkpoint_dir(run_id: uuid.UUID) -> Path:
    return get_settings().data_dir / "runs" / str(run_id)


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_checkpoint(checkpoint: Checkpoint) -> dict:
    path = get_settings().data_dir / checkpoint.uri
    if not path.is_file():
        raise JobFailed(f"The save point at step {checkpoint.step:,} is missing from the disk.")
    if checkpoint.sha256 and file_sha256(path) != checkpoint.sha256:
        raise JobFailed(
            f"The save point at step {checkpoint.step:,} changed on disk since it was saved, "
            "so it can't be trusted. Restore it from a backup or start a new run."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def latest_checkpoint(session: Session, run_id: uuid.UUID) -> Checkpoint | None:
    return session.scalar(
        select(Checkpoint).where(Checkpoint.run_id == run_id).order_by(Checkpoint.step.desc())
    )


def _write_checkpoint(path: Path, state: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def save_point(
    run_id: uuid.UUID,
    step: int,
    state: dict,
    loss: float | None,
    val_loss: float | None,
) -> Checkpoint:
    path = checkpoint_dir(run_id) / f"step-{step:08d}.pt"
    sha = _write_checkpoint(path, state)
    relative = path.relative_to(get_settings().data_dir).as_posix()
    with Session(get_engine()) as session:
        row = session.scalar(
            select(Checkpoint).where(Checkpoint.run_id == run_id, Checkpoint.step == step)
        )
        if row is None:
            row = Checkpoint(run_id=run_id, step=step, uri=relative)
            session.add(row)
        row.loss, row.val_loss, row.uri, row.sha256 = loss, val_loss, relative, sha
        row.size_bytes = path.stat().st_size
        row.metrics = {"tokens_seen": state["tokens_seen"]}
        session.flush()
        rows = list(session.scalars(select(Checkpoint).where(Checkpoint.run_id == run_id)))
        scored = [r for r in rows if r.val_loss is not None and math.isfinite(r.val_loss)]
        best = min(scored, key=lambda r: (r.val_loss, -r.step)) if scored else None
        latest = max(rows, key=lambda r: r.step)
        for other in rows:
            other.is_best = best is not None and other.id == best.id
            if other.id not in {latest.id, best.id if best else None}:
                (get_settings().data_dir / other.uri).unlink(missing_ok=True)
                session.delete(other)
        session.commit()
        session.refresh(row)
        return row


def _set_run(run_id: uuid.UUID, **values) -> None:
    with Session(get_engine()) as session:
        run = session.get(Run, run_id)
        if run is not None:
            for key, value in values.items():
                setattr(run, key, value)
            session.commit()


def _evaluate(
    model: GPT,
    validation: TokenFile,
    window: int,
    settings: TrainSettings,
    size: int,
    device: str,
    dtype: torch.dtype | None,
) -> float | None:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in validation_batches(validation, window, settings.eval_tokens, size):
            xs, ys = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            with torch.autocast(
                device_type=device, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                _, loss = model(xs, ys)
            assert loss is not None
            total += float(loss) * ys.numel()
            count += ys.numel()
    model.train()
    return total / count if count else None


def _nan(step: int) -> JobFailed:
    return JobFailed(
        f"The training loss became NaN or infinite at step {step:,}, so the run stopped. Its last "
        "good save point is kept. Start a new run with a lower learning rate."
    )


def _oom(step: int, micro: int) -> JobFailed:
    smaller = max(1, micro // 2)
    return JobFailed(
        f"The graphics card ran out of memory at step {step:,} with {micro} sequences per pass. "
        f"Set Sequences per pass to {smaller} under Advanced settings and start a new run, or "
        "resume this one after closing other programs that use the card."
    )


def _disk_full(step: int) -> JobFailed:
    return JobFailed(
        f"The disk filled up while saving at step {step:,}. Free some space, then Resume: the "
        "last good save point is kept."
    )


def train(job_id: uuid.UUID) -> dict:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        run_id = uuid.UUID(job.payload["run_id"])
        resume = bool(job.payload.get("resume"))
        run = session.get(Run, run_id)
        if run is None:
            raise JobFailed("This run was deleted.")
        model_row = session.get(Model, run.model_id)
        build = session.get(DatasetBuild, run.build_id) if run.build_id else None
        if model_row is None:
            raise JobFailed("This run's model was deleted.")
        if build is None or build.status != "ready" or not build.manifest_uri:
            raise JobFailed("This run's training set is missing or not ready.")
        settings = TrainSettings.model_validate(run.hyperparams.get("settings", {}))
        config = ModelConfig(
            n_layer=model_row.n_layer,
            d_model=model_row.d_model,
            n_head=model_row.n_head,
            ctx_len=model_row.ctx_len,
            vocab_size=model_row.vocab_size,
            dropout=model_row.dropout,
        )
        seed, asked_device = run.seed, run.device
        manifest_path = get_settings().data_dir / build.manifest_uri
        checkpoint = latest_checkpoint(session, run_id) if resume else None
        if resume and checkpoint is None:
            raise JobFailed("There is no save point to resume from. Start a new run instead.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("vocab_size", config.vocab_size) > config.vocab_size:
        raise JobFailed(
            "The training set uses a bigger vocabulary than the model reads. Make the model with "
            "the same vocabulary as the training set."
        )
    folder = manifest_path.parent
    train_file = TokenFile(folder / manifest["splits"]["train"]["file"])
    validation_file = TokenFile(folder / manifest["splits"]["validation"]["file"])
    window = window_for(train_file, config.ctx_len)

    device = pick_device(asked_device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None
    micro = settings.micro_batch_size or pick_micro_batch(config, settings.batch_size, device)
    if settings.batch_size % micro:
        micro = 1
    accumulation = settings.batch_size // micro

    torch.manual_seed(seed)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(settings.learning_rate, settings.weight_decay)
    start_step, tokens_seen = 0, 0
    if checkpoint is not None:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, tokens_seen = state["step"], state["tokens_seen"]
        torch.set_rng_state(state["rng"]["torch"])
        if device == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(state["rng"]["cuda"])
        with Session(get_engine()) as session:
            session.execute(
                delete(MetricPoint).where(
                    MetricPoint.run_id == run_id, MetricPoint.step > start_step
                )
            )
            session.commit()

    _set_run(
        run_id,
        status="running",
        error=None,
        device=device,
        started_at=datetime.now(UTC),
        completed_at=None,
        metrics={"window": window, "micro_batch_size": micro, "accumulation": accumulation},
    )
    model.train()
    step = start_step
    last_loss: float | None = None
    last_val: float | None = None

    def state_now() -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "config": asdict(config),
            "settings": settings.model_dump(),
            "seed": seed,
            "tokenizer_id": manifest.get("tokenizer_id"),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
            },
        }

    def save(val_loss: float | None) -> None:
        try:
            save_point(run_id, step, state_now(), last_loss, val_loss)
        except OSError as problem:
            if problem.errno == errno.ENOSPC:
                raise _disk_full(step) from problem
            raise

    try:
        with MetricBuffer(open_session, run_id, batch_size=20) as buffer:
            began = time.perf_counter()
            for step in range(start_step, settings.max_steps):
                if step % REPORT_EVERY == 0:
                    report(job_id, step, settings.max_steps, "Training")
                rate = learning_rate(step, settings)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for part in range(accumulation):
                    x, y = train_batch(train_file, window, seed, step, part, micro)
                    xs, ys = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                    with torch.autocast(
                        device_type=device, dtype=dtype or torch.float32, enabled=dtype is not None
                    ):
                        _, loss = model(xs, ys)
                    assert loss is not None
                    (loss / accumulation).backward()
                    total += float(loss.detach()) / accumulation
                if not math.isfinite(total):
                    raise _nan(step + 1)
                if settings.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                optimizer.step()
                step_tokens = settings.batch_size * window
                tokens_seen += step_tokens
                last_loss = total
                done = step + 1
                elapsed = max(time.perf_counter() - began, 1e-9)
                speed = (done - start_step) * step_tokens / elapsed
                val = None
                if done % settings.eval_every == 0 or done == settings.max_steps:
                    val = _evaluate(model, validation_file, window, settings, micro, device, dtype)
                    if val is not None and not math.isfinite(val):
                        raise _nan(done)
                    last_val = val if val is not None else last_val
                buffer.record(step=done, loss=total, val_loss=val, learning_rate=rate, speed=speed)
                if done % REPORT_EVERY == 0 or val is not None:
                    _set_run(
                        run_id,
                        current_step=done,
                        tokens_seen=tokens_seen,
                        loss=total,
                        eval_loss=last_val,
                        tokens_per_sec=speed,
                    )
                step = done
                if done % settings.save_every == 0 or done == settings.max_steps:
                    save(val)
    except Canceled:
        if step > start_step:
            save(None)
        _set_run(run_id, current_step=step, tokens_seen=tokens_seen)
        raise
    except torch.cuda.OutOfMemoryError as problem:
        raise _oom(step + 1, micro) from problem
    except RuntimeError as problem:
        if "out of memory" in str(problem).lower():
            raise _oom(step + 1, micro) from problem
        raise
    _set_run(run_id, current_step=step, tokens_seen=tokens_seen, loss=last_loss, eval_loss=last_val)
    report(job_id, step, settings.max_steps, "Finished")
    return {
        "run_id": str(run_id),
        "steps": step,
        "tokens_seen": tokens_seen,
        "final_loss": last_loss,
        "final_val_loss": last_val,
    }
