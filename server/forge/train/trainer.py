import errno
import gc
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

from forge import chatformat
from forge.config import get_settings
from forge.db import get_engine, open_session
from forge.gpu import free_gpu_memory
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
    train: TokenFile, window: int, seed: int, step: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    # the whole step's batch, from the seed and the step only, however it is cut into passes
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, len(train) - window, size=size)
    rows = np.stack([np.asarray(train.tokens[s : s + window + 1], dtype=np.int64) for s in starts])
    return rows[:, :-1], rows[:, 1:]


def learned_only(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # a chat fine-tune learns only the assistant's words; the loss skips -1
    rows = np.concatenate([x[:, :1], y], axis=1)
    return np.where(chatformat.learned(rows)[:, 1:], y, -1)


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
    # the smallest card's: a pass is split evenly across the cards
    return min(int(torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count()))


def pick_micro_batch(config: ModelConfig, batch_size: int, device: str, cards: int = 1) -> int:
    if device != "cuda":
        return min(batch_size, 8) if batch_size % min(batch_size, 8) == 0 else 1
    free = free_card_memory() or 0
    for size in range(batch_size, 0, -1):
        if batch_size % size:
            continue
        if sum(training_memory(config, math.ceil(size / cards)).values()) <= free * 0.9:
            return size
    return 1


def card_count(device: str) -> int:
    return torch.cuda.device_count() if device == "cuda" else 0


def real_bf16() -> bool:
    # emulated bf16 is slow on older cards
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older PyTorch has no such argument
        return torch.cuda.is_bf16_supported()


def _out_of_memory(problem: BaseException) -> bool:
    return isinstance(problem, torch.cuda.OutOfMemoryError) or (
        isinstance(problem, RuntimeError) and "out of memory" in str(problem).lower()
    )


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


def _time_is_up(run_id: uuid.UUID, began: float) -> bool:
    with Session(get_engine()) as session:
        run = session.get(Run, run_id)
        hours = run.hyperparams.get("stop_after_hours") if run else None
    return bool(hours) and time.perf_counter() - began > hours * 3600


def _evaluate(
    model: GPT,
    validation: TokenFile,
    window: int,
    settings: TrainSettings,
    size: int,
    device: str,
    dtype: torch.dtype | None,
    chat: bool = False,
) -> float | None:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in validation_batches(validation, window, settings.eval_tokens, size):
            y = learned_only(x, y) if chat else y
            learned = int((y != -1).sum())
            if not learned:
                continue
            xs, ys = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            with torch.autocast(
                device_type=device, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                _, loss = model(xs, ys)
            assert loss is not None
            total += float(loss) * learned
            count += learned
    model.train()
    return total / count if count else None


def _nan(step: int) -> JobFailed:
    return JobFailed(
        f"The training loss became NaN or infinite at step {step:,}, so the run stopped. Its last "
        "good save point is kept. Start a new run with a lower learning rate."
    )


def _oom(step: int, micro: int) -> JobFailed:
    if micro <= 1:
        return JobFailed(
            f"The graphics card ran out of memory at step {step:,} even with one sequence per "
            "pass. Choose a smaller model, or resume this one after closing other programs that "
            "use the card."
        )
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
    try:
        return _train(job_id)
    except (JobFailed, Canceled) as problem:
        # its traceback holds the model; drop it so the card memory can be freed
        cause: BaseException | None = problem
        while cause is not None:
            cause.__traceback__ = None
            cause = cause.__cause__ or cause.__context__
        raise
    finally:
        gc.collect()
        free_gpu_memory()


def _not_ready(session: Session, run: Run) -> str:
    # say why the text could not be made ready, when POST /api/train lined it up
    for job_id in run.hyperparams.get("prepare_jobs", []):
        step = session.get(Job, uuid.UUID(job_id))
        if step is not None and step.status in ("failed", "cancelled"):
            return f"The text for this run could not be made ready. {step.error or ''}".strip()
    return "This run's training set is missing or not ready."


def _train(job_id: uuid.UUID) -> dict:
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
            raise JobFailed(_not_ready(session, run))
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
        # a fine-tune starts from another model's save point
        start = None
        if run.init_checkpoint_id and not resume:
            start = session.get(Checkpoint, run.init_checkpoint_id)
            if start is None:
                raise JobFailed(
                    "The save point this fine-tune starts from was deleted. Start a new fine-tune."
                )
        chat = bool(run.hyperparams.get("chat"))

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
    dtype = torch.bfloat16 if device == "cuda" and real_bf16() else None
    cards = card_count(device)
    micro = settings.micro_batch_size or pick_micro_batch(
        config, settings.batch_size, device, max(1, cards)
    )
    if settings.batch_size % micro:
        micro = 1
    accumulation = settings.batch_size // micro

    torch.manual_seed(seed)
    model = GPT(config).to(device)
    # two or more cards split each pass; save points keep the plain model's weights
    runner: torch.nn.Module = model
    if cards > 1:
        model.autocast_dtype = dtype
        runner = torch.nn.DataParallel(model)
    optimizer = model.configure_optimizer(settings.learning_rate, settings.weight_decay)
    start_step, tokens_seen = 0, 0
    if start is not None:
        try:
            # only the weights: a fine-tune starts with a fresh optimizer
            model.load_state_dict(load_checkpoint(start)["model"])
        except RuntimeError as problem:
            raise JobFailed(
                "The save point this fine-tune starts from does not match the model's shape."
            ) from problem
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

    facts = {
        "window": window,
        "micro_batch_size": micro,
        "accumulation": accumulation,
        "cards": cards,
        "card_names": [torch.cuda.get_device_name(i) for i in range(cards)],
    }
    _set_run(
        run_id,
        status="running",
        error=None,
        device=device,
        started_at=datetime.now(UTC),
        completed_at=None,
        metrics=dict(facts),
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
                    if _time_is_up(run_id, began):
                        _set_run(run_id, metrics={**facts, "time_limit_reached": True})
                        raise Canceled()
                rate = learning_rate(step, settings)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                batch_x, batch_y = train_batch(train_file, window, seed, step, settings.batch_size)
                if chat:
                    batch_y = learned_only(batch_x, batch_y)
                learned = int((batch_y != -1).sum())
                total = 0.0
                while learned:
                    try:
                        for first in range(0, settings.batch_size, micro):
                            x, y = batch_x[first : first + micro], batch_y[first : first + micro]
                            share = int((y != -1).sum())
                            if not share:
                                continue
                            xs, ys = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                            with torch.autocast(
                                device_type=device,
                                dtype=dtype or torch.float32,
                                enabled=dtype is not None,
                            ):
                                _, loss = runner(xs, ys)
                            assert loss is not None
                            # one loss per card when the pass was split
                            loss = loss.mean()
                            (loss * share / learned).backward()
                            total += float(loss.detach()) * share / learned
                        break
                    except RuntimeError as problem:
                        if not _out_of_memory(problem) or micro == 1:
                            raise
                    # a small card: the same step again, in passes half the size
                    optimizer.zero_grad(set_to_none=True)
                    total, micro = 0.0, micro // 2
                    gc.collect()
                    free_gpu_memory()
                    facts["micro_batch_size"] = micro
                    _set_run(run_id, metrics=dict(facts))
                if not learned:
                    # nothing to learn in this step, so no update
                    total = last_loss
                elif not math.isfinite(total):
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
                    val = _evaluate(
                        model, validation_file, window, settings, micro, device, dtype, chat
                    )
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
