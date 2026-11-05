import statistics
import time
import uuid

from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import JobFailed, report
from forge.models import Job, SpeedTest
from forge.train.presets import PRESETS, ModelConfig

WARMUP_STEPS = 3
MOST_STEPS = 30
TIME_BUDGET_S = 60.0


def shape_of(payload: dict) -> tuple[str | None, ModelConfig]:
    preset = payload.get("preset")
    if preset:
        if preset not in PRESETS:
            raise JobFailed(f"There is no size called {preset!r}.")
        return preset, PRESETS[preset]
    shape = payload["shape"]
    return None, ModelConfig(
        n_layer=shape["n_layer"],
        d_model=shape["d_model"],
        n_head=shape["n_head"],
        ctx_len=shape["ctx_len"],
        vocab_size=shape.get("vocab_size", 16384),
    )


def run(job_id: uuid.UUID) -> dict:
    try:
        import torch

        from forge.train.model import GPT
        from forge.train.trainer import pick_device
    except ImportError as problem:
        raise JobFailed("The speed test needs PyTorch. Install the train extra first.") from problem

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        payload = dict(job.payload)
    preset, config = shape_of(payload)
    sequences = int(payload.get("sequences_per_step", 4))
    device = pick_device(payload.get("device", "auto"))
    name = torch.cuda.get_device_name(0) if device == "cuda" else "The CPU"

    torch.manual_seed(0)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(1e-4, 0.1)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tokens = torch.randint(0, config.vocab_size, (sequences, config.ctx_len + 1), device=device)
    x, y = tokens[:, :-1], tokens[:, 1:]

    times: list[float] = []
    began = time.perf_counter()
    try:
        for step in range(WARMUP_STEPS + MOST_STEPS):
            report(job_id, step, WARMUP_STEPS + MOST_STEPS, "Timing training steps")
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                _, loss = model(x, y)
            assert loss is not None
            loss.backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= WARMUP_STEPS:
                times.append(time.perf_counter() - start)
            if len(times) >= 3 and time.perf_counter() - began > TIME_BUDGET_S:
                break
    except torch.cuda.OutOfMemoryError as problem:
        raise JobFailed(
            f"This size with {sequences} sequences per step does not fit on {name}. "
            "Try fewer sequences per step or a smaller size."
        ) from problem

    step_s = statistics.median(times)
    tokens_per_sec = sequences * config.ctx_len / step_s
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    with Session(get_engine()) as session:
        row = SpeedTest(
            preset=preset,
            shape={
                "n_layer": config.n_layer,
                "d_model": config.d_model,
                "n_head": config.n_head,
                "ctx_len": config.ctx_len,
                "vocab_size": config.vocab_size,
            },
            sequences_per_step=sequences,
            device=device,
            device_name=name,
            tokens_per_sec=tokens_per_sec,
            peak_memory_bytes=peak,
            steps=len(times),
        )
        session.add(row)
        session.commit()
        speed_id = str(row.id)
    return {
        "speed_test_id": speed_id,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_bytes": peak,
        "device_name": name,
    }
