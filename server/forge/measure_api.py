from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from forge import gpu
from forge.db import get_session
from forge.errors import ApiError
from forge.jobs_api import queue_task
from forge.models import SpeedTest
from forge.schemas import (
    EstimateOut,
    EstimateRequest,
    JobOut,
    ShapeIn,
    SpeedTestCreate,
    SpeedTestOut,
)
from forge.train.presets import PRESETS, ModelConfig, count_params, training_memory

router = APIRouter(tags=["measuring"])

# leave a little room on the card, as training does when it picks a batch
USABLE_SHARE = 0.9
PASS_SIZES = (64, 32, 16, 8, 4, 2, 1)


def _config(preset: str | None, shape: ShapeIn | None) -> ModelConfig:
    if preset:
        if preset not in PRESETS:
            raise ApiError(404, "not_found", f"There is no size called {preset!r}.")
        return PRESETS[preset]
    if shape is None:
        raise ApiError(422, "invalid_shape", "Choose a size or give a shape.")
    try:
        return ModelConfig(**shape.model_dump())
    except ValueError as problem:
        raise ApiError(422, "invalid_shape", str(problem)) from problem


def _shape(config: ModelConfig) -> dict:
    return {
        "n_layer": config.n_layer,
        "d_model": config.d_model,
        "n_head": config.n_head,
        "ctx_len": config.ctx_len,
        "vocab_size": config.vocab_size,
    }


def work_per_token(config: ModelConfig) -> float:
    # forward and backward, plus attention over the window
    return 6 * count_params(config) + 12 * config.n_layer * config.d_model * config.ctx_len


@router.post("/speed-tests", response_model=JobOut, status_code=202)
def start_speed_test(payload: SpeedTestCreate, session: Session = Depends(get_session)) -> JobOut:
    config = _config(payload.preset, payload.shape)
    label = payload.preset or f"{count_params(config) / 1e6:.1f}M"
    return queue_task(
        session,
        "speed_test",
        {
            "preset": payload.preset,
            "shape": _shape(config),
            "sequences_per_step": payload.sequences_per_step,
            "device": payload.device,
            "target_name": f"Speed test, {label}",
        },
    )


@router.get("/speed-tests", response_model=list[SpeedTestOut])
def list_speed_tests(session: Session = Depends(get_session)) -> list[SpeedTestOut]:
    rows = session.scalars(select(SpeedTest).order_by(SpeedTest.created_at.desc()).limit(50))
    return [SpeedTestOut.model_validate(row) for row in rows]


def _free_bytes() -> int | None:
    cards = gpu.read().cards
    return max(card.free_bytes for card in cards) if cards else None


@router.post("/estimate", response_model=EstimateOut)
def estimate(payload: EstimateRequest, session: Session = Depends(get_session)) -> EstimateOut:
    config = _config(payload.preset, payload.shape)
    shape = _shape(config)
    tests = list(session.scalars(select(SpeedTest).order_by(SpeedTest.created_at.desc())))
    exact = next(
        (
            row
            for row in tests
            if row.shape == shape and row.sequences_per_step == payload.sequences_per_step
        ),
        None,
    )
    source, chosen, speed = "none", None, None
    if exact is not None:
        source, chosen, speed = "measured", exact, exact.tokens_per_sec
    elif tests:
        # the newest test on the graphics card if there is one, scaled by the work
        chosen = next((row for row in tests if row.device == "cuda"), tests[0])
        other = ModelConfig(**chosen.shape)
        speed = chosen.tokens_per_sec * work_per_token(other) / work_per_token(config)
        source = "scaled"

    needed = sum(training_memory(config, payload.sequences_per_step).values())
    free = _free_bytes()
    fits = None if free is None else needed <= free * USABLE_SHARE
    suggestion = None
    if fits is False:
        assert free is not None
        smaller = next(
            (
                size
                for size in PASS_SIZES
                if size < payload.sequences_per_step
                and sum(training_memory(config, size).values()) <= free * USABLE_SHARE
            ),
            None,
        )
        suggestion = (
            f"Use {smaller} sequences per pass. The run still sees the same batch, in more passes."
            if smaller
            else "Even one sequence per pass does not fit in the free memory. Close other "
            "programs that use the card, or choose a smaller size."
        )
    return EstimateOut(
        parameters=count_params(config),
        train_tokens=payload.train_tokens,
        speed_source=source,
        speed_test_id=chosen.id if chosen else None,
        device_name=chosen.device_name if chosen else None,
        tokens_per_sec=speed,
        hours=payload.train_tokens / speed / 3600 if speed else None,
        memory_bytes=needed,
        free_bytes=free,
        fits=fits,
        suggestion=suggestion,
    )
