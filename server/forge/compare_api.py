import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge import harness
from forge.db import get_session
from forge.errors import ApiError
from forge.jobs_api import queue_task
from forge.measuring import pick_checkpoint
from forge.models import BpbEvaluation, Job, Model, PromptSet, Run
from forge.schemas import (
    BenchmarkCreate,
    BenchmarkInfo,
    BenchmarkPlanOut,
    BenchmarkRefusal,
    BenchmarkResult,
    CompareModel,
    CompareOut,
    JobOut,
    PromptCompareCreate,
    PromptCompareResult,
)

router = APIRouter(tags=["comparing"])


def _model(session: Session, model_id: uuid.UUID) -> Model:
    model = session.get(Model, model_id)
    if model is None:
        raise ApiError(404, "not_found", "Model not found.")
    return model


def _measurable(session: Session, model: Model) -> None:
    if model.source == "forge" and pick_checkpoint(session, model.id) is None:
        raise ApiError(
            409,
            "not_trained",
            f"{model.name} has not been trained yet, so there is nothing to ask.",
        )


@router.get("/benchmarks/plan", response_model=BenchmarkPlanOut)
def benchmark_plan(model_id: uuid.UUID, session: Session = Depends(get_session)):
    model = _model(session, model_id)
    allowed, refused = harness.plan(model.param_count, None)
    return BenchmarkPlanOut(
        parameters=model.param_count,
        allowed=[BenchmarkInfo(task=b.task, name=b.name, chance=b.chance) for b in allowed],
        refused=[BenchmarkRefusal(**row) for row in refused],
    )


@router.post("/benchmarks", response_model=JobOut, status_code=202)
def start_benchmark(payload: BenchmarkCreate, session: Session = Depends(get_session)) -> JobOut:
    model = _model(session, payload.model_id)
    unknown = [task for task in payload.tasks or [] if task not in harness.BY_TASK]
    if unknown:
        raise ApiError(422, "unknown_benchmark", f"There is no benchmark called {unknown[0]!r}.")
    allowed, refused = harness.plan(model.param_count, payload.tasks)
    if not allowed:
        raise ApiError(409, "too_small", refused[0]["reason"])
    _measurable(session, model)
    return queue_task(
        session,
        "benchmark",
        {
            "model_id": str(model.id),
            "checkpoint_id": str(payload.checkpoint_id) if payload.checkpoint_id else None,
            "tasks": payload.tasks,
            "limit": payload.limit,
            "device": payload.device,
            "target_name": f"Benchmarks, {model.name}",
        },
    )


@router.post("/prompt-comparisons", response_model=JobOut, status_code=202)
def start_prompt_comparison(
    payload: PromptCompareCreate, session: Session = Depends(get_session)
) -> JobOut:
    first, second = (_model(session, model_id) for model_id in payload.model_ids)
    if first.id == second.id:
        raise ApiError(422, "same_model", "Choose two different models.")
    prompts = list(payload.prompts)
    if payload.prompt_set_id:
        saved = session.get(PromptSet, payload.prompt_set_id)
        if saved is None:
            raise ApiError(404, "not_found", "That prompt set was deleted.")
        # a copy, so editing the set later does not change this comparison
        prompts = [str(prompt) for prompt in saved.prompts][:20]
    if not prompts:
        raise ApiError(422, "no_prompts", "Write at least one prompt or choose a prompt set.")
    for model in (first, second):
        _measurable(session, model)
    return queue_task(
        session,
        "prompt_compare",
        {
            "model_ids": [str(first.id), str(second.id)],
            "prompts": prompts,
            "settings": {
                "temperature": payload.temperature,
                "top_k": payload.top_k,
                "max_new_tokens": payload.max_new_tokens,
                "seed": payload.seed,
            },
            "device": payload.device,
            "target_name": f"{first.name} and {second.name}",
        },
    )


def _finished(session: Session, kind: str) -> list[Job]:
    return list(
        session.scalars(
            select(Job)
            .where(Job.job_type == kind, Job.status == "completed")
            .order_by(Job.completed_at.desc())
        )
    )


def _benchmark(session: Session, model_id: uuid.UUID) -> BenchmarkResult | None:
    for job in _finished(session, "benchmark"):
        if job.result.get("model_id") == str(model_id):
            checkpoint = job.result.get("checkpoint_id")
            return BenchmarkResult(
                job_id=job.id,
                finished_at=job.completed_at,
                checkpoint_id=uuid.UUID(checkpoint) if checkpoint else None,
                limit=job.result.get("limit"),
                harness_version=job.result.get("harness_version"),
                scores=job.result.get("scores", []),
                refused=job.result.get("refused", []),
            )
    return None


def _side(session: Session, model: Model) -> CompareModel:
    best = session.scalar(
        select(func.min(Run.eval_loss)).where(Run.model_id == model.id, Run.eval_loss.is_not(None))
    )
    seen = session.scalar(select(func.max(Run.tokens_seen)).where(Run.model_id == model.id))
    # the newest finished bits per byte measurement for this model
    bits = session.scalar(
        select(BpbEvaluation.bpb)
        .where(
            BpbEvaluation.model_id == model.id,
            BpbEvaluation.status == "completed",
            BpbEvaluation.bpb.is_not(None),
        )
        .order_by(BpbEvaluation.completed_at.desc())
    )
    hub = model.config.get("hub") or {}
    return CompareModel(
        id=model.id,
        name=model.name,
        source=model.source,
        parameters=model.param_count,
        vocab_size=model.vocab_size,
        ctx_len=model.ctx_len,
        license=hub.get("license"),
        tokens_seen=seen if model.source == "forge" else None,
        best_val_loss=best,
        bits_per_byte=bits,
        benchmark=_benchmark(session, model.id),
    )


@router.get("/compare", response_model=CompareOut)
def compare(
    model_ids: list[uuid.UUID] = Query(min_length=1, max_length=2),
    session: Session = Depends(get_session),
) -> CompareOut:
    models = [_model(session, model_id) for model_id in model_ids]
    wanted = {str(model.id) for model in models}
    prompt_runs = []
    for job in _finished(session, "prompt_compare"):
        if len(wanted) == 2 and set(job.payload.get("model_ids", [])) != wanted:
            continue
        if len(wanted) == 1 and not wanted & set(job.payload.get("model_ids", [])):
            continue
        prompt_runs.append(
            PromptCompareResult(
                job_id=job.id,
                finished_at=job.completed_at,
                settings=job.result.get("settings", {}),
                model_names=[side["name"] for side in job.result.get("models", [])],
                rows=job.result.get("rows", []),
            )
        )
    return CompareOut(models=[_side(session, model) for model in models], prompt_runs=prompt_runs)
