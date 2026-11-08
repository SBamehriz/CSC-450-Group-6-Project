"""Benchmarks through EleutherAI's lm-evaluation-harness, gated by model size.

Below a size a benchmark only measures chance, so it is refused with that
reason instead of shown as a bad score. Every score comes with its 95%
interval and the chance level next to it.
"""

import math
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import JobFailed, report
from forge.models import EvalResult, Job, Model
from forge.version import current_commit

Z95 = 1.96
SMALLEST = 100_000_000
ABOUT_A_BILLION = 900_000_000


@dataclass(frozen=True)
class Benchmark:
    task: str
    name: str
    chance: float
    floor: int


BENCHMARKS = (
    Benchmark("hellaswag", "HellaSwag", 0.25, SMALLEST),
    Benchmark("piqa", "PIQA", 0.5, SMALLEST),
    Benchmark("winogrande", "WinoGrande", 0.5, SMALLEST),
    Benchmark("lambada_openai", "LAMBADA", 0.0, SMALLEST),
    Benchmark("arc_easy", "ARC-Easy", 0.25, SMALLEST),
    Benchmark("openbookqa", "OpenBookQA", 0.25, SMALLEST),
    Benchmark("boolq", "BoolQ", 0.5, SMALLEST),
    Benchmark("copa", "COPA", 0.5, SMALLEST),
    Benchmark("commonsense_qa", "CommonsenseQA", 0.2, SMALLEST),
    Benchmark("sciq", "SciQ", 0.25, SMALLEST),
    Benchmark("arc_challenge", "ARC-Challenge", 0.25, ABOUT_A_BILLION),
    Benchmark("mmlu", "MMLU", 0.25, ABOUT_A_BILLION),
)
BY_TASK = {one.task: one for one in BENCHMARKS}


def size_words(parameters: int) -> str:
    if parameters >= ABOUT_A_BILLION:
        return f"{parameters / 1e9:.1f} billion"
    return f"{parameters / 1e6:.0f} million"


def refusal(benchmark: Benchmark, parameters: int) -> str | None:
    if parameters >= benchmark.floor:
        return None
    floor = "about 1 billion" if benchmark.floor == ABOUT_A_BILLION else "100 million"
    return (
        f"{benchmark.name} needs a model of at least {floor} parameters, and this one has "
        f"{size_words(parameters)}. Below that, scores are chance-level noise. Compare loss "
        "and the prompt answers instead."
    )


def plan(parameters: int, asked: list[str] | None) -> tuple[list[Benchmark], list[dict]]:
    chosen = [BY_TASK[task] for task in asked] if asked else list(BENCHMARKS)
    allowed, refused = [], []
    for benchmark in chosen:
        reason = refusal(benchmark, parameters)
        if reason:
            refused.append({"task": benchmark.task, "name": benchmark.name, "reason": reason})
        else:
            allowed.append(benchmark)
    return allowed, refused


def _score(task: str, numbers: dict) -> tuple[str, float, float | None] | None:
    # length-normalised accuracy where the harness gives it, plain accuracy otherwise
    for metric in ("acc_norm", "acc"):
        value = numbers.get(f"{metric},none")
        if isinstance(value, int | float):
            error = numbers.get(f"{metric}_stderr,none")
            stderr = float(error) if isinstance(error, int | float) else None
            return metric, float(value), stderr
    return None


def scores_from(results: dict, allowed: list[Benchmark]) -> list[dict]:
    rows = []
    samples = results.get("n-samples", {})
    for benchmark in allowed:
        numbers = results.get("results", {}).get(benchmark.task)
        found = _score(benchmark.task, numbers) if numbers else None
        if found is None:
            continue
        metric, value, stderr = found
        margin = Z95 * stderr if stderr is not None and math.isfinite(stderr) else None
        rows.append(
            {
                "task": benchmark.task,
                "name": benchmark.name,
                "metric": metric,
                "score": value,
                "low": max(0.0, value - margin) if margin is not None else None,
                "high": min(1.0, value + margin) if margin is not None else None,
                "chance": benchmark.chance,
                "questions": (samples.get(benchmark.task) or {}).get("effective"),
            }
        )
    return rows


def harness_version() -> str | None:
    try:
        import lm_eval
    except ImportError:
        return None
    return f"lm-eval {lm_eval.__version__}"


def evaluate(loaded, tasks: list[str], limit: int | None) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as problem:
        raise JobFailed(
            "Benchmarks need the lm-evaluation-harness. Install the bench extra first."
        ) from problem

    # the harness's own signatures confuse type checkers, so call through Any
    make: Any = HFLM
    simple: Any = lm_eval.simple_evaluate
    harness_model = make(
        pretrained=loaded.model, tokenizer=loaded.tokenizer, batch_size=8, device=loaded.device
    )
    return simple(model=harness_model, tasks=tasks, limit=limit, log_samples=False) or {}


def run(job_id: uuid.UUID) -> dict:
    from forge.measuring import load

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        payload = dict(job.payload)
        model_id = uuid.UUID(payload["model_id"])
        row = session.get(Model, model_id)
        if row is None:
            raise JobFailed("This model was deleted.")
        parameters = row.param_count
    allowed, refused = plan(parameters, payload.get("tasks"))
    if not allowed:
        raise JobFailed(refused[0]["reason"] if refused else "Choose at least one benchmark.")
    report(job_id, 0, 2, "Loading the model")
    checkpoint = payload.get("checkpoint_id")
    loaded = load(
        model_id, uuid.UUID(checkpoint) if checkpoint else None, payload.get("device", "auto")
    )
    report(job_id, 1, 2, f"Answering {len(allowed)} benchmark{'s' if len(allowed) > 1 else ''}")
    try:
        results = evaluate(loaded, [one.task for one in allowed], payload.get("limit"))
    except (ConnectionError, OSError) as problem:
        raise JobFailed(
            "The benchmark questions could not be downloaded from the Hugging Face hub "
            f"({problem}). Check the internet connection and Retry."
        ) from problem
    return {
        "model_id": str(model_id),
        "model_name": loaded.name,
        "parameters": parameters,
        "checkpoint_id": str(loaded.checkpoint_id) if loaded.checkpoint_id else None,
        "scores": scores_from(results or {}, allowed),
        "refused": refused,
        "limit": payload.get("limit"),
        "harness_version": harness_version(),
    }


def finish(job_id: uuid.UUID, status: str, message: str | None) -> None:
    if status != "completed":
        return
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if not job or not job.result:
            return
        res = job.result
        model_id_str = res.get("model_id")
        ckpt_id_str = res.get("checkpoint_id")
        model_id = uuid.UUID(model_id_str) if model_id_str else None
        ckpt_id = uuid.UUID(ckpt_id_str) if ckpt_id_str else None
        model_name = res.get("model_name") or "unknown"
        code_ver = current_commit()
        for row in res.get("scores", []):
            session.add(
                EvalResult(
                    id=uuid.uuid4(),
                    result_type="benchmark",
                    model_id=model_id,
                    model_name=model_name,
                    checkpoint_id=ckpt_id,
                    evaluation_id=job_id,
                    settings={
                        "limit": res.get("limit"),
                        "harness_version": res.get("harness_version"),
                        "metric": row.get("metric"),
                        "chance": row.get("chance"),
                    },
                    code_version=code_ver,
                    benchmark_name=row.get("name") or row.get("task"),
                    benchmark_score=row.get("score"),
                    benchmark_scores=dict(row),
                )
            )
        session.commit()
