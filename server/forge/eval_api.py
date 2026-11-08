"""API endpoints for named prompt sets and model evaluations."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.core.metrics import ImportedModelAdapter, _get_validation_file_info
from forge.db import get_session
from forge.errors import ApiError
from forge.models import (
    BpbEvaluation,
    Checkpoint,
    Dataset,
    EvalResult,
    Evaluation,
    Job,
    Model,
    PromptSet,
    Run,
)
from forge.schemas import (
    BpbEvaluationCreate,
    BpbEvaluationOut,
    EvalResultOut,
    EvaluationCreate,
    EvaluationOut,
    PromptSetCreate,
    PromptSetOut,
    PromptSetUpdate,
)

router = APIRouter(tags=["evaluations"])


def _validate_prompts(prompts: list[Any]) -> list[str]:
    """Validate prompt list following project conventions."""
    if not prompts or not isinstance(prompts, list) or len(prompts) == 0:
        raise ApiError(400, "empty_prompt_set", "A prompt set must contain at least one prompt.")

    clean_prompts: list[str] = []
    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise ApiError(400, "invalid_prompt", f"Prompt #{idx + 1} must be a text string.")
        cleaned = prompt.strip()
        if not cleaned:
            raise ApiError(400, "invalid_prompt", f"Prompt #{idx + 1} cannot be empty.")
        if len(cleaned) > 4000:
            raise ApiError(
                400,
                "invalid_prompt",
                f"Prompt #{idx + 1} exceeds maximum length of 4000 characters.",
            )
        clean_prompts.append(cleaned)
    return clean_prompts


def _prompt_set_to_out(ps: PromptSet) -> PromptSetOut:
    return PromptSetOut(
        id=ps.id,
        name=ps.name,
        description=ps.description,
        prompts=ps.prompts,
        prompt_count=len(ps.prompts),
        created_at=ps.created_at,
        updated_at=ps.updated_at,
    )


def _evaluation_to_out(ev: Evaluation) -> EvaluationOut:
    return EvaluationOut(
        id=ev.id,
        name=ev.name,
        model_id=ev.model_id,
        model_name=ev.model_name,
        prompt_set_id=ev.prompt_set_id,
        prompt_set_name=ev.prompt_set_name,
        captured_prompts=ev.captured_prompts,
        results=ev.results,
        status=ev.status,
        error=ev.error,
        job_id=ev.job_id,
        started_at=ev.started_at,
        completed_at=ev.completed_at,
        created_at=ev.created_at,
    )


def _bpb_eval_to_out(ev: BpbEvaluation) -> BpbEvaluationOut:
    return BpbEvaluationOut(
        id=ev.id,
        name=ev.name,
        model_id=ev.model_id,
        model_name=ev.model_name,
        model_source=ev.model_source,
        checkpoint_id=ev.checkpoint_id,
        dataset_id=ev.dataset_id,
        dataset_name=ev.dataset_name,
        validation_file_hash=ev.validation_file_hash,
        tokenizer_identity=ev.tokenizer_identity,
        bpb=ev.bpb,
        total_nll=ev.total_nll,
        scored_bytes=ev.scored_bytes,
        scored_tokens=ev.scored_tokens,
        status=ev.status,
        error=ev.error,
        settings=ev.settings,
        job_id=ev.job_id,
        started_at=ev.started_at,
        completed_at=ev.completed_at,
        created_at=ev.created_at,
    )


# --- Prompt Set Endpoints ---


@router.get("/prompt-sets", response_model=list[PromptSetOut])
def list_prompt_sets(session: Session = Depends(get_session)) -> list[PromptSetOut]:
    """List all saved prompt sets, newest first."""
    records = (
        session.execute(select(PromptSet).order_by(PromptSet.created_at.desc())).scalars().all()
    )
    return [_prompt_set_to_out(ps) for ps in records]


@router.post("/prompt-sets", response_model=PromptSetOut, status_code=201)
def create_prompt_set(
    payload: PromptSetCreate,
    session: Session = Depends(get_session),
) -> PromptSetOut:
    """Create a persistent named prompt set containing an ordered collection of prompts."""
    clean_name = payload.name.strip()
    if not clean_name:
        raise ApiError(400, "invalid_name", "Prompt set name cannot be empty.")
    if len(clean_name) > 64:
        raise ApiError(400, "invalid_name", "Prompt set name cannot exceed 64 characters.")

    existing = session.scalar(select(PromptSet).where(PromptSet.name == clean_name))
    if existing:
        raise ApiError(409, "name_conflict", f"A prompt set named '{clean_name}' already exists.")

    clean_prompts = _validate_prompts(payload.prompts)

    now = datetime.now(UTC)
    prompt_set = PromptSet(
        id=uuid.uuid4(),
        name=clean_name,
        description=payload.description.strip(),
        prompts=clean_prompts,
        created_at=now,
        updated_at=now,
    )
    session.add(prompt_set)
    session.commit()
    session.refresh(prompt_set)
    return _prompt_set_to_out(prompt_set)


@router.get("/prompt-sets/{prompt_set_id}", response_model=PromptSetOut)
def get_prompt_set(
    prompt_set_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> PromptSetOut:
    """Retrieve a single prompt set by ID."""
    prompt_set = session.get(PromptSet, prompt_set_id)
    if not prompt_set:
        raise ApiError(404, "not_found", "Prompt set not found.")
    return _prompt_set_to_out(prompt_set)


@router.put("/prompt-sets/{prompt_set_id}", response_model=PromptSetOut)
def update_prompt_set(
    prompt_set_id: uuid.UUID,
    payload: PromptSetUpdate,
    session: Session = Depends(get_session),
) -> PromptSetOut:
    """Update a prompt set's name, description, or ordered prompts."""
    prompt_set = session.get(PromptSet, prompt_set_id)
    if not prompt_set:
        raise ApiError(404, "not_found", "Prompt set not found.")

    if payload.name is not None:
        clean_name = payload.name.strip()
        if not clean_name:
            raise ApiError(400, "invalid_name", "Prompt set name cannot be empty.")
        if len(clean_name) > 64:
            raise ApiError(400, "invalid_name", "Prompt set name cannot exceed 64 characters.")
        existing = session.scalar(
            select(PromptSet).where(PromptSet.name == clean_name, PromptSet.id != prompt_set_id)
        )
        if existing:
            raise ApiError(
                409, "name_conflict", f"A prompt set named '{clean_name}' already exists."
            )
        prompt_set.name = clean_name

    if payload.description is not None:
        prompt_set.description = payload.description.strip()

    if payload.prompts is not None:
        clean_prompts = _validate_prompts(payload.prompts)
        prompt_set.prompts = clean_prompts

    prompt_set.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(prompt_set)
    return _prompt_set_to_out(prompt_set)


@router.delete("/prompt-sets/{prompt_set_id}", status_code=204)
def delete_prompt_set(
    prompt_set_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a prompt set.

    Leaves previous evaluations and their captured prompts intact without cascading deletion.
    """
    prompt_set = session.get(PromptSet, prompt_set_id)
    if not prompt_set:
        raise ApiError(404, "not_found", "Prompt set not found.")
    session.delete(prompt_set)
    session.commit()
    return Response(status_code=204)


# --- Evaluation Endpoints ---


@router.get("/evaluations", response_model=list[EvaluationOut])
def list_evaluations(session: Session = Depends(get_session)) -> list[EvaluationOut]:
    """List all model evaluations, newest first."""
    records = (
        session.execute(select(Evaluation).order_by(Evaluation.created_at.desc())).scalars().all()
    )
    return [_evaluation_to_out(ev) for ev in records]


@router.post("/evaluations", response_model=EvaluationOut, status_code=201)
def create_evaluation(
    payload: EvaluationCreate,
    session: Session = Depends(get_session),
) -> EvaluationOut:
    """Start an evaluation on a supported model using a saved prompt set.

    Captures an independent copy of the prompt contents and order.
    """
    model = session.get(Model, payload.model_id)
    if not model:
        raise ApiError(404, "not_found", "Selected model was not found.")

    prompt_set = session.get(PromptSet, payload.prompt_set_id)
    if not prompt_set:
        raise ApiError(404, "not_found", "Selected prompt set was not found.")

    if not prompt_set.prompts or len(prompt_set.prompts) == 0:
        raise ApiError(400, "empty_prompt_set", "Selected prompt set has no prompts to evaluate.")

    # Requirement 5: Capture an independent copy of exact prompt contents and order
    frozen_prompts = [str(p) for p in prompt_set.prompts]
    eval_name = (payload.name or f"{model.name} - {prompt_set.name}").strip()
    if not eval_name:
        eval_name = f"{model.name} - {prompt_set.name}"
    if len(eval_name) > 128:
        eval_name = eval_name[:128]

    eval_id = uuid.uuid4()
    job_id = uuid.uuid4()

    evaluation = Evaluation(
        id=eval_id,
        name=eval_name,
        model_id=model.id,
        model_name=model.name,
        prompt_set_id=prompt_set.id,
        prompt_set_name=prompt_set.name,
        captured_prompts=frozen_prompts,
        results=[],
        status="pending",
        job_id=job_id,
    )
    session.add(evaluation)

    job = Job(
        id=job_id,
        job_type="eval",
        status="pending",
        payload={
            "evaluation_id": str(eval_id),
            "model_id": str(model.id),
            "prompts": frozen_prompts,
            "max_new_tokens": payload.max_new_tokens,
            "temperature": payload.temperature,
        },
        progress={"current": 0, "total": len(frozen_prompts), "stage": "Queued"},
    )
    session.add(job)
    session.commit()
    session.refresh(evaluation)

    if os.environ.get("PYTEST_CURRENT_TEST"):
        from forge.eval import execute_eval_job

        execute_eval_job(job_id)
        session.refresh(evaluation)
    else:
        try:
            from forge.jobs import start_runner

            start_runner()
        except OSError:
            pass

    return _evaluation_to_out(evaluation)


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationOut)
def get_evaluation(
    evaluation_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> EvaluationOut:
    """Retrieve an evaluation record including captured prompts and outputs."""
    evaluation = session.get(Evaluation, evaluation_id)
    if not evaluation:
        raise ApiError(404, "not_found", "Evaluation not found.")
    return _evaluation_to_out(evaluation)


@router.delete("/evaluations/{evaluation_id}", status_code=204)
def delete_evaluation(
    evaluation_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete an evaluation record and its associated saved result rows."""
    evaluation = session.get(Evaluation, evaluation_id)
    if not evaluation:
        raise ApiError(404, "not_found", "Evaluation not found.")
    if evaluation.job_id:
        job = session.get(Job, evaluation.job_id)
        if job and job.status in ("pending", "running"):
            job.cancel_requested = True
    linked_results = (
        session.execute(select(EvalResult).where(EvalResult.evaluation_id == evaluation_id))
        .scalars()
        .all()
    )
    for r in linked_results:
        session.delete(r)
    session.delete(evaluation)
    session.commit()
    return Response(status_code=204)


# --- Bits Per Byte (BPB) Evaluation Endpoints ---


@router.get("/bpb-evaluations", response_model=list[BpbEvaluationOut])
def list_bpb_evaluations(
    session: Session = Depends(get_session),
) -> list[BpbEvaluationOut]:
    """List all BPB evaluations on held-out text, newest first."""
    records = (
        session.execute(select(BpbEvaluation).order_by(BpbEvaluation.created_at.desc()))
        .scalars()
        .all()
    )
    return [_bpb_eval_to_out(ev) for ev in records]


@router.post("/bpb-evaluations", response_model=BpbEvaluationOut, status_code=201)
def create_bpb_evaluation(
    payload: BpbEvaluationCreate,
    session: Session = Depends(get_session),
) -> BpbEvaluationOut:
    """Launch a bits-per-byte evaluation job on held-out text from a frozen snapshot."""
    dataset = session.get(Dataset, payload.dataset_id)
    if not dataset or dataset.status != "ready":
        raise ApiError(
            400, "snapshot_not_ready", "Selected dataset snapshot is not ready or was removed."
        )

    _, val_hash = _get_validation_file_info(dataset)

    model_source = payload.model_source
    imported_name = payload.imported_model_name
    picked = session.get(Model, payload.model_id) if payload.model_id else None
    if picked is not None and picked.source == "hub" and picked.artifact_uri:
        # a model imported from the hub is scored from its own folder
        model_source = "imported"
        imported_name = str(get_settings().data_dir / picked.artifact_uri)
    if model_source == "imported":
        model_name = (imported_name or "").strip()
        if not model_name:
            raise ApiError(400, "missing_model_name", "An imported model name must be provided.")
        if not ImportedModelAdapter.is_available():
            raise ApiError(
                501,
                "imported_model_unavailable",
                "Imported model evaluation depends on SP9.8 (transformers/imported model "
                "interface), which is not yet installed in this environment.",
            )
        model_id = picked.id if picked is not None else None
        if picked is not None:
            model_name = picked.name
        tok_identity = f"hf-tokenizer:{model_name}"
    else:
        if not payload.model_id:
            raise ApiError(400, "missing_model_id", "A native model must be selected.")
        model = session.get(Model, payload.model_id)
        if not model:
            raise ApiError(404, "not_found", "Selected model was not found.")
        model_name = model.name
        model_id = model.id
        tok_identity = "native-bpe"

    eval_name = (payload.name or f"BPB - {model_name} on {dataset.name}").strip()
    if len(eval_name) > 128:
        eval_name = eval_name[:128]

    eval_id = uuid.uuid4()
    job_id = uuid.uuid4()

    settings_dict = {
        "ctx_len": payload.ctx_len or 512,
        "model_source": model_source,
        "checkpoint_id": str(payload.checkpoint_id) if payload.checkpoint_id else None,
    }

    bpb_eval = BpbEvaluation(
        id=eval_id,
        name=eval_name,
        model_id=model_id,
        model_name=model_name,
        model_source=model_source,
        checkpoint_id=payload.checkpoint_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        validation_file_hash=val_hash,
        tokenizer_identity=tok_identity,
        bpb=None,
        total_nll=None,
        scored_bytes=None,
        scored_tokens=None,
        status="pending",
        error=None,
        settings=settings_dict,
        job_id=job_id,
    )
    session.add(bpb_eval)

    job = Job(
        id=job_id,
        job_type="eval",
        status="pending",
        payload={
            "eval_type": "bpb",
            "bpb_eval_id": str(eval_id),
            "model_id": str(model_id) if model_id else None,
            "imported_model_name": imported_name,
            "model_source": model_source,
            "checkpoint_id": str(payload.checkpoint_id) if payload.checkpoint_id else None,
            "dataset_id": str(dataset.id),
            "ctx_len": payload.ctx_len or 512,
        },
        progress={"current": 0, "total": 1, "stage": "Queued"},
    )
    session.add(job)
    session.commit()
    session.refresh(bpb_eval)

    if os.environ.get("PYTEST_CURRENT_TEST"):
        from forge.core.metrics import execute_bpb_eval_job

        execute_bpb_eval_job(job_id)
        session.refresh(bpb_eval)
    else:
        try:
            from forge.jobs import start_runner

            start_runner()
        except OSError:
            pass

    return _bpb_eval_to_out(bpb_eval)


@router.get("/bpb-evaluations/{evaluation_id}", response_model=BpbEvaluationOut)
def get_bpb_evaluation(
    evaluation_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> BpbEvaluationOut:
    """Retrieve details for a single bits-per-byte evaluation."""
    evaluation = session.get(BpbEvaluation, evaluation_id)
    if not evaluation:
        raise ApiError(404, "not_found", "BPB evaluation not found.")
    return _bpb_eval_to_out(evaluation)


@router.delete("/bpb-evaluations/{evaluation_id}", status_code=204)
def delete_bpb_evaluation(
    evaluation_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a BPB evaluation record and request job cancellation if running."""
    evaluation = session.get(BpbEvaluation, evaluation_id)
    if not evaluation:
        raise ApiError(404, "not_found", "BPB evaluation not found.")
    if evaluation.job_id:
        job = session.get(Job, evaluation.job_id)
        if job and job.status in ("pending", "running"):
            job.cancel_requested = True
    linked_results = (
        session.execute(select(EvalResult).where(EvalResult.evaluation_id == evaluation_id))
        .scalars()
        .all()
    )
    for r in linked_results:
        session.delete(r)
    session.delete(evaluation)
    session.commit()
    return Response(status_code=204)


# --- Evaluation Results Endpoints ---


def _eval_result_to_out(r: EvalResult) -> EvalResultOut:
    return EvalResultOut(
        id=r.id,
        result_type=r.result_type,
        model_id=r.model_id,
        model_name=r.model_name,
        checkpoint_id=r.checkpoint_id,
        evaluation_id=r.evaluation_id,
        settings=r.settings,
        code_version=r.code_version,
        prompt=r.prompt,
        answer=r.answer,
        bpb=r.bpb,
        total_nll=r.total_nll,
        scored_bytes=r.scored_bytes,
        scored_tokens=r.scored_tokens,
        benchmark_name=r.benchmark_name,
        score=r.benchmark_score,
        benchmark_score=r.benchmark_score,
        benchmark_scores=r.benchmark_scores,
        created_at=r.created_at,
    )


@router.get("/eval-results", response_model=list[EvalResultOut])
def list_eval_results(
    result_type: str | None = None,
    model_id: uuid.UUID | None = None,
    session: Session = Depends(get_session),
) -> list[EvalResultOut]:
    """List saved evaluation results, newest first.

    Pass ``result_type`` to filter by ``prompt``, ``bpb``, or ``benchmark``.
    Pass ``model_id`` to filter by model (including results associated with its checkpoints).
    """
    query = select(EvalResult).order_by(EvalResult.created_at.desc())
    if result_type is not None:
        query = query.where(EvalResult.result_type == result_type)
    if model_id is not None:
        ckpt_ids_subquery = (
            select(Checkpoint.id)
            .join(Run, Checkpoint.run_id == Run.id)
            .where(Run.model_id == model_id)
        )
        query = query.where(
            or_(
                EvalResult.model_id == model_id,
                EvalResult.checkpoint_id.in_(ckpt_ids_subquery),
            )
        )
    records = session.execute(query).scalars().all()
    return [_eval_result_to_out(r) for r in records]


@router.get("/eval-results/{result_id}", response_model=EvalResultOut)
def get_eval_result(
    result_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> EvalResultOut:
    """Retrieve a single saved evaluation result by ID."""
    result = session.get(EvalResult, result_id)
    if not result:
        raise ApiError(404, "not_found", "Evaluation result not found.")
    return _eval_result_to_out(result)


@router.delete("/eval-results/{result_id}", status_code=204)
def delete_eval_result(
    result_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a saved evaluation result record."""
    result = session.get(EvalResult, result_id)
    if not result:
        raise ApiError(404, "not_found", "Evaluation result not found.")
    ev_id = result.evaluation_id
    session.delete(result)
    session.flush()
    if ev_id is not None:
        remaining = (
            session.scalar(
                select(func.count(EvalResult.id)).where(EvalResult.evaluation_id == ev_id)
            )
            or 0
        )
        if remaining == 0:
            parent_eval = session.get(Evaluation, ev_id)
            if parent_eval is not None:
                session.delete(parent_eval)
            parent_bpb = session.get(BpbEvaluation, ev_id)
            if parent_bpb is not None:
                session.delete(parent_bpb)
            parent_job = session.get(Job, ev_id)
            if parent_job is not None and parent_job.job_type in ("benchmark", "prompt_compare"):
                session.delete(parent_job)
    session.commit()
    return Response(status_code=204)
