import shutil
import uuid

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, object_session

from forge import hub
from forge.chat_api import unload_models
from forge.config import get_settings
from forge.db import get_session
from forge.errors import ApiError
from forge.jobs_api import queue_task
from forge.measuring import pick_checkpoint
from forge.models import (
    BpbEvaluation,
    Checkpoint,
    EvalResult,
    Evaluation,
    Model,
    Run,
    Tokenizer,
)
from forge.schemas import (
    HubImportRequest,
    JobOut,
    ModelCreate,
    ModelOut,
    Page,
)
from forge.train.presets import (
    ModelConfig,
    count_params,
    explain_shape_error,
    training_memory,
    validate,
)

router = APIRouter(tags=["models"])


def _model_to_out(model: Model) -> ModelOut:
    mem: int | None = None
    try:
        if model.source != "forge":
            raise ValueError("not trained here")
        cfg = ModelConfig(
            n_layer=model.n_layer,
            d_model=model.d_model,
            n_head=model.n_head,
            ctx_len=model.ctx_len,
            vocab_size=model.vocab_size,
            dropout=model.dropout,
        )
        mem = sum(training_memory(cfg, 8).values())
    except Exception:
        mem = None

    session = object_session(model)
    best = pick_checkpoint(session, model.id) if session else None
    parent = session.get(Model, model.parent_id) if session and model.parent_id else None
    folder = get_settings().data_dir / model.artifact_uri if model.artifact_uri else None
    return ModelOut(
        id=model.id,
        name=model.name,
        description=model.description,
        preset=model.preset,
        n_layer=model.n_layer,
        d_model=model.d_model,
        n_head=model.n_head,
        ctx_len=model.ctx_len,
        vocab_size=model.vocab_size,
        dropout=model.dropout,
        param_count=model.param_count,
        config=model.config,
        tokenizer_id=model.tokenizer_id,
        source=model.source,
        estimated_memory_bytes=mem,
        parent_id=model.parent_id,
        parent_name=parent.name if parent else None,
        chat=model.chat,
        ready=best is not None or (model.source == "hub" and bool(folder and folder.is_dir())),
        best_val_loss=best.val_loss if best else None,
        created_at=model.created_at,
    )


@router.get("/models", response_model=Page[ModelOut])
def list_models(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    ready: bool | None = Query(default=None),
    session: Session = Depends(get_session),
) -> Page[ModelOut]:
    """List saved models, newest first, with stable pagination."""
    if ready is not None:
        # only models with weights to chat with, or only those without
        every = session.scalars(select(Model).order_by(Model.created_at.desc(), Model.id.desc()))
        outs = [out for out in map(_model_to_out, every) if out.ready == ready]
        return Page[ModelOut](items=outs[offset : offset + limit], total=len(outs))
    total = session.scalar(select(func.count(Model.id))) or 0
    records = (
        session.execute(
            select(Model)
            .order_by(Model.created_at.desc(), Model.id.desc())
            .offset(offset)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return Page[ModelOut](items=[_model_to_out(m) for m in records], total=total)


@router.post("/models", response_model=ModelOut, status_code=201)
def create_model(
    payload: ModelCreate,
    session: Session = Depends(get_session),
) -> ModelOut:
    """Create and save a new model configuration."""
    existing = session.execute(
        select(Model).where(Model.name == payload.name.strip())
    ).scalar_one_or_none()
    if existing:
        msg = f"A model named '{payload.name.strip()}' already exists."
        raise ApiError(409, "duplicate_name", msg)

    vocab_size = payload.vocab_size
    config_data = dict(payload.config)
    tokenizer_id = None

    if payload.tokenizer_id:
        vocabulary = session.get(Tokenizer, payload.tokenizer_id)
        if vocabulary is None or vocabulary.status != "ready" or not vocabulary.vocab_size:
            raise ApiError(
                400, "invalid_vocabulary", "Choose a ready vocabulary from the Vocabularies page."
            )
        vocab_size = vocabulary.vocab_size
        tokenizer_id = vocabulary.id
        config_data["vocabulary_name"] = vocabulary.name

    shape_error = explain_shape_error(
        n_layer=payload.n_layer,
        d_model=payload.d_model,
        n_head=payload.n_head,
        ctx_len=payload.ctx_len,
        vocab_size=vocab_size,
        dropout=payload.dropout,
    )
    if shape_error:
        raise ApiError(422, "invalid_shape", shape_error)

    try:
        model_cfg = ModelConfig(
            n_layer=payload.n_layer,
            d_model=payload.d_model,
            n_head=payload.n_head,
            ctx_len=payload.ctx_len,
            vocab_size=vocab_size,
            dropout=payload.dropout,
        )
        validate(model_cfg)
        params = count_params(model_cfg)
    except ValueError as err:
        raise ApiError(422, "invalid_shape", str(err)) from err

    model = Model(
        id=uuid.uuid4(),
        name=payload.name.strip(),
        description=payload.description.strip(),
        n_layer=payload.n_layer,
        d_model=payload.d_model,
        n_head=payload.n_head,
        ctx_len=payload.ctx_len,
        vocab_size=vocab_size,
        dropout=payload.dropout,
        param_count=params,
        config=config_data,
        tokenizer_id=tokenizer_id,
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return _model_to_out(model)


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(
    model_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> ModelOut:
    """Retrieve a saved model by ID."""
    model = session.get(Model, model_id)
    if not model:
        raise ApiError(404, "not_found", "Model not found.")
    return _model_to_out(model)
@router.delete("/models/all", status_code=204)
def delete_all_models(
    confirm: bool = False,
    session: Session = Depends(get_session),
) -> Response:
    """Delete all unreferenced models, or refuse without changing any records.

    Runs and saved evaluations are intentionally not erased by this endpoint.
    """
    if not confirm:
        return Response(
            content='{"detail":"Confirmation required"}',
            status_code=400,
            media_type="application/json",
        )

    models = session.scalars(select(Model)).all()
    if not models:
        return Response(status_code=204)

    # Check all dependencies before changing anything: no partial deletion.
    if (
        session.scalar(select(func.count(Run.id)))
        or session.scalar(select(func.count(Evaluation.id)))
        or session.scalar(select(func.count(EvalResult.id)).where(EvalResult.model_id.is_not(None)))
        or session.scalar(select(func.count(EvalResult.id)).where(EvalResult.checkpoint_id.is_not(None)))
        or session.scalar(select(func.count(BpbEvaluation.id)).where(BpbEvaluation.model_id.is_not(None)))
        or session.scalar(select(func.count(BpbEvaluation.id)).where(BpbEvaluation.checkpoint_id.is_not(None)))
    ):
        raise ApiError(
            409,
            "models_in_use",
            "Cannot delete all models while training runs or saved evaluations reference models. "
            "Remove dependent records first.",
        )

    folders = [
        get_settings().data_dir / model.artifact_uri
        for model in models
        if model.artifact_uri
    ]
    try:
        for model in models:
            session.delete(model)
        session.commit()
    except IntegrityError as err:
        session.rollback()
        raise ApiError(
            409, "models_in_use", "Some models are still referenced by other records."
        ) from err

    for model in models:
        unload_models(model.id)
    data_dir = get_settings().data_dir.resolve()
    for folder in folders:
        resolved = folder.resolve()
        if resolved != data_dir and data_dir in resolved.parents and resolved.is_dir():
            shutil.rmtree(resolved, ignore_errors=True)
    return Response(status_code=204)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a saved model only if no saved results or training runs reference it."""
    model = session.get(Model, model_id)
    if not model:
        raise ApiError(404, "not_found", "Model not found.")

    ckpt_ids_subquery = (
        select(Checkpoint.id).join(Run, Checkpoint.run_id == Run.id).where(Run.model_id == model_id)
    )
    saved_results_count = (
        session.scalar(
            select(func.count(EvalResult.id)).where(
                or_(
                    EvalResult.model_id == model_id,
                    EvalResult.checkpoint_id.in_(ckpt_ids_subquery),
                )
            )
        )
        or 0
    )
    eval_count = (
        session.scalar(select(func.count(Evaluation.id)).where(Evaluation.model_id == model_id))
        or 0
    )
    bpb_eval_count = (
        session.scalar(
            select(func.count(BpbEvaluation.id)).where(
                or_(
                    BpbEvaluation.model_id == model_id,
                    BpbEvaluation.checkpoint_id.in_(ckpt_ids_subquery),
                )
            )
        )
        or 0
    )
    total_saved = saved_results_count or (eval_count + bpb_eval_count)
    if total_saved > 0:
        msg = (
            f"Cannot delete model '{model.name}' because it has {total_saved} saved "
            "evaluation result(s). Delete its saved results first."
        )
        raise ApiError(409, "model_has_saved_results", msg)

    referencing_runs = (
        session.scalar(select(func.count(Run.id)).where(Run.model_id == model_id)) or 0
    )
    if referencing_runs > 0:
        msg = (
            f"Cannot delete model '{model.name}' because it is used by "
            f"{referencing_runs} training run(s)."
        )
        raise ApiError(409, "model_in_use", msg)

    folder = get_settings().data_dir / model.artifact_uri if model.artifact_uri else None
    unload_models(model_id)
    # its fine-tunes keep working, they just lose the link back
    session.execute(update(Model).where(Model.parent_id == model_id).values(parent_id=None))
    try:
        session.delete(model)
        session.commit()
    except IntegrityError as err:
        session.rollback()
        raise ApiError(
            409,
            "model_has_saved_results",
            f"Cannot delete model '{model.name}' because it has saved evaluation results. "
            "Delete its saved results first.",
        ) from err
    if folder is not None and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    return Response(status_code=204)


@router.post("/models/import", response_model=JobOut, status_code=202)
def import_model(payload: HubImportRequest, session: Session = Depends(get_session)) -> JobOut:
    """Check a hub model, then download it in the background."""
    try:
        planned = hub.plan(payload.repo_id.strip(), payload.revision.strip(), payload.on_cpu)
    except hub.Refused as problem:
        raise ApiError(409, "import_refused", str(problem)) from problem
    except ImportError as problem:
        raise ApiError(
            409, "hub_missing", "Importing needs the hub extra. Install it from the README first."
        ) from problem
    return queue_task(
        session,
        "model_import",
        {"plan": planned, "name": payload.name, "target_name": planned["repo_id"]},
    )
