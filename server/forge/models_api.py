import uuid

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.errors import ApiError
from forge.models import Dataset, Model, Run
from forge.schemas import (
    ModelCreate,
    ModelOut,
    ModelPreviewOut,
    ModelPreviewRequest,
    Page,
    VocabularyOut,
)
from forge.train.presets import (
    ALLOWED_CTX,
    MAX_D_MODEL,
    MAX_DROPOUT,
    MAX_N_LAYER,
    PRESETS,
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
        estimated_memory_bytes=mem,
        created_at=model.created_at,
    )


@router.get("/models/presets")
def get_presets() -> dict:
    """Return model preset shapes and validation boundaries."""
    return {
        "presets": {
            name: {
                "n_layer": cfg.n_layer,
                "d_model": cfg.d_model,
                "n_head": cfg.n_head,
                "ctx_len": cfg.ctx_len,
                "vocab_size": cfg.vocab_size,
                "dropout": cfg.dropout,
                "param_count": count_params(cfg),
                "estimated_memory_bytes": sum(training_memory(cfg, 8).values()),
            }
            for name, cfg in PRESETS.items()
        },
        "limits": {
            "max_n_layer": MAX_N_LAYER,
            "max_d_model": MAX_D_MODEL,
            "allowed_ctx": list(ALLOWED_CTX),
            "max_dropout": MAX_DROPOUT,
        },
    }


@router.get("/models/vocabularies", response_model=list[VocabularyOut])
def list_vocabularies(session: Session = Depends(get_session)) -> list[VocabularyOut]:
    """List available vocabularies learned from prepared datasets."""
    datasets = session.execute(select(Dataset).order_by(Dataset.created_at.desc())).scalars().all()

    vocabularies: list[VocabularyOut] = []
    for ds in datasets:
        prep = ds.config.get("prepared", {})
        if prep.get("ready") and prep.get("vocab_size"):
            vocabularies.append(
                VocabularyOut(
                    dataset_id=ds.id,
                    dataset_name=ds.name,
                    vocab_size=int(prep["vocab_size"]),
                    requested_vocab_size=int(prep.get("requested_vocab_size", prep["vocab_size"])),
                    train_tokens=int(prep.get("train_tokens", 0)),
                    trained_on_documents=int(prep.get("trained_on_documents", 0)),
                )
            )
    return vocabularies


@router.post("/models/preview", response_model=ModelPreviewOut)
def preview_model_shape(
    payload: ModelPreviewRequest,
    session: Session = Depends(get_session),
) -> ModelPreviewOut:
    """Preview model parameter count and estimated memory before saving."""
    vocab_size = 16384
    if payload.dataset_id:
        ds = session.get(Dataset, payload.dataset_id)
        if not ds:
            return ModelPreviewOut(
                valid=False,
                error="The selected vocabulary dataset was not found.",
            )
        prep = ds.config.get("prepared", {})
        if not prep.get("ready") or not prep.get("vocab_size"):
            return ModelPreviewOut(
                valid=False,
                error=f"Dataset '{ds.name}' does not have prepared token data yet.",
            )
        vocab_size = int(prep["vocab_size"])
    elif payload.vocab_size:
        vocab_size = payload.vocab_size

    # Check validation rules and explain in plain language
    shape_error = explain_shape_error(
        n_layer=payload.n_layer,
        d_model=payload.d_model,
        n_head=payload.n_head,
        ctx_len=payload.ctx_len,
        vocab_size=vocab_size,
        dropout=payload.dropout,
    )
    if shape_error:
        return ModelPreviewOut(valid=False, vocab_size=vocab_size, error=shape_error)

    try:
        cfg = ModelConfig(
            n_layer=payload.n_layer,
            d_model=payload.d_model,
            n_head=payload.n_head,
            ctx_len=payload.ctx_len,
            vocab_size=vocab_size,
            dropout=payload.dropout,
        )
        validate(cfg)
        params = count_params(cfg)
        mem = sum(training_memory(cfg, 8).values())
        return ModelPreviewOut(
            valid=True,
            param_count=params,
            estimated_memory_bytes=mem,
            estimated_memory_mb=round(mem / (1024 * 1024), 2),
            vocab_size=vocab_size,
        )
    except ValueError as err:
        return ModelPreviewOut(valid=False, vocab_size=vocab_size, error=str(err))


@router.get("/models", response_model=Page[ModelOut])
def list_models(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> Page[ModelOut]:
    """List saved models, newest first, with stable pagination."""
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

    if payload.dataset_id:
        ds = session.get(Dataset, payload.dataset_id)
        if not ds:
            raise ApiError(
                400, "invalid_vocabulary", "The selected vocabulary dataset does not exist."
            )
        prep = ds.config.get("prepared", {})
        if not prep.get("ready") or not prep.get("vocab_size"):
            msg = (
                f"Dataset '{ds.name}' has not been prepared yet. "
                "Prepare it before creating a model."
            )
            raise ApiError(400, "invalid_vocabulary", msg)
        vocab_size = int(prep["vocab_size"])
        # Preserve vocabulary association using existing config pattern
        config_data["dataset_id"] = str(ds.id)
        config_data["dataset_name"] = ds.name
        config_data["vocabulary_source"] = "prepared_dataset"
        config_data["vocab_size"] = vocab_size

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
        preset=payload.preset,
        n_layer=payload.n_layer,
        d_model=payload.d_model,
        n_head=payload.n_head,
        ctx_len=payload.ctx_len,
        vocab_size=vocab_size,
        dropout=payload.dropout,
        param_count=params,
        config=config_data,
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


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a saved model only if no training run references it."""
    model = session.get(Model, model_id)
    if not model:
        raise ApiError(404, "not_found", "Model not found.")

    referencing_runs = (
        session.scalar(select(func.count(Run.id)).where(Run.model_id == model_id)) or 0
    )
    if referencing_runs > 0:
        msg = (
            f"Cannot delete model '{model.name}' because it is used by "
            f"{referencing_runs} training run(s)."
        )
        raise ApiError(409, "model_in_use", msg)

    session.delete(model)
    session.commit()
    return Response(status_code=204)
