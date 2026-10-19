from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from forge.db import get_session
from forge.models import Dataset, Job
from forge import gpu
from forge.schemas import GpuCard, GpuCheck, PresetMemory, PresetOut
from forge.train.presets import PRESETS, count_params, training_memory
from uuid import UUID
from pydantic import BaseModel, Field

router = APIRouter(tags=["training"])

SEQUENCES_PER_STEP = (1, 2, 4, 8, 16)


class TrainingJobCreate(BaseModel):
    dataset_id: UUID
    preset: str
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=1, ge=1, le=100)
@router.post("/jobs", status_code=201)
def create_training_job(
    payload: TrainingJobCreate,
    session: Session = Depends(get_session),
):
    dataset = session.get(Dataset, payload.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found",
        )

    if payload.preset not in PRESETS:
        raise HTTPException(
            status_code=422,
            detail="Invalid training preset",
        )

    job = Job(
        job_type="training",
        status="pending",
        payload={
            "dataset_id": str(payload.dataset_id),
            "preset": payload.preset,
            "learning_rate": payload.learning_rate,
            "epochs": payload.epochs,
        },
    )

    session.add(job)

    try:
        session.commit()
        session.refresh(job)
    except Exception:
        session.rollback()
        raise

    return {
        "id": str(job.id),
        "status": job.status,
    }

@router.get("/presets", response_model=list[PresetOut])
def list_presets() -> list[PresetOut]:
    return [
        PresetOut(
            name=name,
            n_layer=config.n_layer,
            d_model=config.d_model,
            n_head=config.n_head,
            ctx_len=config.ctx_len,
            vocab_size=config.vocab_size,
            parameters=count_params(config),
            memory=[
                PresetMemory(
                    sequences_per_step=sequences,
                    bytes=sum(training_memory(config, micro_batch_seqs=sequences).values()),
                )
                for sequences in SEQUENCES_PER_STEP
            ],
        )
        for name, config in PRESETS.items()
    ]


@router.get("/gpu", response_model=GpuCheck)
def check_gpu() -> GpuCheck:
    seen = gpu.read()
    return GpuCheck(
        cards=[
            GpuCard(
                index=card.index,
                name=card.name,
                memory_bytes=card.memory_bytes,
                driver=card.driver,
                compute_capability=card.compute_capability,
            )
            for card in seen.cards
        ],
        tool_said=seen.said,
        torch=seen.torch,
        torch_version=seen.torch_version,
        torch_cuda=seen.torch_cuda,
    )
