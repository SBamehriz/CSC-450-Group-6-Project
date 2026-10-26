from fastapi import APIRouter

from forge import gpu
from forge.schemas import GpuCard, GpuCheck, PresetMemory, PresetOut
from forge.train.presets import PRESETS, count_params, training_memory

router = APIRouter(tags=["training"])

SEQUENCES_PER_STEP = (1, 2, 4, 8, 16)


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
