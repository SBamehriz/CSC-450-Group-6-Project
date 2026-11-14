from fastapi import APIRouter

from forge import gpu
from forge.schemas import GpuCard, GpuCheck

router = APIRouter(tags=["training"])


@router.get("/gpu", response_model=GpuCheck)
def check_gpu() -> GpuCheck:
    seen = gpu.read()
    return GpuCheck(
        cards=[
            GpuCard(
                index=card.index,
                name=card.name,
                memory_bytes=card.memory_bytes,
                free_bytes=card.free_bytes,
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
