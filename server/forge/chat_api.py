import asyncio
import gc
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anyio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from forge import measuring
from forge.db import get_engine
from forge.gpu import free_gpu_memory
from forge.jobs import JobFailed
from forge.models import Job, Model
from forge.schemas import ChatRequest

router = APIRouter(tags=["chat"])
log = logging.getLogger(__name__)

# models kept loaded between answers
KEEP = 2
ACTIVE = ("pending", "running")


@dataclass
class Slot:
    loaded: measuring.Loaded
    # one answer at a time per model
    lock: threading.Lock = field(default_factory=threading.Lock)


_slots: OrderedDict[uuid.UUID, Slot] = OrderedDict()
_slots_lock = threading.Lock()


def _training(session: Session) -> bool:
    return bool(
        session.scalar(
            select(Job.id).where(Job.job_type == "train_run", Job.status.in_(ACTIVE)).limit(1)
        )
    )


def _slot(model_id: uuid.UUID) -> Slot:
    with Session(get_engine()) as session:
        best = measuring.pick_checkpoint(session, model_id)
        # a training run needs the whole graphics card, so chat uses the processor
        training = _training(session)
        row = session.get(Model, model_id)
        # float32 weights plus room to run, against what the card has free
        room = measuring.card_room()
        needed = row.param_count * 4 * measuring.RUN_OVERHEAD if row else 0
        big = room is not None and needed > room
    with _slots_lock:
        slot = _slots.get(model_id)
        same = slot is not None and slot.loaded.checkpoint_id == (best.id if best else None)
        if slot is not None and same and not (training and slot.loaded.device != "cpu"):
            _slots.move_to_end(model_id)
            return slot
        _slots.pop(model_id, None)
        while len(_slots) >= KEEP:
            _slots.popitem(last=False)
        _free()
        slot = Slot(measuring.load(model_id, None, "cpu" if training or big else "auto"))
        _slots[model_id] = slot
        return slot


def _free() -> None:
    gc.collect()
    free_gpu_memory()


def unload_models(model_id: uuid.UUID | None = None) -> None:
    # training calls this to get the card back
    with _slots_lock:
        if model_id is None:
            _slots.clear()
        else:
            _slots.pop(model_id, None)
        _free()


def _event(value: dict) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"


def _problem(problem: Exception) -> str:
    if isinstance(problem, JobFailed):
        return str(problem)
    log.exception("chat answer failed", exc_info=problem)
    if "out of memory" in str(problem).lower():
        return (
            "The graphics card ran out of memory. Close other programs that use it, or start "
            "a new conversation."
        )
    return "The model stopped before it finished this answer. Try again."


def _write(
    slot: Slot, payload: ChatRequest, ids: list[int], room: int, streamer, stop, made: dict
) -> None:
    try:
        with slot.lock:
            if not stop.is_set():
                made["ids"] = measuring.generate(
                    slot.loaded,
                    ids,
                    room,
                    payload.temperature,
                    payload.top_p,
                    payload.seed,
                    streamer=streamer,
                    stop=stop,
                )
    except Exception as problem:
        made["error"] = problem
    finally:
        streamer.end()


async def _answer(payload: ChatRequest, stop: threading.Event) -> AsyncIterator[str]:
    began = time.perf_counter()
    try:
        from transformers import TextIteratorStreamer

        slot = await asyncio.to_thread(_slot, payload.model_id)
        loaded = slot.loaded
        turns = [turn.model_dump() for turn in payload.messages]
        ids, room = await asyncio.to_thread(measuring.fit, loaded, turns, payload.max_new_tokens)
    except ImportError:
        yield _event({"error": "Chat needs PyTorch and transformers. Install the train extra."})
        return
    except Exception as problem:
        yield _event({"error": _problem(problem)})
        return
    streamer = TextIteratorStreamer(loaded.tokenizer, skip_prompt=True, skip_special_tokens=True)
    made: dict = {}
    writer = threading.Thread(
        target=_write, args=(slot, payload, ids, room, streamer, stop, made), daemon=True
    )
    writer.start()
    try:
        while (text := await asyncio.to_thread(next, streamer, None)) is not None:
            if text:
                yield _event({"text": text})
    finally:
        # done, or the client went away: either way the model is free after this piece
        stop.set()
        with anyio.CancelScope(shield=True):
            await asyncio.to_thread(writer.join)
    if "error" in made:
        yield _event({"error": _problem(made["error"])})
        return
    new = made.get("ids", [])
    ended = bool(new) and new[-1] in loaded.ends
    yield _event(
        {
            "done": True,
            "tokens": len(new) - ended,
            "seconds": round(time.perf_counter() - began, 3),
            "stopped": "end" if ended else "length",
        }
    )


@router.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    """Streams the answer as server-sent events."""
    stop = threading.Event()
    return StreamingResponse(
        _answer(payload, stop),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
        background=BackgroundTask(stop.set),
    )
