# from a size or the hours you have to a whole training plan, each choice with a reason

import math
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge import gpu
from forge.measure_api import work_per_token
from forge.measuring import pick_checkpoint
from forge.models import Bucket, Dataset, DatasetBuild, Document, Model, SpeedTest
from forge.schemas import PlanData, PlanMemory, PlanOut, PlanRequest, PlanTime, ShapeIn
from forge.train.presets import (
    MAX_D_MODEL,
    MAX_N_LAYER,
    ModelConfig,
    count_params,
)
from forge.train.presets import estimate_vram_bytes as memory_bytes
from forge.train.settings import TrainSettings, steps_for_tokens

TOKENS_PER_PARAM = 20
WIDTH_PER_LAYER = 48
# share of free card memory training may use
USABLE = 0.9
# share of the hours spent learning; checks and save points take the rest
BUDGET = 0.9
# a second card adds a little less than a whole card
EXTRA_CARD = 0.9
MANY_PASSES = 4
FINE_TUNE_PASSES = 3
DEFAULT_HOURS = 1.0
SMALLEST, LARGEST = 100_000, 100_000_000_000
SMALLEST_DEFAULT = 1_000_000
CPU_PASS = 8
# a guess of the word pieces in some text, until it has been made into a training set
CHARS_PER_TOKEN = 4
# rough speeds when nothing was measured: per GB of card memory, per processor core
GUESS_CARD_FLOPS_PER_GB = 1e12
GUESS_CORE_FLOPS = 5e9
SHAPE_KEYS = ("n_layer", "d_model", "n_head", "ctx_len", "vocab_size")
SETTING_KEYS = ("batch_size", "learning_rate", "warmup_steps", "weight_decay")
Source = Literal["measured", "scaled", "none"]


@dataclass(frozen=True)
class Computer:
    # the cards training would use; none means the processor
    cards: list[gpu.Card]
    cpu_count: int


def estimate_tokens(session: Session, bucket_id: uuid.UUID) -> tuple[int, bool]:
    """Training word pieces in a collection, and whether that is counted or a guess."""
    snapshot = session.scalar(
        select(Dataset)
        .where(Dataset.bucket_id == bucket_id, Dataset.status == "ready")
        .order_by(Dataset.created_at.desc())
    )
    if snapshot is not None:
        build = session.scalar(
            select(DatasetBuild)
            .where(DatasetBuild.dataset_id == snapshot.id, DatasetBuild.status == "ready")
            .order_by(DatasetBuild.created_at.desc())
        )
        counted = (build.stats.get("train") or {}).get("tokens") if build else None
        if counted:
            return int(counted), True
    chars = session.scalar(
        select(func.coalesce(func.sum(Document.char_count), 0)).where(
            Document.bucket_id == bucket_id, Document.status == "parsed"
        )
    )
    return int(chars or 0) // CHARS_PER_TOKEN, False


def window_and_vocabulary(parameters: float) -> tuple[int, int]:
    for most, ctx, vocab in ((10e6, 256, 4096), (50e6, 512, 8192), (300e6, 1024, 16384)):
        if parameters < most:
            return ctx, vocab
    return 2048, 32768


def head_size(width: int) -> int:
    return 32 if width <= 256 else 64


def shape_for(target: float, given: dict[str, int] | None = None) -> ModelConfig:
    """The shape nearest the target size, with depth growing with width."""
    given = {key: value for key, value in (given or {}).items() if value}
    ctx, vocab = window_and_vocabulary(target)
    ctx, vocab = given.get("ctx_len", ctx), given.get("vocab_size", vocab)
    heads = given.get("n_head")
    if "d_model" in given:
        widths = [given["d_model"]]
    else:
        widths = [w for w in range(32, MAX_D_MODEL + 1, 32) if w <= 256 or w % 64 == 0]
        if heads:
            widths = [w for w in widths if w % heads == 0] or [heads * 64]
        if "n_layer" not in given:
            # depth follows width, so the size grows about as width cubed
            middle = (4 * target) ** (1 / 3)
            widths = [w for w in widths if middle / 4 <= w <= middle * 1.5] or widths[:1]
    best: tuple[float, ModelConfig] | None = None
    for width in widths:
        n_head = heads or max(1, width // head_size(width))
        if width % n_head:
            continue
        rule = max(1.0, width / WIDTH_PER_LAYER)
        if "n_layer" in given:
            depths = [given["n_layer"]]
        else:
            depths = range(max(1, int(rule * 0.6)), min(MAX_N_LAYER, math.ceil(rule * 1.4)) + 1)
        for n_layer in depths:
            config = ModelConfig(n_layer, width, n_head, ctx, vocab)
            miss = abs(math.log(count_params(config) / target))
            score = miss + 0.02 * abs(math.log(n_layer / rule))
            if best is None or score < best[0]:
                best = (score, config)
    if best is None:
        raise ValueError("No shape fits these choices. Change the width or the number of heads.")
    return best[1]


def pass_size(config: ModelConfig, most: int, free: int) -> int:
    """The most sequences per pass, up to `most`, that fit in `free` bytes. 0 if none do."""
    room = free * USABLE
    if memory_bytes(config, 1) > room:
        return 0
    low, high = 1, max(1, most)
    while low < high:
        middle = (low + high + 1) // 2
        if memory_bytes(config, middle) <= room:
            low = middle
        else:
            high = middle - 1
    return low


def computer(device: str) -> tuple[Computer, list[str]]:
    """What training would run on, and blockers if it can't run."""
    cpus = os.cpu_count() or 1
    reading = gpu.read()
    if reading.torch == "missing":
        return Computer([], cpus), [
            "PyTorch is not installed, so training can't start. Install the train extra."
        ]
    usable = reading.cards if reading.torch in ("cuda", "rocm") else []
    if device == "cuda" and not usable:
        return Computer([], cpus), [
            "This plan asks for a graphics card, but PyTorch can't use one on this computer."
        ]
    return Computer([] if device == "cpu" else usable, cpus), []


def _shape_dict(config: ModelConfig) -> dict[str, int]:
    return {key: getattr(config, key) for key in SHAPE_KEYS}


def _speed(config: ModelConfig, test: SpeedTest | None, here: Computer) -> tuple[float, Source]:
    """Word pieces per second for this shape, and where that number came from."""
    cards = max(1, len(here.cards))
    if test is None:
        if here.cards:
            flops = sum(card.memory_bytes / 2**30 for card in here.cards) * GUESS_CARD_FLOPS_PER_GB
        else:
            flops = here.cpu_count * GUESS_CORE_FLOPS
        return flops / work_per_token(config), "none"
    # a speed test times one card
    tested = ModelConfig(**{key: test.shape.get(key, 16384) for key in SHAPE_KEYS})
    speed = test.tokens_per_sec * work_per_token(tested) / work_per_token(config)
    if cards > 1:
        speed *= cards * EXTRA_CARD
    same = _shape_dict(tested) == _shape_dict(config) and cards == 1
    return speed, "measured" if same else "scaled"


def _hours(tokens: int, speed: float) -> float:
    return tokens / (speed * BUDGET) / 3600


def _biggest(works: Callable[[float], bool]) -> float:
    """The biggest size, in parameters, that `works`."""
    low, high = float(SMALLEST), float(LARGEST)
    if not works(low):
        return low
    if works(high):
        return high
    while high / low > 1.01:
        middle = math.sqrt(low * high)
        if works(middle):
            low = middle
        else:
            high = middle
    return low


def _free(here: Computer) -> int | None:
    return min(card.free_bytes for card in here.cards) if here.cards else None


def _gb(value: float) -> str:
    return f"{value / 1e9:.1f} GB"


def _count(value: float) -> str:
    for size, mark in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:.3g}{mark}"
    return f"{value:.0f}"


def _batch(parameters: int, ctx: int, tokens: int, fewest_steps: int) -> int:
    # about 25 x the square root of the size in word pieces per step, as a power of two
    target = min(max(25 * math.sqrt(parameters), 16_384), 1_048_576)
    batch = max(1, min(4096, 2 ** round(math.log2(target / ctx))))
    while batch > 1 and tokens / (batch * ctx) < fewest_steps:
        batch //= 2
    return batch


def _learning_rate(parameters: int) -> float:
    rate = 6e-4 * (124e6 / parameters) ** 0.33
    return float(f"{min(3e-3, max(6e-5, rate)):.2g}")


def _base(session: Session, model_id) -> tuple[Model | None, str | None]:
    base = session.get(Model, model_id)
    if base is None:
        return None, "That model no longer exists."
    if base.source == "hub":
        return base, (
            f"{base.name} came from the Hugging Face hub. Fine-tuning starts from a model "
            "trained in Forge."
        )
    if pick_checkpoint(session, base.id) is None:
        return base, f"{base.name} has no save point yet. Train it first."
    return base, None


def plan(session: Session, request: PlanRequest) -> PlanOut:
    here, blockers = computer(request.device)
    warnings: list[str] = []
    why: dict[str, str] = {}
    chosen = request.model_dump(exclude_none=True)

    tokens: int | None = None
    exact = False
    bucket = session.get(Bucket, request.bucket_id) if request.bucket_id else None
    if request.bucket_id and bucket is None:
        blockers.append("That collection no longer exists.")
    if bucket is not None:
        tokens, exact = estimate_tokens(session, bucket.id)
        if not tokens:
            blockers.append(f"{bucket.name} has no text yet. Add some on the Data page first.")

    device = "cuda" if here.cards else "cpu"
    test = session.scalar(
        select(SpeedTest).where(SpeedTest.device == device).order_by(SpeedTest.created_at.desc())
    )
    free = _free(here)

    def fits(config: ModelConfig) -> bool:
        return free is None or memory_bytes(config, 1) <= free * USABLE

    base = None
    if request.base_model_id:
        base, problem = _base(session, request.base_model_id)
        if problem:
            blockers.append(problem)
        if bucket is None:
            blockers.append("Choose a collection to fine-tune on.")

    if base is not None:
        config = ModelConfig(base.n_layer, base.d_model, base.n_head, base.ctx_len, base.vocab_size)
        parameters = count_params(config)
        for key in SHAPE_KEYS:
            why[key] = f"A fine-tune keeps the shape of {base.name}."
        why["parameters"] = f"The same size as {base.name}, the model it starts from."
        enough_for = TOKENS_PER_PARAM * parameters
        train_tokens = min(FINE_TUNE_PASSES * (tokens or 0) or enough_for, enough_for)
        why["train_tokens"] = (
            f"About {FINE_TUNE_PASSES} passes over the collection, enough to learn its style "
            "without forgetting what the model already knows."
        )
        if request.hours:
            speed, _ = _speed(config, test, here)
            budget = int(speed * BUDGET * request.hours * 3600)
            if budget < train_tokens:
                train_tokens = budget
                why["train_tokens"] = f"As much as fits in {request.hours:g} hours."
    else:
        given = {key: chosen[key] for key in SHAPE_KEYS if key in chosen}
        try:
            shape_for(SMALLEST, given)
        except ValueError as problem:
            # say so, and plan around the choices that clash
            blockers.append(str(problem))
            given = {}

        def tokens_for(config: ModelConfig) -> int:
            return request.train_tokens or TOKENS_PER_PARAM * count_params(config)

        def works_in(hours: float) -> Callable[[float], bool]:
            def works(size: float) -> bool:
                config = shape_for(size, given)
                speed, _ = _speed(config, test, here)
                return fits(config) and _hours(tokens_for(config), speed) <= hours

            return works

        if request.parameters:
            target = float(request.parameters)
            why["parameters"] = f"You asked for about {_count(target)} parameters."
        elif request.hours:
            target = _biggest(works_in(request.hours))
            why["parameters"] = (
                f"The biggest model that trains well in {request.hours:g} hours on this computer."
            )
            if not works_in(request.hours)(SMALLEST):
                warnings.append(
                    f"Even a very small model needs more than {request.hours:g} hours here. "
                    "It will take longer than you asked."
                )
        else:
            target = _biggest(works_in(DEFAULT_HOURS))
            if tokens:
                target = min(target, max(SMALLEST_DEFAULT, tokens / TOKENS_PER_PARAM))
            why["parameters"] = (
                "A size that suits your collection and trains in about an hour on this computer."
            )
        config = shape_for(target, given)
        parameters = count_params(config)
        why["n_layer"] = (
            "Deeper models can learn harder patterns. Forge uses about one layer for every "
            f"{WIDTH_PER_LAYER} of width, in line with well known models."
        )
        why["d_model"] = (
            "Width is how many numbers describe each word piece. It is picked with the layers "
            "to reach the size."
        )
        why["n_head"] = (
            f"Each attention head looks at {config.head_dim} of those numbers, so the model can "
            "follow several things at once."
        )
        why["ctx_len"] = (
            "How many word pieces the model reads at once. Bigger models make good use of a "
            "longer window."
        )
        why["vocab_size"] = (
            "How many different word pieces the model knows. Bigger models can use more."
        )
        train_tokens = tokens_for(config)
        why["train_tokens"] = (
            f"About {TOKENS_PER_PARAM} word pieces for every parameter, the usual amount for a "
            "model to learn well."
        )

    if request.train_tokens:
        train_tokens = request.train_tokens
    train_tokens = max(1, int(train_tokens))
    passes = train_tokens / tokens if tokens else None

    fine_tune = base is not None
    batch = request.batch_size or _batch(
        parameters, config.ctx_len, train_tokens, 50 if fine_tune else 200
    )
    why["batch_size"] = (
        "Texts learned from in one step. Bigger models need more of them for a steady signal."
    )
    steps = steps_for_tokens(train_tokens, batch, config.ctx_len)
    rate = _learning_rate(parameters)
    why["learning_rate"] = "Bigger models learn better with smaller steps."
    if fine_tune:
        rate = float(f"{rate * 0.3:.2g}")
        why["learning_rate"] = "Lower than for a new model, so it keeps what it already knows."
    rate = request.learning_rate or rate
    warmup = min(100, max(5, steps // 50)) if fine_tune else min(2000, max(10, steps // 50))
    warmup = min(request.warmup_steps if request.warmup_steps is not None else warmup, steps - 1)
    why["warmup_steps"] = (
        "The learning rate starts low and climbs over the first steps, so training starts calmly."
    )
    if fine_tune:
        why["warmup_steps"] = "A short warm up: the model is already trained."
    decay = request.weight_decay if request.weight_decay is not None else 0.1
    why["weight_decay"] = "A gentle pull towards zero keeps the weights from growing too big."
    many = passes is not None and passes > MANY_PASSES
    if fine_tune:
        dropout = 0.1
        why["dropout"] = "A little dropout so the model does not learn a small collection by heart."
    elif many:
        dropout = 0.1
        why["dropout"] = "Some dropout, because the model will read the same text many times."
    else:
        dropout = 0.0
        why["dropout"] = "No dropout: with enough text the model has no time to learn it by heart."
    if request.dropout is not None:
        dropout = request.dropout
    for key in (*SHAPE_KEYS, *SETTING_KEYS, "dropout", "train_tokens"):
        if key in chosen and not (fine_tune and key in SHAPE_KEYS):
            why[key] = "You chose this."

    eval_every = min(steps, min(1000, max(10, steps // 20)))
    settings = TrainSettings(
        batch_size=batch,
        max_steps=steps,
        learning_rate=rate,
        warmup_steps=max(0, warmup),
        weight_decay=decay,
        eval_every=eval_every,
        save_every=min(steps, eval_every * 5),
    )

    if many and passes is not None and tokens is not None:
        enough = train_tokens // MANY_PASSES
        warnings.append(
            f"The model would read your collection about {passes:.0f} times, so it may learn it "
            f"by heart. About {_count(enough)} word pieces of text would be enough for this "
            f"size: add {_count(max(0, enough - tokens))} more, or choose a smaller model."
        )
    if tokens is not None and not exact:
        warnings.append(
            "The amount of text is a guess until it has been prepared for training once."
        )

    cards = len(here.cards)
    share = math.ceil(batch / max(1, cards))
    if here.cards:
        assert free is not None
        per_pass = pass_size(config, share, free)
        if not per_pass:
            blockers.append(
                f"This model needs about {_gb(memory_bytes(config, 1))} of graphics card memory "
                f"to train, and {_gb(free)} is free. Choose a smaller size, or a computer with a "
                "bigger graphics card."
            )
    else:
        per_pass = min(share, CPU_PASS)
        warnings.append(
            "There is no graphics card to use, so training runs on the processor. That is much "
            "slower: keep the model small."
        )
    name = None
    if here.cards:
        name = here.cards[0].name if cards == 1 else f"{cards} x {here.cards[0].name}"
    memory = PlanMemory(
        needed_bytes=memory_bytes(config, max(1, per_pass)),
        free_bytes=free,
        cards=cards,
        card_name=name,
        sequences_per_pass=max(1, per_pass),
        fits=None if not here.cards else per_pass > 0,
    )

    speed, source = _speed(config, test, here)
    if source == "none":
        warnings.append(
            "The time is a rough guess. Run the speed test in Settings for a real number."
        )
    shape = ShapeIn(**_shape_dict(config))
    return PlanOut(
        fine_tune=fine_tune,
        shape=shape,
        dropout=dropout,
        parameters=parameters,
        train_tokens=train_tokens,
        settings=settings,
        why=why,
        data=PlanData(
            tokens=tokens,
            exact=exact,
            passes=passes,
            enough=None if passes is None else not many,
        ),
        memory=memory,
        time=PlanTime(
            tokens_per_sec=speed,
            hours=_hours(steps * batch * config.ctx_len, speed),
            source=source,
        ),
        warnings=warnings,
        blockers=blockers,
    )
