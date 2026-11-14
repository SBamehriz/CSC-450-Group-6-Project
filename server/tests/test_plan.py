import math
import time
import uuid

import pytest

from forge import gpu
from forge.models import Bucket, Checkpoint, Dataset, DatasetBuild, Model, SpeedTest, Tokenizer
from forge.schemas import PlanRequest
from forge.train import plan as planner
from forge.train.plan import memory_bytes, pass_size, shape_for, work_per_token
from forge.train.presets import ModelConfig, count_params

GB = 2**30
SMALL = {"n_layer": 6, "d_model": 384, "n_head": 6, "ctx_len": 512, "vocab_size": 8192}


def card(free_gb: float, total_gb: float = 80, name: str = "NVIDIA A100-SXM4-80GB") -> gpu.Card:
    return gpu.Card(0, name, int(total_gb * GB), "580.1", "8.0", int(free_gb * GB))


@pytest.fixture()
def computer(monkeypatch):
    """Choose the cards this computer has; none by default."""

    def use(*cards: gpu.Card, torch: str = "cuda") -> None:
        reading = gpu.Reading(list(cards), None, torch, "2.13", "13.0")  # type: ignore[arg-type]
        monkeypatch.setattr(gpu, "read", lambda: reading)

    use()
    return use


@pytest.fixture()
def make_bucket(db):
    def make(kind: str = "text") -> Bucket:
        bucket = Bucket(name=f"bucket-{uuid.uuid4().hex[:8]}", kind=kind)
        db.add(bucket)
        db.commit()
        return bucket

    return make


@pytest.fixture()
def text(monkeypatch, make_bucket, make_document):
    """A collection holding this many word pieces."""

    def make(tokens: int, exact: bool = True, kind: str = "text"):
        bucket = make_bucket(kind=kind)
        make_document(bucket.id, chars=tokens * 4)
        monkeypatch.setattr(planner, "estimate_tokens", lambda session, bucket_id: (tokens, exact))
        return bucket

    return make


def model_row(db, **values) -> Model:
    shape = {"n_layer": 4, "d_model": 128, "n_head": 4, "ctx_len": 512, "vocab_size": 16384}
    row = Model(name=f"model-{uuid.uuid4().hex[:8]}", param_count=1, **{**shape, **values})
    db.add(row)
    db.commit()
    return row


def speed_row(db, tokens_per_sec: float, shape: dict = SMALL) -> None:
    db.add(
        SpeedTest(
            shape=shape,
            sequences_per_step=8,
            device="cuda",
            device_name="NVIDIA A100-SXM4-80GB",
            tokens_per_sec=tokens_per_sec,
            steps=30,
        )
    )
    db.commit()


def ask(db, **values):
    return planner.plan(db, PlanRequest(**values))


@pytest.mark.parametrize("target", [5e6, 20e6, 124e6, 1e9, 5e9])
def test_a_size_is_reached_closely_at_any_scale(target):
    config = shape_for(target)
    assert abs(count_params(config) / target - 1) < 0.05
    assert config.d_model % config.n_head == 0 and config.head_dim in (32, 64)
    # deeper as it gets wider, in line with GPT-2 small (12 x 768) and large (36 x 1280)
    assert 1 / 80 < config.n_layer / config.d_model < 1 / 30 or config.n_layer <= 2


@pytest.mark.parametrize(
    "known, layers, width",
    [
        ({"vocab_size": 50257, "ctx_len": 1024}, 12, 768),  # GPT-2 small
        ({"vocab_size": 49152, "ctx_len": 2048}, 30, 576),  # SmolLM 135M
        ({"vocab_size": 50257, "ctx_len": 1024}, 36, 1280),  # GPT-2 large
    ],
)
def test_shapes_sit_near_well_known_models(known, layers, width):
    target = count_params(ModelConfig(layers, width, width // 64, **known))
    config = shape_for(target, known)
    assert abs(count_params(config) / target - 1) < 0.02
    assert width / 1.5 < config.d_model < width * 1.5
    assert layers / 2.6 < config.n_layer < layers * 1.6


def test_what_you_set_is_kept_and_the_rest_fits_around_it():
    config = shape_for(50e6, {"n_layer": 4, "ctx_len": 256, "vocab_size": 300})
    assert (config.n_layer, config.ctx_len, config.vocab_size) == (4, 256, 300)
    assert abs(count_params(config) / 50e6 - 1) < 0.05
    assert shape_for(10e6, {"d_model": 512, "n_head": 4}).n_head == 4


def test_tiny_models_use_small_heads_short_windows_and_small_vocabularies():
    small = shape_for(2e6)
    assert small.head_dim == 32 and small.ctx_len == 256 and small.vocab_size == 4096
    big = shape_for(2e9)
    assert big.head_dim == 64 and big.ctx_len == 2048 and big.vocab_size == 32768


def test_the_planner_is_quick():
    began = time.perf_counter()
    for target in (1e6, 1e8, 5e9):
        shape_for(target)
    assert time.perf_counter() - began < 1.0


def test_the_pass_size_is_the_most_that_fits():
    config = ModelConfig(**SMALL)
    free = memory_bytes(config, 5) + 1
    assert pass_size(config, 64, int(free / 0.9)) == 5
    assert pass_size(config, 3, 10**12) == 3
    assert pass_size(config, 64, memory_bytes(config, 1) // 2) == 0


def test_the_amount_of_text_comes_from_the_newest_training_set_else_a_guess(
    db, make_bucket, make_document
):
    bucket = make_bucket()
    make_document(bucket.id, chars=4000)
    make_document(bucket.id, chars=400, status="failed")
    assert planner.estimate_tokens(db, bucket.id) == (1000, False)
    snapshot = Dataset(name="snap", bucket_id=bucket.id, status="ready")
    db.add(snapshot)
    db.flush()
    vocabulary = Tokenizer(
        name="v", dataset_id=snapshot.id, dataset_name="snap", requested_vocab_size=300
    )
    db.add(vocabulary)
    db.flush()
    stats = {"train": {"documents": 1, "tokens": 777, "bytes": 1}}
    db.add(
        DatasetBuild(
            name="b",
            tokenizer_id=vocabulary.id,
            dataset_id=snapshot.id,
            dataset_name="snap",
            status="ready",
            stats=stats,
        )
    )
    db.commit()
    assert planner.estimate_tokens(db, bucket.id) == (777, True)


def test_a_new_model_plan_explains_every_choice(db, text, computer):
    computer(card(70))
    out = ask(db, bucket_id=text(400_000_000).id, parameters=20_000_000)
    assert abs(out.parameters / 20e6 - 1) < 0.05
    assert out.train_tokens == 20 * out.parameters
    for key in (
        "parameters",
        "n_layer",
        "d_model",
        "n_head",
        "ctx_len",
        "vocab_size",
        "train_tokens",
        "batch_size",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "dropout",
    ):
        assert out.why[key], key
    assert "20 word pieces" in out.why["train_tokens"]
    assert out.data.passes is not None and out.data.passes == pytest.approx(1.0, rel=0.05)
    assert out.data.enough and out.dropout == 0.0
    assert out.memory.fits and out.memory.cards == 1 and out.memory.sequences_per_pass >= 1
    steps = out.settings
    assert steps.max_steps * steps.batch_size * out.shape.ctx_len >= out.train_tokens
    assert 1 <= steps.eval_every <= steps.save_every <= steps.max_steps
    assert out.blockers == []


def test_bigger_models_learn_with_smaller_steps(db, computer):
    small, big = ask(db, parameters=5_000_000), ask(db, parameters=1_000_000_000)
    assert big.settings.learning_rate < small.settings.learning_rate
    assert big.settings.batch_size * big.shape.ctx_len > small.settings.batch_size * 256


def test_too_little_text_warns_and_says_how_much_more(db, text, computer):
    out = ask(db, bucket_id=text(1_000_000).id, parameters=10_000_000)
    assert out.data.passes is not None and out.data.passes > 100
    assert out.data.enough is False and out.dropout == 0.1
    assert any(
        "learn it by heart" in line and "more, or choose a smaller model" in line
        for line in out.warnings
    )


def test_choices_you_make_are_kept(db, computer):
    out = ask(
        db, parameters=10_000_000, batch_size=24, learning_rate=0.01, dropout=0.05, train_tokens=1e6
    )
    assert out.settings.batch_size == 24 and out.settings.learning_rate == 0.01
    assert out.dropout == 0.05 and out.train_tokens == 1_000_000
    assert out.why["batch_size"] == "You chose this."


def test_hours_pick_the_biggest_model_that_fits_the_time(db, computer):
    computer(card(70))
    speed_row(db, 1_000_000)
    one, ten = ask(db, hours=1), ask(db, hours=10)
    assert one.time.source == "scaled" and ten.parameters > one.parameters
    for out, hours in ((one, 1), (ten, 10)):
        assert out.time.hours is not None and out.time.hours <= hours * 1.05
        assert out.time.hours > hours * 0.8
    # scaled by the work per word piece of each shape
    shape = ModelConfig(**one.shape.model_dump())
    speed = 1_000_000 * work_per_token(ModelConfig(**SMALL)) / work_per_token(shape)
    assert one.time.tokens_per_sec == pytest.approx(speed)


def test_a_speed_test_of_the_same_shape_is_measured(db, computer):
    computer(card(70))
    out = ask(db, parameters=10_000_000)
    speed_row(db, 50_000, shape=out.shape.model_dump())
    again = ask(db, parameters=10_000_000)
    assert again.time.source == "measured" and again.time.tokens_per_sec == 50_000


def test_a_second_card_adds_speed(db, computer):
    speed_row(db, 1e6)
    computer(card(70))
    one = ask(db, parameters=50_000_000)
    computer(card(70), card(70))
    two = ask(db, parameters=50_000_000)
    assert two.memory.cards == 2 and two.memory.card_name == "2 x NVIDIA A100-SXM4-80GB"
    assert one.time.tokens_per_sec is not None and two.time.tokens_per_sec is not None
    assert 1.5 < two.time.tokens_per_sec / one.time.tokens_per_sec < 2.0


def test_without_a_speed_test_the_time_is_a_marked_guess(db, computer):
    computer(card(3, 4, "NVIDIA GeForce RTX 3050 Laptop GPU"))
    out = ask(db, hours=2)
    assert out.time.source == "none" and out.time.hours is not None
    assert any("speed test in Settings" in line for line in out.warnings)


def test_a_model_too_big_for_the_card_is_blocked_plainly(db, computer):
    computer(card(3, 4, "NVIDIA GeForce RTX 3050 Laptop GPU"))
    out = ask(db, parameters=5_000_000_000)
    assert out.memory.fits is False
    assert any("smaller size" in line and "bigger graphics card" in line for line in out.blockers)
    # hours never pick more than fits
    assert ask(db, hours=10_000).memory.fits


def test_a_processor_only_computer_gets_a_warning_not_a_blocker(db, computer):
    out = ask(db, parameters=2_000_000)
    assert out.memory.fits is None and out.memory.cards == 0 and out.blockers == []
    assert any("processor" in line for line in out.warnings)


def test_no_size_given_suits_the_collection(db, text, computer):
    computer(card(70))
    out = ask(db, bucket_id=text(200_000_000).id)
    assert out.parameters <= 200_000_000 / 20 * 1.05
    assert out.parameters >= 1_000_000


def test_a_fine_tune_keeps_the_base_shape(db, text, computer, make_run):
    computer(card(70))
    base = model_row(db, n_layer=4, d_model=256, n_head=4, ctx_len=256, vocab_size=8192)
    run = make_run(base.id)
    db.add(Checkpoint(run_id=run.id, step=10, uri="x.pt", val_loss=2.0))
    db.commit()
    out = ask(db, bucket_id=text(1_000_000, kind="chat").id, base_model_id=base.id)
    assert out.fine_tune and out.blockers == []
    assert (out.shape.n_layer, out.shape.d_model, out.shape.vocab_size) == (4, 256, 8192)
    assert out.train_tokens == 3_000_000
    new = ask(db, parameters=out.parameters)
    assert out.settings.learning_rate < new.settings.learning_rate
    assert out.settings.warmup_steps <= 100
    assert "fine-tune keeps the shape" in out.why["n_layer"]


def test_hub_models_and_untrained_models_cannot_be_fine_tuned(db, text, computer):
    bucket = text(1_000_000)
    hub = model_row(db, source="hub")
    out = ask(db, bucket_id=bucket.id, base_model_id=hub.id)
    assert any("Hugging Face" in line for line in out.blockers)
    fresh = model_row(db)
    out = ask(db, bucket_id=bucket.id, base_model_id=fresh.id)
    assert any("Train it first" in line for line in out.blockers)


def test_an_empty_collection_and_missing_torch_are_blockers(db, text, computer):
    out = ask(db, bucket_id=text(0).id, parameters=1_000_000)
    assert any("no text yet" in line for line in out.blockers)
    computer(torch="missing")
    assert any("PyTorch is not installed" in line for line in ask(db).blockers)


def test_the_plan_endpoint_answers(client, computer):
    response = client.post("/api/plan", json={"parameters": 3_000_000, "device": "cpu"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["shape"]["n_layer"] >= 1 and body["time"]["source"] == "none"
    assert math.isclose(body["parameters"], 3e6, rel_tol=0.05)
    bad = client.post("/api/plan", json={"parameters": 5, "hours": 1})
    assert bad.status_code == 422


@pytest.mark.parametrize("asked", [{"hours": 2}, {}, {"parameters": 3_000_000}])
def test_a_width_the_heads_cannot_split_is_a_blocker_not_a_crash(client, computer, asked):
    body = {"d_model": 96, "n_head": 5, "device": "cpu", **asked}
    response = client.post("/api/plan", json=body)
    assert response.status_code == 200, response.text
    assert any("No shape fits" in line for line in response.json()["blockers"])
