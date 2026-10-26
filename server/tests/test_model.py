import math
from dataclasses import replace

import pytest

from forge.train import presets

torch = pytest.importorskip("torch", reason="torch is an optional extra: uv sync --extra train")

from forge.train.model import GPT  # noqa: E402

TINY = presets.ModelConfig(n_layer=2, d_model=32, n_head=4, ctx_len=16, vocab_size=64)


def test_presets_are_the_sizes_we_advertise():
    sizes = {name: presets.count_params(cfg) / 1e6 for name, cfg in presets.PRESETS.items()}
    assert 2.5 < sizes["nano"] < 3.5
    assert 8.0 < sizes["micro"] < 10.0
    assert 30.0 < sizes["small"] < 36.0
    assert 95.0 < sizes["large"] < 100.0
    assert 130.0 < sizes["135m"] < 136.0


def test_param_formula_matches_the_real_model():
    model = GPT(TINY)
    assert model.num_params() == presets.count_params(TINY)


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_formula_matches_for_every_preset(name):
    config = presets.get_preset(name)
    model = GPT(config)
    assert model.num_params() == presets.count_params(config)


def test_forward_shapes_and_loss():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, TINY.ctx_len))
    y = torch.randint(0, TINY.vocab_size, (2, TINY.ctx_len))

    logits, loss = model(x, y)
    assert logits.shape == (2, TINY.ctx_len, TINY.vocab_size)
    assert loss is not None and math.isfinite(loss.item())


def test_untrained_loss_is_about_ln_vocab():
    # a fresh model should be maximally confused
    torch.manual_seed(0)
    config = presets.ModelConfig(n_layer=2, d_model=64, n_head=4, ctx_len=32, vocab_size=512)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (8, config.ctx_len))
    y = torch.randint(0, config.vocab_size, (8, config.ctx_len))
    _, loss = model(x, y)
    assert loss is not None
    assert abs(loss.item() - math.log(config.vocab_size)) < 0.3


def test_generation_only_returns_the_last_position():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (1, 4))
    logits, loss = model(x)
    assert loss is None
    assert logits.shape == (1, 1, TINY.vocab_size)


def test_attention_cannot_see_the_future():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 8))
    changed = x.clone()
    changed[0, -1] = (changed[0, -1] + 1) % TINY.vocab_size

    with torch.no_grad():
        a, _ = model(x, x)
        b, _ = model(changed, changed)
    assert torch.allclose(a[:, :-1, :], b[:, :-1, :], atol=1e-5)


def test_generate_extends_the_sequence():
    torch.manual_seed(0)
    model = GPT(TINY)
    start = torch.randint(0, TINY.vocab_size, (1, 3))
    out = model.generate(start, max_new_tokens=5, temperature=1.0, top_k=4)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], start)
    assert out.max().item() < TINY.vocab_size


def test_generate_never_exceeds_the_context_window():
    model = GPT(TINY)
    start = torch.randint(0, TINY.vocab_size, (1, TINY.ctx_len))
    out = model.generate(start, max_new_tokens=3)
    assert out.shape == (1, TINY.ctx_len + 3)


def test_too_long_a_sequence_is_a_clear_error():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (1, TINY.ctx_len + 1))
    with pytest.raises(ValueError, match="longer than ctx_len"):
        model(x, x)


def test_weight_tying_is_actually_tied():
    model = GPT(TINY)
    assert model.lm_head.weight is model.token_emb.weight


def test_optimizer_splits_decay_groups():
    model = GPT(TINY)
    optimizer = model.configure_optimizer(lr=1e-3)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # 1-d is biases and layernorms
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_one_step_actually_lowers_the_loss():
    torch.manual_seed(0)
    model = GPT(TINY)
    optimizer = model.configure_optimizer(lr=1e-2)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.ctx_len))
    y = torch.randint(0, TINY.vocab_size, (4, TINY.ctx_len))

    _, first = model(x, y)
    assert first is not None
    for _ in range(20):
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _, final = model(x, y)
    assert final is not None
    assert final.item() < first.item()


def test_bad_config_is_caught_early():
    with pytest.raises(ValueError) as problem:
        presets.ModelConfig(n_layer=2, d_model=30, n_head=4, ctx_len=16)
    assert str(problem.value) == (
        "A width of 30 can't be split evenly between 4 attention heads: choose a width "
        "that divides by the number of heads."
    )

    nano = presets.get_preset("nano")
    for changed, said in (
        (
            {"ctx_len": 333},
            "This model would read a context of 333 word pieces, and Forge's "
            "GPT reads 64 to 2,048, in steps of 64.",
        ),
        ({"n_layer": 25}, "This model would have 25 blocks, and Forge's GPT has 1 to 24."),
        ({"n_layer": 0}, "This model would have 0 blocks, and Forge's GPT has 1 to 24."),
        (
            {"d_model": 1024, "n_head": 8},
            "This model would have a width of 1,024, and Forge's GPT has a width of 1 to 768.",
        ),
        ({"dropout": 0.5}, "Dropout goes from 0, the default, to 0.2."),
    ):
        with pytest.raises(ValueError) as problem:
            presets.validate(replace(nano, **changed))
        assert str(problem.value) == said, changed

    with pytest.raises(ValueError, match="unknown preset"):
        presets.get_preset("enormous")


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_every_preset_is_a_valid_shape(name):
    presets.validate(presets.PRESETS[name])


MEASURED_PEAKS = [
    (replace(presets.PRESETS["nano"], vocab_size=16384), 16, 1_827_428_864),
    (replace(presets.PRESETS["micro"], vocab_size=16384), 16, 2_222_380_544),
    (replace(presets.PRESETS["small"], vocab_size=16384), 16, 3_356_820_992),
]


def test_training_memory_is_its_parts_worked_by_hand():
    config = replace(presets.PRESETS["nano"], vocab_size=16384)
    parts = presets.training_memory(config, micro_batch_seqs=16)
    assert parts == {
        "weights": 16 * 2_956_032,
        "weight_copies": 2 * 2_956_032,
        "activations": 16 * 512 * 4 * 34 * 128,
        "attention": 0,
        "logits": 13 * 16 * 512 * 16384,
        "cuda_context": 500_000_000,
    }
    assert sum(parts.values()) == 2_440_645_376


@pytest.mark.parametrize("config, micro, peak", MEASURED_PEAKS)
def test_training_memory_never_falls_under_a_measured_peak(config, micro, peak):
    parts = presets.training_memory(config, micro_batch_seqs=micro)
    assert sum(parts.values()) - parts["cuda_context"] >= peak


def test_the_bytes_a_logit_costs_are_read_off_the_measured_presets():
    left = []
    for config, micro, peak in MEASURED_PEAKS:
        parts = presets.training_memory(config, micro_batch_seqs=micro)
        logits = micro * config.ctx_len * config.vocab_size
        weights = parts["weights"] + parts["weight_copies"]
        left.append((peak - weights - parts["activations"]) / logits)
    assert [round(value, 1) for value in left] == [12.2, 12.2, 12.0]
    assert presets.LOGIT_BYTES == math.ceil(max(left)) == 13


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_the_presets_keep_no_attention_scores(name):
    config = presets.PRESETS[name]
    assert config.head_dim % 8 == 0
    assert presets.training_memory(config, micro_batch_seqs=16)["attention"] == 0


def test_an_odd_head_width_keeps_every_score_matrix():
    config = presets.ModelConfig(n_layer=2, d_model=200, n_head=8, ctx_len=4096)
    parts = presets.training_memory(config, micro_batch_seqs=3)
    assert parts["attention"] == 7 * 3 * 8 * 4096 * 4096 * 2 == 5_637_144_576
    fused = presets.ModelConfig(n_layer=2, d_model=200, n_head=25, ctx_len=4096)
    assert presets.training_memory(fused, micro_batch_seqs=3)["attention"] == 0


MEASURED_RESERVED = [
    (4, 128, 4, 512, 8192, 1, 127_926_272),
    (6, 256, 8, 2048, 8192, 2, 811_597_824),
    (8, 512, 8, 2048, 8192, 6, 3_552_575_488),
    (6, 256, 8, 2048, 8192, 10, 3_370_123_264),
    (6, 256, 8, 4096, 8192, 5, 3_370_123_264),
    (4, 128, 4, 8192, 8192, 1, 1_044_381_696),
    (4, 128, 4, 512, 262144, 1, 2_204_106_752),
    (4, 200, 8, 2048, 8192, 2, 2_367_684_608),
    (4, 128, 4, 512, 16384, 16, 1_843_396_608),
    (6, 256, 8, 512, 16384, 16, 2_281_701_376),
    (8, 512, 8, 512, 16384, 16, 3_391_094_784),
]


@pytest.mark.parametrize("layers, width, heads, ctx, vocab, micro, reserved", MEASURED_RESERVED)
def test_training_memory_covers_what_the_card_reserved(
    layers, width, heads, ctx, vocab, micro, reserved
):
    config = presets.ModelConfig(layers, width, heads, ctx, vocab)
    assert sum(presets.training_memory(config, micro_batch_seqs=micro).values()) >= reserved


def test_the_135m_size_memory_is_worked_out_in_full():
    config = presets.PRESETS["135m"]
    one = sum(presets.training_memory(config, micro_batch_seqs=1).values())
    eight = sum(presets.training_memory(config, micro_batch_seqs=8).values())
    assert one < eight
    assert eight == 8_290_856_704
