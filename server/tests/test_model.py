import math

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
    with pytest.raises(ValueError, match="divide evenly"):
        presets.ModelConfig(n_layer=2, d_model=30, n_head=4, ctx_len=16)

    with pytest.raises(ValueError, match="ctx_len"):
        presets.validate(presets.get_preset("nano", ctx_len=333))

    with pytest.raises(ValueError, match="unknown preset"):
        presets.get_preset("enormous")


def test_vram_estimate_grows_with_size():
    nano = presets.estimate_vram_bytes(presets.PRESETS["nano"], micro_batch_seqs=16)
    small = presets.estimate_vram_bytes(presets.PRESETS["small"], micro_batch_seqs=16)
    assert 0 < nano < small
