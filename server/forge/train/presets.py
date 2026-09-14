from dataclasses import dataclass, replace

MAX_N_LAYER = 12
MAX_D_MODEL = 768
ALLOWED_CTX = (256, 512, 1024)
MAX_DROPOUT = 0.2


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    ctx_len: int
    vocab_size: int = 16384
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model {self.d_model} must divide evenly by n_head {self.n_head}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


PRESETS: dict[str, ModelConfig] = {
    "nano": ModelConfig(n_layer=4, d_model=128, n_head=4, ctx_len=512),
    "micro": ModelConfig(n_layer=6, d_model=256, n_head=8, ctx_len=512),
    "small": ModelConfig(n_layer=8, d_model=512, n_head=8, ctx_len=512),
}


def get_preset(name: str, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; pick one of {sorted(PRESETS)}")
    config = PRESETS[name]
    return replace(config, **overrides) if overrides else config


def validate(config: ModelConfig) -> None:
    if not 1 <= config.n_layer <= MAX_N_LAYER:
        raise ValueError(f"n_layer must be between 1 and {MAX_N_LAYER}")
    if not 1 <= config.d_model <= MAX_D_MODEL:
        raise ValueError(f"d_model must be between 1 and {MAX_D_MODEL}")
    if config.ctx_len not in ALLOWED_CTX:
        raise ValueError(f"ctx_len must be one of {ALLOWED_CTX}")
    if not 0.0 <= config.dropout <= MAX_DROPOUT:
        raise ValueError(f"dropout must be between 0 and {MAX_DROPOUT}")


def count_params(config: ModelConfig) -> int:
    # no torch, so the api can call it
    c = config
    # tied lm head adds nothing
    embeddings = c.vocab_size * c.d_model + c.ctx_len * c.d_model
    # qkv, out proj, mlp, two norms
    attn = 4 * c.d_model * c.d_model + 4 * c.d_model
    mlp = 8 * c.d_model * c.d_model + 5 * c.d_model
    norms = 4 * c.d_model
    per_block = attn + mlp + norms
    final_norm = 2 * c.d_model
    return embeddings + c.n_layer * per_block + final_norm


def estimate_vram_bytes(config: ModelConfig, micro_batch_seqs: int) -> int:
    # ~16 bytes/param for weights, grads, adam
    optimizer_and_weights = 16 * count_params(config)
    activations = micro_batch_seqs * config.ctx_len * config.d_model * config.n_layer * 16
    return optimizer_and_weights + activations


def describe(config: ModelConfig) -> str:
    params = count_params(config)
    return (
        f"{config.n_layer}L x {config.d_model}d x {config.n_head}h, "
        f"ctx {config.ctx_len}, vocab {config.vocab_size} -> {params / 1e6:.2f}M params"
    )
