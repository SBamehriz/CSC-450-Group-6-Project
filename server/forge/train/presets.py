from dataclasses import dataclass

MAX_N_LAYER = 128
MAX_D_MODEL = 8192
MAX_DROPOUT = 0.2
MIN_CTX = 64
MAX_CTX = 8192
# token files hold unsigned 16-bit numbers
MAX_VOCAB = 65535
CTX_MULTIPLE = 64


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
            raise ValueError(
                f"A width of {self.d_model:,} can't be split evenly between {self.n_head} "
                "attention heads: choose a width that divides by the number of heads."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


def validate(config: ModelConfig) -> None:
    blocks = config.n_layer
    if not 1 <= blocks <= MAX_N_LAYER:
        raise ValueError(
            f"This model would have {blocks} block{'' if blocks == 1 else 's'}, and Forge's "
            f"GPT has 1 to {MAX_N_LAYER}."
        )
    if not 1 <= config.d_model <= MAX_D_MODEL:
        raise ValueError(
            f"This model would have a width of {config.d_model:,}, and Forge's GPT has a "
            f"width of 1 to {MAX_D_MODEL:,}."
        )
    if not MIN_CTX <= config.ctx_len <= MAX_CTX or config.ctx_len % CTX_MULTIPLE:
        raise ValueError(
            f"This model would read a context of {config.ctx_len:,} word pieces, and Forge's "
            f"GPT reads {MIN_CTX} to {MAX_CTX:,}, in steps of {CTX_MULTIPLE}."
        )
    if not 0.0 <= config.dropout <= MAX_DROPOUT:
        raise ValueError(f"Dropout goes from 0, the default, to {MAX_DROPOUT}.")


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


HALF_BYTES = 2
TRAINED_BYTES = 16
LOGIT_BYTES = 13
MATH_ATTENTION_BYTES = 7
CUDA_CONTEXT_BYTES = 500_000_000


def activation_bytes_per_token(hidden: int, intermediate: int) -> int:
    return 18 * hidden + 4 * intermediate


def training_memory(config: ModelConfig, micro_batch_seqs: int = 8) -> dict[str, int]:
    tokens = micro_batch_seqs * config.ctx_len
    d = config.d_model
    block = activation_bytes_per_token(d, 4 * d)
    scores = micro_batch_seqs * config.n_head * config.ctx_len**2 * config.n_layer
    return {
        "weights": TRAINED_BYTES * count_params(config),
        "weight_copies": HALF_BYTES * count_params(config),
        "activations": tokens * block * config.n_layer,
        "attention": 0 if config.head_dim % 8 == 0 else MATH_ATTENTION_BYTES * scores,
        "logits": LOGIT_BYTES * tokens * config.vocab_size,
        "cuda_context": CUDA_CONTEXT_BYTES,
    }


def estimate_vram_bytes(config: ModelConfig, micro_batch_seqs: int = 8) -> int:
    """Estimated total training memory in bytes."""
    return sum(training_memory(config, micro_batch_seqs).values())


def explain_shape_error(
    n_layer: int,
    d_model: int,
    n_head: int,
    ctx_len: int,
    vocab_size: int = 16384,
    dropout: float = 0.0,
) -> str | None:
    """Explains why a model configuration shape is invalid in plain language."""
    if n_head <= 0:
        return "Number of attention heads must be at least 1."
    if d_model % n_head != 0:
        head_dim = d_model / n_head
        return (
            f"Model width ({d_model}) must divide evenly by attention heads ({n_head}). "
            f"Each head would get {head_dim:.1f} dimensions, but head dimensions must be integers."
        )
    if not 1 <= n_layer <= MAX_N_LAYER:
        return f"Layer count (blocks) must be between 1 and {MAX_N_LAYER} (got {n_layer})."
    if not 1 <= d_model <= MAX_D_MODEL:
        return f"Model width (d_model) must be between 1 and {MAX_D_MODEL} (got {d_model})."
    if not MIN_CTX <= ctx_len <= MAX_CTX or ctx_len % CTX_MULTIPLE:
        return (
            f"Context length must be {MIN_CTX} to {MAX_CTX:,} tokens, in steps of "
            f"{CTX_MULTIPLE} (got {ctx_len})."
        )
    if not 0.0 <= dropout <= MAX_DROPOUT:
        return f"Dropout must be between 0.0 and {MAX_DROPOUT} (got {dropout})."
    if vocab_size < 2:
        return f"Vocabulary size must be at least 2 tokens (got {vocab_size})."
    return None
