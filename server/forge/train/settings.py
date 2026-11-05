import math

from pydantic import BaseModel, Field, model_validator


class TrainSettings(BaseModel):
    batch_size: int = Field(default=16, ge=1, le=4096)
    micro_batch_size: int = Field(default=0, ge=0, le=4096)
    max_steps: int = Field(default=1000, ge=1, le=10_000_000)
    learning_rate: float = Field(default=6e-4, gt=0.0, le=1.0)
    warmup_steps: int = Field(default=-1, ge=-1)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0, le=100.0)
    eval_every: int = Field(default=100, ge=1)
    eval_tokens: int = Field(default=100_000, ge=1)
    save_every: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def _micro_divides(self) -> "TrainSettings":
        if self.micro_batch_size and self.batch_size % self.micro_batch_size:
            raise ValueError(
                f"Sequences per pass ({self.micro_batch_size}) must divide the sequences per "
                f"step ({self.batch_size}) evenly."
            )
        return self

    def warmup(self) -> int:
        if self.warmup_steps >= 0:
            return min(self.warmup_steps, self.max_steps - 1)
        return min(self.max_steps - 1, max(1, min(1000, self.max_steps // 20)))


def learning_rate(step: int, settings: TrainSettings) -> float:
    warmup = settings.warmup()
    if step < warmup:
        return settings.learning_rate * (step + 1) / warmup
    fraction = (step - warmup) / max(1, settings.max_steps - warmup - 1)
    fraction = min(1.0, fraction)
    return settings.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)


def steps_for_tokens(tokens: int, batch_size: int, window: int) -> int:
    return max(1, math.ceil(tokens / (batch_size * window)))
