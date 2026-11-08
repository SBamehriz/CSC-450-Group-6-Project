import uuid
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge.train.presets import MAX_D_MODEL, MAX_N_LAYER
from forge.train.settings import TrainSettings

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int


class BucketStats(BaseModel):
    documents: int
    parsed_documents: int
    chars: int
    est_tokens: int


class BucketCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("cannot be blank")
        return value


class BucketUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("cannot be blank")
        return value


class BucketOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    created_at: datetime
    stats: BucketStats
    source_dataset: str | None = None
    resolved_revision: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    text_column: str | None = None
    import_limit: int | None = None


class DocumentOut(BaseModel):
    id: uuid.UUID
    bucket_id: uuid.UUID
    filename: str
    source_format: str
    source_note: str
    status: str
    error: str | None
    content_hash: str | None
    char_count: int
    word_count: int
    quality: dict
    created_at: datetime


class RecentUpload(DocumentOut):
    bucket_name: str


class OverviewStats(BaseModel):
    buckets: int
    documents: int
    parsed_documents: int
    failed_documents: int
    rejected_documents: int
    pending_documents: int
    flagged_documents: int
    chars: int
    words: int
    est_tokens: int
    snapshots: int
    ready_snapshots: int


class UploadOutcome(BaseModel):
    """One line per file the user handed us, in the order they sent them."""

    filename: str
    status: str  # parsed | failed
    document_id: uuid.UUID | None = None
    error: str | None = None


class UploadResult(BaseModel):
    parsed: int
    failed: int
    outcomes: list[UploadOutcome]


class DocumentText(BaseModel):
    """A window into the cleaned text, never the whole 50MB document."""

    text: str
    offset: int
    length: int
    total: int


class MoveDocument(BaseModel):
    bucket_id: uuid.UUID


class RejectDocument(BaseModel):
    reason: str = Field(default="", max_length=500)

    @field_validator("reason")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class Health(BaseModel):
    status: str
    database: str


class DatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)
    bucket_id: uuid.UUID
    validation_fraction: float = Field(default=0.1, gt=0, le=0.5)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("cannot be blank")
        return value


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    bucket_id: uuid.UUID | None
    status: str
    doc_count: int
    char_count: int
    token_count: int
    config: dict
    artifact_uri: str | None
    created_at: datetime
    download_ready: bool = False
    readiness_note: str = ""
    job_id: uuid.UUID | None = None


class SnapshotCheck(BaseModel):
    valid: bool
    message: str
    train_documents: int = 0
    validation_documents: int = 0


class JobCreate(BaseModel):
    job_type: str
    payload: dict = Field(default_factory=dict)


class PrepareCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vocab_size: int = Field(default=4096, ge=257, le=16384)


class PrepareTools(BaseModel):
    available: bool
    max_text_mb: int


class PreparedOut(BaseModel):
    ready: bool
    message: str = ""
    vocab_size: int = 0
    requested_vocab_size: int = 0
    train_tokens: int = 0
    validation_tokens: int = 0
    trained_on_documents: int = 0
    eos_id: int = 0
    sha256: str = ""
    snapshot_sha256: str = ""


class TokenPreviewCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TokenPreview(BaseModel):
    token_count: int
    ids: list[int]
    pieces: list[str]


class PresetMemory(BaseModel):
    sequences_per_step: int
    bytes: int


class PresetOut(BaseModel):
    name: str
    n_layer: int
    d_model: int
    n_head: int
    ctx_len: int
    vocab_size: int
    parameters: int
    memory: list[PresetMemory]


class GpuCard(BaseModel):
    index: int
    name: str
    memory_bytes: int
    free_bytes: int
    driver: str
    compute_capability: str


class GpuCheck(BaseModel):
    cards: list[GpuCard]
    tool_said: str | None
    torch: Literal["cuda", "rocm", "cpu", "missing", "unknown"]
    torch_version: str | None
    torch_cuda: str | None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_type: str
    status: str
    payload: dict
    progress: dict
    error: str | None
    attempts: int = 0
    cancel_requested: bool = False
    result: dict = Field(default_factory=dict)
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class JobPayload(BaseModel):
    target_id: uuid.UUID
    target_name: str


class JobProgress(BaseModel):
    completed: int = 0
    total: int = 0
    message: str = ""
    processed: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0


class HfInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=200)

    @field_validator("dataset")
    @classmethod
    def _strip_dataset(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("cannot be blank")
        return cleaned


class HfSplitInfo(BaseModel):
    name: str
    num_examples: int | None = None


class HfConfigInfo(BaseModel):
    name: str
    splits: list[HfSplitInfo]
    columns: list[str]
    text_columns: list[str]


class HfInspectResponse(BaseModel):
    dataset: str
    canonical_id: str
    sha: str
    configs: list[HfConfigInfo]


class DatasetImportCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=200)
    config: str = Field(min_length=1, max_length=100)
    split: str = Field(min_length=1, max_length=100)
    text_column: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=500, ge=1, le=50000)

    @field_validator("dataset", "config", "split", "text_column")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("cannot be blank")
        return cleaned


class SnapshotSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    dataset_id: uuid.UUID

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cannot be blank")
        return value.strip()


class TokenizerCreate(SnapshotSource):
    vocab_size: int = Field(default=4096, ge=257, le=65535)


class DatasetBuildCreate(SnapshotSource):
    tokenizer_id: uuid.UUID


class TokenizerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    dataset_id: uuid.UUID | None
    dataset_name: str
    requested_vocab_size: int
    vocab_size: int | None
    trained_on: dict
    status: str
    error: str | None
    created_at: datetime


class SplitStats(BaseModel):
    documents: int = 0
    tokens: int = 0
    bytes: int = 0


class DatasetBuildOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    tokenizer_id: uuid.UUID
    dataset_id: uuid.UUID | None
    dataset_name: str
    status: str
    stats: dict[str, SplitStats]
    error: str | None
    created_at: datetime


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=2000)
    preset: str | None = None
    dataset_id: uuid.UUID | None = None
    tokenizer_id: uuid.UUID | None = None
    n_layer: int = Field(ge=1, le=MAX_N_LAYER)
    d_model: int = Field(ge=1, le=MAX_D_MODEL)
    n_head: int = Field(ge=1)
    ctx_len: int = Field(ge=128, le=2048)
    vocab_size: int = 16384
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    config: dict = Field(default_factory=dict)


class ModelPreviewRequest(BaseModel):
    preset: str | None = None
    dataset_id: uuid.UUID | None = None
    tokenizer_id: uuid.UUID | None = None
    n_layer: int = Field(default=4, ge=1, le=MAX_N_LAYER)
    d_model: int = Field(default=128, ge=1, le=MAX_D_MODEL)
    n_head: int = Field(default=4, ge=1)
    ctx_len: int = Field(default=512)
    vocab_size: int | None = None
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)


class ModelPreviewOut(BaseModel):
    valid: bool
    param_count: int = 0
    estimated_memory_bytes: int = 0
    estimated_memory_mb: float = 0.0
    vocab_size: int = 0
    error: str | None = None


class VocabularyOut(BaseModel):
    dataset_id: uuid.UUID
    dataset_name: str
    vocab_size: int
    requested_vocab_size: int
    train_tokens: int
    trained_on_documents: int


class ModelOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    preset: str | None
    n_layer: int
    d_model: int
    n_head: int
    ctx_len: int
    vocab_size: int
    dropout: float
    param_count: int
    config: dict
    tokenizer_id: uuid.UUID | None = None
    source: str = "forge"
    estimated_memory_bytes: int | None = None
    created_at: datetime


class HubImportRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200)
    revision: str = Field(default="main", min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=64)
    # it does not fit the card, run it on the processor instead
    on_cpu: bool = False


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    model_id: uuid.UUID
    dataset_id: uuid.UUID | None = None
    # the training set to train on; needed before the run can start
    build_id: uuid.UUID | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    train_tokens: int | None = Field(default=None, ge=1)
    settings: TrainSettings = Field(default_factory=TrainSettings)
    batch_size: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0.0)
    max_steps: int | None = Field(default=None, ge=1)
    hyperparams: dict = Field(default_factory=dict)


class RunOut(BaseModel):
    id: uuid.UUID
    name: str
    model_id: uuid.UUID
    model_name: str | None = None
    model_preset: str | None = None
    dataset_id: uuid.UUID | None
    dataset_name: str | None = None
    build_id: uuid.UUID | None = None
    build_name: str | None = None
    seed: int = 0
    tokens_seen: int = 0
    settings: TrainSettings | None = None
    job_id: uuid.UUID | None = None
    job_status: str | None = None
    status: str
    device: str
    batch_size: int
    learning_rate: float
    max_steps: int
    current_step: int
    loss: float | None
    eval_loss: float | None
    best_val_loss: float | None = None
    bpb: float | None = None
    bpb_details: dict | None = None
    tokens_per_sec: float | None
    metrics: dict
    hyperparams: dict
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class CheckpointOut(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    step: int
    epoch: int
    loss: float | None
    val_loss: float | None
    uri: str
    size_bytes: int
    is_best: bool
    metrics: dict
    sha256: str | None = None
    created_at: datetime


class MetricPointOut(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    step: int
    loss: float | None
    val_loss: float | None
    learning_rate: float | None
    speed: float | None
    tokens_per_sec: float | None = None
    created_at: datetime


class RunMetricsOut(BaseModel):
    run_id: uuid.UUID
    items: list[MetricPointOut]
    total: int


class PromptSetCreate(BaseModel):
    name: str
    description: str = ""
    prompts: list[str]


class PromptSetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    prompts: list[str] | None = None


class PromptSetOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    prompts: list[str]
    prompt_count: int
    created_at: datetime
    updated_at: datetime


class EvaluationCreate(BaseModel):
    name: str | None = None
    model_id: uuid.UUID
    prompt_set_id: uuid.UUID
    max_new_tokens: int = 32
    temperature: float = 0.7


class PromptEvalResult(BaseModel):
    prompt: str
    output: str
    tokens_generated: int = 0


class EvaluationOut(BaseModel):
    id: uuid.UUID
    name: str
    model_id: uuid.UUID
    model_name: str
    prompt_set_id: uuid.UUID | None
    prompt_set_name: str | None
    captured_prompts: list[str]
    results: list[dict]
    status: str
    error: str | None
    job_id: uuid.UUID | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class BpbEvaluationCreate(BaseModel):
    name: str | None = None
    model_id: uuid.UUID | None = None
    imported_model_name: str | None = None
    model_source: str = "native"
    checkpoint_id: uuid.UUID | None = None
    dataset_id: uuid.UUID
    ctx_len: int | None = None


class BpbEvaluationOut(BaseModel):
    id: uuid.UUID
    name: str
    model_id: uuid.UUID | None
    model_name: str
    model_source: str
    checkpoint_id: uuid.UUID | None
    dataset_id: uuid.UUID | None
    dataset_name: str
    validation_file_hash: str
    tokenizer_identity: str
    bpb: float | None
    total_nll: float | None
    scored_bytes: int | None
    scored_tokens: int | None
    status: str
    error: str | None
    settings: dict
    job_id: uuid.UUID | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class ShapeIn(BaseModel):
    n_layer: int = Field(ge=1, le=MAX_N_LAYER)
    d_model: int = Field(ge=1, le=MAX_D_MODEL)
    n_head: int = Field(ge=1)
    ctx_len: int = Field(ge=64, le=2048)
    vocab_size: int = Field(default=16384, ge=2, le=262_144)


class SpeedTestCreate(BaseModel):
    preset: str | None = None
    shape: ShapeIn | None = None
    sequences_per_step: int = Field(default=4, ge=1, le=256)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class SpeedTestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    preset: str | None
    shape: dict
    sequences_per_step: int
    device: str
    device_name: str
    tokens_per_sec: float
    peak_memory_bytes: int | None
    steps: int
    created_at: datetime


class EstimateRequest(BaseModel):
    preset: str | None = None
    shape: ShapeIn | None = None
    train_tokens: int = Field(ge=1)
    sequences_per_step: int = Field(default=4, ge=1, le=256)


class EstimateOut(BaseModel):
    parameters: int
    train_tokens: int
    speed_source: Literal["measured", "scaled", "none"]
    speed_test_id: uuid.UUID | None
    device_name: str | None
    tokens_per_sec: float | None
    hours: float | None
    memory_bytes: int
    free_bytes: int | None
    fits: bool | None
    suggestion: str | None


class BenchmarkInfo(BaseModel):
    task: str
    name: str
    chance: float


class BenchmarkRefusal(BaseModel):
    task: str
    name: str
    reason: str


class BenchmarkPlanOut(BaseModel):
    parameters: int
    allowed: list[BenchmarkInfo]
    refused: list[BenchmarkRefusal]


class BenchmarkCreate(BaseModel):
    model_id: uuid.UUID
    checkpoint_id: uuid.UUID | None = None
    tasks: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=100_000)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class BenchmarkScore(BaseModel):
    task: str
    name: str
    metric: str
    score: float
    low: float | None
    high: float | None
    chance: float
    questions: int | None = None


class BenchmarkResult(BaseModel):
    job_id: uuid.UUID
    finished_at: datetime | None
    checkpoint_id: uuid.UUID | None
    limit: int | None
    harness_version: str | None
    scores: list[BenchmarkScore]
    refused: list[BenchmarkRefusal]


class PromptCompareCreate(BaseModel):
    model_ids: list[uuid.UUID] = Field(min_length=2, max_length=2)
    prompts: list[str] = Field(default_factory=list, max_length=20)
    prompt_set_id: uuid.UUID | None = None
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=1000)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @field_validator("prompts")
    @classmethod
    def _real_prompts(cls, prompts: list[str]) -> list[str]:
        kept = [prompt for prompt in prompts if prompt.strip()]
        if any(len(prompt) > 4000 for prompt in kept):
            raise ValueError("Keep each prompt under 4,000 characters.")
        return kept


class PromptRow(BaseModel):
    prompt: str
    answers: list[str]


class PromptCompareResult(BaseModel):
    job_id: uuid.UUID
    finished_at: datetime | None
    settings: dict
    model_names: list[str]
    rows: list[PromptRow]


class CompareModel(BaseModel):
    id: uuid.UUID
    name: str
    source: str
    parameters: int
    vocab_size: int
    ctx_len: int
    license: str | None
    tokens_seen: int | None
    best_val_loss: float | None
    bits_per_byte: float | None
    benchmark: BenchmarkResult | None


class CompareOut(BaseModel):
    models: list[CompareModel]
    prompt_runs: list[PromptCompareResult]


class EvalResultOut(BaseModel):
    id: uuid.UUID
    result_type: str
    model_id: uuid.UUID | None
    model_name: str
    checkpoint_id: uuid.UUID | None
    evaluation_id: uuid.UUID | None
    settings: dict[str, Any]
    code_version: str | None
    prompt: str | None
    answer: str | None
    bpb: float | None
    total_nll: float | None
    scored_bytes: int | None
    scored_tokens: int | None
    benchmark_name: str | None
    score: float | None = None
    benchmark_score: float | None
    benchmark_scores: dict[str, Any]
    created_at: datetime
