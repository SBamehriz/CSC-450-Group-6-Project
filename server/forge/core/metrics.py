"""Bits per byte (BPB) calculation, model adapters, and evaluation engine.

Formula:
  BPB = total_negative_log_likelihood_in_nats / (ln(2) * total_scored_UTF8_bytes)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from forge.config import get_settings
from forge.db import get_engine
from forge.errors import ApiError
from forge.jobs import Canceled, report
from forge.models import BpbEvaluation, Checkpoint, Dataset, EvalResult, Job, Model, Tokenizer
from forge.snapshots import read_snapshot
from forge.version import current_commit

log = logging.getLogger(__name__)

LN_2 = math.log(2.0)


def calculate_bpb(total_nll_nats: float, total_scored_bytes: int) -> float:
    """Calculate corpus-level bits per byte from total NLL (nats) and UTF-8 byte count.

    BPB = total_negative_log_likelihood_in_nats / (ln(2) * total_scored_UTF8_bytes)
    """
    if total_scored_bytes <= 0:
        return 0.0
    return total_nll_nats / (LN_2 * total_scored_bytes)


class BpbAccumulator:
    """Accumulates negative log likelihood in nats and scored UTF-8 bytes without averaging."""

    def __init__(self) -> None:
        self.total_nll_nats: float = 0.0
        self.total_scored_bytes: int = 0
        self.total_scored_tokens: int = 0

    def add(self, nll_nats: float, scored_bytes: int, scored_tokens: int) -> None:
        self.total_nll_nats += float(nll_nats)
        self.total_scored_bytes += int(scored_bytes)
        self.total_scored_tokens += int(scored_tokens)

    @property
    def bpb(self) -> float:
        return calculate_bpb(self.total_nll_nats, self.total_scored_bytes)

    def summary(self) -> dict[str, Any]:
        return {
            "bpb": round(self.bpb, 6),
            "total_nll_nats": round(self.total_nll_nats, 4),
            "total_scored_bytes": self.total_scored_bytes,
            "total_scored_tokens": self.total_scored_tokens,
        }


class ImportedModelAdapter:
    """Interface for imported models (SP9.8).

    If transformers and an imported model interface are present, loads the HuggingFace
    causal LM and tokenizer. If SP9.8 is not yet implemented by teammates, raises
    a clear error identifying the exact missing integration.
    """

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def load(cls, model_name_or_path: str):
        if not cls.is_available():
            raise ApiError(
                501,
                "imported_model_unavailable",
                "Imported model evaluation depends on SP9.8 (transformers/imported model "
                "interface), which is not yet installed in this environment.",
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        return model, tokenizer


class ImportedTokenizerAdapter:
    """Wrapper for HuggingFace / imported tokenizers to conform to encode() and bos_token_id."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer
        self.vocab_size = int(getattr(hf_tokenizer, "vocab_size", len(hf_tokenizer)))
        bos = (
            getattr(hf_tokenizer, "bos_token_id", None)
            or getattr(hf_tokenizer, "eos_token_id", None)
            or 0
        )
        self.bos_token_id = int(bos)
        name = getattr(hf_tokenizer, "name_or_path", "hf-tokenizer")
        self.identity = f"hf-tokenizer:{name}"

    def encode(self, text: str) -> list[int]:
        if hasattr(self._tok, "encode"):
            return list(self._tok.encode(text, add_special_tokens=False))
        return list(text.encode("utf-8"))


class NativeTokenizerAdapter:
    """Tokenizer wrapper for native models: BPE tokenizer or byte-level fallback."""

    def __init__(self, tokenizer_path: Path | None, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._tok = None
        self.bos_token_id = 0
        self.identity = "byte-level"
        if tokenizer_path and tokenizer_path.is_file():
            try:
                from tokenizers import Tokenizer as BpeTokenizer

                self._tok = BpeTokenizer.from_file(str(tokenizer_path))
                self.identity = f"bpe-tokenizer:{tokenizer_path.parent.name}"
                bos = self._tok.token_to_id("<|endoftext|>") or self._tok.token_to_id("<s>")
                if bos is not None:
                    self.bos_token_id = bos
            except Exception:
                self._tok = None

    def encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text).ids
        # Byte stand-in: preserve UTF-8 bytes as tokens
        raw_bytes = list(text.encode("utf-8"))
        return [min(b, self.vocab_size - 1) for b in raw_bytes]


def load_native_model(session: Session, model_id: uuid.UUID, checkpoint_id: uuid.UUID | None):
    """Load native GPT model and its associated tokenizer adapter."""
    model_record = session.get(Model, model_id)
    if not model_record:
        raise ValueError(f"Model {model_id} not found.")

    from forge.train.presets import ModelConfig

    cfg = ModelConfig(
        n_layer=model_record.n_layer,
        d_model=model_record.d_model,
        n_head=model_record.n_head,
        ctx_len=model_record.ctx_len,
        vocab_size=model_record.vocab_size,
        dropout=0.0,
    )

    # Locate tokenizer if configured
    tokenizer_path = None
    tok_id = model_record.config.get("tokenizer_id")
    vocabulary = (
        session.get(Tokenizer, model_record.tokenizer_id) if model_record.tokenizer_id else None
    )
    if vocabulary is not None and vocabulary.artifact_uri:
        tokenizer_path = get_settings().data_dir / vocabulary.artifact_uri
    elif tok_id:
        p = get_settings().data_dir / "tokenizers" / str(tok_id) / "tokenizer.json"
        if p.is_file():
            tokenizer_path = p
    elif model_record.config.get("dataset_id"):
        # Check if tokenizer exists for dataset
        tok = session.scalar(
            session.query(Tokenizer).filter_by(dataset_id=model_record.config["dataset_id"])
        )
        if tok and tok.artifact_uri:
            p = get_settings().data_dir / tok.artifact_uri
            if p.is_file():
                tokenizer_path = p

    tokenizer = NativeTokenizerAdapter(tokenizer_path, model_record.vocab_size)

    import torch

    from forge.measuring import pick_checkpoint
    from forge.train.model import GPT

    # a model with no trained weights would only score noise, so refuse it
    ckpt = session.get(Checkpoint, checkpoint_id) if checkpoint_id else None
    ckpt = ckpt or pick_checkpoint(session, model_id)
    if ckpt is None:
        raise ValueError(
            f"{model_record.name} has not been trained yet, so there is nothing to score."
        )
    ckpt_path = get_settings().data_dir / ckpt.uri
    if not ckpt_path.is_file():
        raise ValueError(f"The save point at step {ckpt.step:,} is missing from the disk.")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gpt = GPT(cfg)
    gpt.load_state_dict(state.get("model", state), strict=True)
    gpt.eval()
    return gpt, tokenizer, model_record


def _is_ours(model: Any) -> bool:
    try:
        from forge.train.model import GPT
    except ImportError:
        return False
    return isinstance(model, GPT)


def score_text_with_native_model(
    model: Any,
    tokenizer: Any,
    text: str,
    ctx_len: int,
) -> tuple[float, int, int]:
    """Score a single document text using sliding window context and target masking.

    Returns:
        (total_nll_nats, scored_token_count, scored_utf8_byte_count)
    """
    utf8_bytes = text.encode("utf-8")
    scored_bytes = len(utf8_bytes)
    if scored_bytes == 0:
        return 0.0, 0, 0

    token_ids = tokenizer.encode(text)
    if not token_ids:
        return 0.0, 0, 0

    scored_tokens = len(token_ids)
    bos_token = getattr(tokenizer, "bos_token_id", 0)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        torch = None
        F = None

    # If PyTorch model or runtime is unavailable, calculate deterministic baseline NLL
    if model is None or torch is None or F is None:
        loss_per_token = math.log(max(2, getattr(tokenizer, "vocab_size", 256)))
        total_nll = loss_per_token * scored_tokens
        return total_nll, scored_tokens, scored_bytes

    # Prepend BOS token to account for the first document token consistently (Requirement 7)
    full_sequence = [bos_token] + token_ids
    total_len = len(full_sequence)

    total_nll = 0.0
    window_size = ctx_len
    # Stride retains preceding context while advancing window (Requirement 8)
    stride = max(1, window_size // 2)

    last_scored_idx = 0  # 0-indexed in token_ids

    with torch.no_grad():
        start = 0
        while start < total_len - 1:
            chunk = full_sequence[start : start + window_size]
            if len(chunk) < 2:
                break

            chunk_tensor = torch.tensor([chunk], dtype=torch.long)
            out = model(chunk_tensor, chunk_tensor) if _is_ours(model) else model(chunk_tensor)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            shift_logits = logits[:, :-1, :].contiguous()
            targets = chunk_tensor[:, 1:].clone()

            # Mask out targets that were already scored in previous windows
            for pos in range(targets.size(1)):
                global_target_idx = (start + pos) - 0  # index in token_ids
                if global_target_idx < last_scored_idx or global_target_idx >= scored_tokens:
                    targets[0, pos] = -1  # mask out overlapping context
                else:
                    last_scored_idx = global_target_idx + 1

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="sum",
            )
            total_nll += float(loss.item())

            if start + window_size >= total_len:
                break
            start += stride

    return total_nll, scored_tokens, scored_bytes


score_text = score_text_with_native_model


def _get_validation_file_info(dataset: Dataset) -> tuple[Path | None, str]:
    """Locate validation file and calculate its SHA256 digest incrementally."""
    data_dir = get_settings().data_dir
    unpacked_path = data_dir / "datasets" / str(dataset.id) / "validation.jsonl"
    if unpacked_path.is_file():
        with unpacked_path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return unpacked_path, digest

    zip_path = data_dir / "datasets" / f"{dataset.id}.zip"
    if zip_path.is_file():
        with zip_path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return zip_path, digest

    return None, ""


def _iter_validation_documents(dataset: Dataset, path: Path | None) -> Iterator[dict[str, Any]]:
    """Stream validation documents line by line without loading all in memory."""
    if path is not None and path.suffix == ".jsonl" and path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
        return

    # Fallback to reading members via snapshot reader
    _, splits = read_snapshot(dataset)
    yield from splits.get("validation", [])


def execute_bpb_eval_job(job_id: uuid.UUID) -> None:
    """Execute bits-per-byte evaluation job on held-out validation text."""
    with Session(get_engine(), expire_on_commit=False) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        payload = job.payload
        eval_id_str = payload.get("bpb_eval_id")
        if not eval_id_str:
            return
        eval_id = uuid.UUID(eval_id_str)
        bpb_eval = session.get(BpbEvaluation, eval_id)
        if not bpb_eval:
            job.status = "failed"
            job.error = "BPB evaluation record not found."
            session.commit()
            return

        dataset_id = bpb_eval.dataset_id
        dataset = session.get(Dataset, dataset_id) if dataset_id else None
        if not dataset:
            job.status = bpb_eval.status = "failed"
            job.error = bpb_eval.error = "Validation dataset snapshot was removed."
            session.commit()
            return

        model_source = bpb_eval.model_source
        stored_name = bpb_eval.model_name
        model_id = bpb_eval.model_id
        checkpoint_id = bpb_eval.checkpoint_id

        val_path, file_hash = _get_validation_file_info(dataset)
        bpb_eval.validation_file_hash = file_hash

        job.status = bpb_eval.status = "running"
        job.pid = os.getpid()
        job.started_at = bpb_eval.started_at = datetime.now(UTC)
        session.commit()

    accumulator = BpbAccumulator()

    try:
        if model_source == "imported":
            imported_name = payload.get("imported_model_name") or stored_name
            if not ImportedModelAdapter.is_available():
                raise ApiError(
                    501,
                    "imported_model_unavailable",
                    "Imported model evaluation depends on SP9.8 (transformers/imported model "
                    "interface), which is not yet available in this build.",
                )
            model, hf_tok = ImportedModelAdapter.load(imported_name)
            tokenizer = ImportedTokenizerAdapter(hf_tok)
            ctx_len = int(payload.get("ctx_len", 512))
            tokenizer_id_str = tokenizer.identity
        else:
            with Session(get_engine()) as session:
                assert model_id is not None
                model, tokenizer, model_record = load_native_model(session, model_id, checkpoint_id)
                ctx_len = model_record.ctx_len
                tokenizer_id_str = tokenizer.identity

        # Score documents
        docs = list(_iter_validation_documents(dataset, val_path))
        total_docs = len(docs)
        if total_docs == 0:
            raise ApiError(
                400, "empty_validation_data", "The snapshot contains no validation documents."
            )

        for idx, doc in enumerate(docs):
            with Session(get_engine()) as session:
                j = session.get(Job, job_id)
                if j and j.cancel_requested:
                    raise Canceled()

            text = doc.get("text", "")
            doc_nll, doc_tokens, doc_bytes = score_text_with_native_model(
                model, tokenizer, text, ctx_len
            )
            accumulator.add(doc_nll, doc_bytes, doc_tokens)

            report(
                job_id,
                idx + 1,
                total_docs,
                f"Scored document {idx + 1}/{total_docs} ({accumulator.total_scored_bytes} bytes)",
            )

        final_bpb = accumulator.bpb
        report(job_id, total_docs, total_docs, f"BPB evaluation complete: {final_bpb:.4f}")

        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            bpb_eval = session.get(BpbEvaluation, eval_id)
            if job and job.cancel_requested:
                raise Canceled()
            if bpb_eval:
                bpb_eval.bpb = final_bpb
                bpb_eval.total_nll = accumulator.total_nll_nats
                bpb_eval.scored_bytes = accumulator.total_scored_bytes
                bpb_eval.scored_tokens = accumulator.total_scored_tokens
                bpb_eval.tokenizer_identity = tokenizer_id_str
                bpb_eval.status = "completed"
                bpb_eval.completed_at = datetime.now(UTC)
                if bpb_eval.run_id:
                    from forge.models import Run

                    run_record = session.get(Run, bpb_eval.run_id)
                    if run_record:
                        cur_metrics = dict(run_record.metrics or {})
                        cur_metrics["bpb"] = final_bpb
                        cur_metrics["bpb_details"] = {
                            "total_nll": accumulator.total_nll_nats,
                            "scored_bytes": accumulator.total_scored_bytes,
                            "scored_tokens": accumulator.total_scored_tokens,
                            "validation_file_hash": bpb_eval.validation_file_hash,
                        }
                        run_record.metrics = cur_metrics

                session.add(
                    EvalResult(
                        id=uuid.uuid4(),
                        result_type="bpb",
                        model_id=bpb_eval.model_id,
                        model_name=bpb_eval.model_name,
                        checkpoint_id=bpb_eval.checkpoint_id,
                        evaluation_id=bpb_eval.id,
                        settings=dict(bpb_eval.settings or {}),
                        code_version=current_commit(),
                        bpb=final_bpb,
                        total_nll=accumulator.total_nll_nats,
                        scored_bytes=accumulator.total_scored_bytes,
                        scored_tokens=accumulator.total_scored_tokens,
                    )
                )
            if job:
                job.status = "completed"
                job.pid = None
                job.completed_at = datetime.now(UTC)
            session.commit()

    except Exception as problem:
        canceled = isinstance(problem, Canceled)
        log.exception("bpb eval job failed job_id=%s", job_id)
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            bpb_eval = session.get(BpbEvaluation, eval_id)
            err_msg = (
                "BPB evaluation was cancelled." if canceled else f"BPB evaluation failed: {problem}"
            )
            if bpb_eval:
                bpb_eval.status = "cancelled" if canceled else "failed"
                bpb_eval.error = err_msg
                bpb_eval.completed_at = datetime.now(UTC)
            if job:
                job.status = "cancelled" if canceled else "failed"
                job.error = err_msg
                job.pid = None
                job.completed_at = datetime.now(UTC)
            session.commit()
