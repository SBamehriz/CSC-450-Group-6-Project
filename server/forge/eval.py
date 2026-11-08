"""Evaluation job execution and generation logic."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import Canceled, report
from forge.models import EvalResult, Evaluation, Job
from forge.version import current_commit

log = logging.getLogger(__name__)


def execute_eval_job(job_id: uuid.UUID) -> None:
    """Execute evaluation job from the captured prompts copy in job payload."""
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        payload = job.payload
        eval_id_str = payload.get("evaluation_id")
        if not eval_id_str:
            return
        eval_id = uuid.UUID(eval_id_str)
        evaluation = session.get(Evaluation, eval_id)
        if not evaluation:
            job.status = "failed"
            job.error = "Evaluation record not found."
            session.commit()
            return

        model_id_str = payload.get("model_id")
        model_id = uuid.UUID(model_id_str) if model_id_str else None

        # Execute queued jobs and retries from the CAPTURED copy (Requirement 6)
        prompts = [str(p) for p in payload.get("prompts", [])]
        max_new_tokens = int(payload.get("max_new_tokens", 32))
        temperature = float(payload.get("temperature", 0.7))

        job.status = "running"
        job.pid = os.getpid()
        job.started_at = datetime.now(UTC)
        evaluation.status = "running"
        evaluation.started_at = datetime.now(UTC)
        session.commit()

    results: list[dict] = []
    total = len(prompts)
    try:
        if model_id is None:
            raise ValueError("This evaluation has no model.")
        from forge.measuring import load
        from forge.prompts import answer

        # the model's real trained weights, through the same path as Compare
        loaded = load(model_id, None, "auto")
        settings = {"temperature": temperature, "top_k": 40, "max_new_tokens": max_new_tokens}
        for idx, prompt in enumerate(prompts):
            with Session(get_engine()) as session:
                j = session.get(Job, job_id)
                if j and j.cancel_requested:
                    raise Canceled()

            report(job_id, idx, total, f"Evaluating prompt {idx + 1}/{total}")
            output_text = answer(loaded, prompt, settings, idx)
            results.append(
                {
                    "prompt": prompt,
                    "output": output_text,
                    "tokens_generated": len(output_text.split()),
                }
            )

        # report on its own connection before this session takes the write lock
        report(job_id, total, total, "Evaluation complete")
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            evaluation = session.get(Evaluation, eval_id)
            if job and job.cancel_requested:
                raise Canceled()
            if evaluation:
                evaluation.results = results
                evaluation.status = "completed"
                evaluation.completed_at = datetime.now(UTC)
                code_ver = current_commit()
                for entry in results:
                    session.add(
                        EvalResult(
                            id=uuid.uuid4(),
                            result_type="prompt",
                            model_id=model_id,
                            model_name=loaded.name,
                            checkpoint_id=loaded.checkpoint_id,
                            evaluation_id=eval_id,
                            settings=settings,
                            code_version=code_ver,
                            prompt=entry.get("prompt"),
                            answer=entry.get("output"),
                        )
                    )
            if job:
                job.status = "completed"
                job.pid = None
                job.completed_at = datetime.now(UTC)
            session.commit()

    except Exception as problem:
        canceled = isinstance(problem, Canceled)
        log.exception("eval job failed job_id=%s", job_id)
        with Session(get_engine()) as session:
            job = session.get(Job, job_id)
            evaluation = session.get(Evaluation, eval_id)
            err_msg = "Evaluation was cancelled." if canceled else f"Evaluation failed: {problem}"
            if evaluation:
                evaluation.status = "cancelled" if canceled else "failed"
                evaluation.error = err_msg
                evaluation.completed_at = datetime.now(UTC)
            if job:
                job.status = "cancelled" if canceled else "failed"
                job.error = err_msg
                job.pid = None
                job.completed_at = datetime.now(UTC)
            session.commit()
