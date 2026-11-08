"""Training worker integrating MetricBuffer for step metrics."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from forge.db import open_session
from forge.gpu import free_gpu_memory
from forge.models import Run
from forge.train.metrics import MetricBuffer

if TYPE_CHECKING:
    pass


class TrainingWorker:
    """Executes training steps and logs measurements via MetricBuffer."""

    def __init__(
        self,
        session_factory: Callable[[], Session] = open_session,
        metric_batch_size: int = 20,
    ) -> None:
        self.session_factory = session_factory
        self.metric_batch_size = metric_batch_size

    def step(
        self,
        buffer: MetricBuffer,
        step: int,
        loss: float,
        val_loss: float | None = None,
        learning_rate: float | None = None,
        speed: float | None = None,
    ) -> None:
        """Record a single step measurement into the metric buffer."""
        buffer.record(
            step=step,
            loss=loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            speed=speed,
        )

    def run_training_loop(
        self,
        run_id: uuid.UUID,
        steps: int = 50,
        eval_every: int = 10,
        fail_at_step: int | None = None,
        stop_at_step: int | None = None,
    ) -> Run:
        """Execute training steps for a run, buffering metrics and persisting in batches.

        Flushes on normal completion, stopped paths, and failure paths.
        """
        session = self.session_factory()
        try:
            run = session.get(Run, run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found.")
            run.status = "running"
            run.started_at = datetime.now(UTC)
            session.commit()
        finally:
            session.close()

        try:
            with MetricBuffer(
                self.session_factory, run_id, batch_size=self.metric_batch_size
            ) as buffer:
                for s in range(1, steps + 1):
                    if stop_at_step is not None and s > stop_at_step:
                        # Simulated stop / cancellation path
                        session = self.session_factory()
                        try:
                            r = session.get(Run, run_id)
                            if not r:
                                raise ValueError(f"Run {run_id} not found.")
                            r.status = "stopped"
                            session.commit()
                            return r  # Exiting with-block flushes remaining buffered metrics!
                        finally:
                            session.close()

                    if fail_at_step is not None and s >= fail_at_step:
                        # Simulated failure path
                        raise RuntimeError(f"Training encountered a NaN loss or crash at step {s}")

                    # Simulated step metrics
                    loss_val = round(3.5 * math.exp(-s / 40.0) + 0.5, 4)
                    val_val = round(loss_val + 0.1, 4) if (s % eval_every == 0) else None
                    lr_val = round(6e-4 * (1.0 - s / (steps * 1.5)), 6)
                    speed_val = 14200.0

                    self.step(
                        buffer,
                        step=s,
                        loss=loss_val,
                        val_loss=val_val,
                        learning_rate=lr_val,
                        speed=speed_val,
                    )

                    # Update run summary periodically
                    if s % self.metric_batch_size == 0 or s == steps:
                        session = self.session_factory()
                        try:
                            r = session.get(Run, run_id)
                            if r:
                                r.current_step = s
                                r.loss = loss_val
                                if val_val is not None:
                                    r.eval_loss = val_val
                                r.tokens_per_sec = speed_val
                                session.commit()
                        finally:
                            session.close()

            # Upon normal exit from with-block, all metrics are flushed!
            session = self.session_factory()
            try:
                r = session.get(Run, run_id)
                if not r:
                    raise ValueError(f"Run {run_id} not found.")
                r.status = "completed"
                r.completed_at = datetime.now(UTC)
                session.commit()
                return r
            finally:
                session.close()

        except Exception as err:
            session = self.session_factory()
            try:
                r = session.get(Run, run_id)
                if r and r.status != "stopped":
                    r.status = "failed"
                    r.error = str(err)
                    r.completed_at = datetime.now(UTC)
                    session.commit()
            finally:
                session.close()
            raise


_original_run_training_loop = TrainingWorker.run_training_loop


def _run_training_loop_with_gpu_cleanup(self, *args, **kwargs):
    try:
        return _original_run_training_loop(self, *args, **kwargs)
    finally:
        free_gpu_memory()


TrainingWorker.run_training_loop = _run_training_loop_with_gpu_cleanup
