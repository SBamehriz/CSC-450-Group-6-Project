"""Training metrics buffer and persistence.

Records loss, validation loss, learning rate, and speed per training step.
Measurements are buffered in memory and persisted in batches to avoid committing
a separate database transaction on every step. Remaining measurements are flushed
on completion, stop, or failure paths.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge.models import Checkpoint, MetricPoint, Run

if TYPE_CHECKING:
    pass


class MetricBuffer:
    """Buffers step measurements in memory and commits them in batches.

    Uncommitted measurements remain in memory in self._buffer until either:
      1. The buffer reaches batch_size and triggers an automatic flush, or
      2. flush() is explicitly called, or
      3. The context manager exits on completion, stop, or failure.

    If training is forcibly killed before a flush, uncommitted points in the buffer
    are lost, while all previously committed points remain durable in SQLite.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        run_id: uuid.UUID,
        batch_size: int = 20,
    ) -> None:
        self.session_factory = session_factory
        self.run_id = run_id
        self.batch_size = max(1, batch_size)
        self._buffer: list[dict] = []

    @property
    def pending_count(self) -> int:
        """Number of measurements currently buffered and not yet committed."""
        return len(self._buffer)

    def record(
        self,
        step: int,
        loss: float | None = None,
        val_loss: float | None = None,
        learning_rate: float | None = None,
        speed: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        """Add a step measurement to the buffer, flushing if batch_size is reached."""
        effective_speed = speed if speed is not None else tokens_per_sec
        self._buffer.append(
            {
                "step": step,
                "loss": loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "speed": effective_speed,
                "created_at": datetime.now(UTC),
            }
        )
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Persist all buffered measurements in a single database transaction.

        Handles retries gracefully: if a measurement for (run_id, step) already exists,
        its values are updated rather than raising a unique constraint violation.
        """
        if not self._buffer:
            return 0

        to_commit = list(self._buffer)
        self._buffer.clear()

        steps = [point["step"] for point in to_commit]
        count = len(to_commit)

        session = self.session_factory()
        try:
            # Query existing points for these steps to handle retries without duplicate errors
            existing_records = (
                session.execute(
                    select(MetricPoint).where(
                        MetricPoint.run_id == self.run_id,
                        MetricPoint.step.in_(steps),
                    )
                )
                .scalars()
                .all()
            )
            existing_by_step = {record.step: record for record in existing_records}

            for item in to_commit:
                step = item["step"]
                if step in existing_by_step:
                    # Update existing record on retry
                    point = existing_by_step[step]
                    point.loss = item["loss"]
                    point.val_loss = item["val_loss"]
                    point.learning_rate = item["learning_rate"]
                    point.speed = item["speed"]
                    point.created_at = item["created_at"]
                else:
                    new_point = MetricPoint(
                        id=uuid.uuid4(),
                        run_id=self.run_id,
                        step=step,
                        loss=item["loss"],
                        val_loss=item["val_loss"],
                        learning_rate=item["learning_rate"],
                        speed=item["speed"],
                        created_at=item["created_at"],
                    )
                    session.add(new_point)

            session.commit()
            return count
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __enter__(self) -> MetricBuffer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Flush remaining buffered measurements on completion, stop, or failure."""
        try:
            self.flush()
        except Exception:
            if exc_type is None:
                raise


def get_metric_points(
    session: Session,
    run_id: uuid.UUID,
    limit: int | None = None,
    offset: int = 0,
) -> list[MetricPoint]:
    """Retrieve metric points for a training run, ordered chronologically by step."""
    query = (
        select(MetricPoint)
        .where(MetricPoint.run_id == run_id)
        .order_by(MetricPoint.step.asc())
        .offset(offset)
    )
    if limit is not None:
        query = query.limit(limit)
    return list(session.execute(query).scalars().all())


def get_best_validation_loss(session: Session, run_id: uuid.UUID) -> float | None:
    """Find the lowest validation loss recorded for a run across metrics and checkpoints."""
    metric_best = session.execute(
        select(func.min(MetricPoint.val_loss)).where(
            MetricPoint.run_id == run_id,
            MetricPoint.val_loss.is_not(None),
        )
    ).scalar_one_or_none()

    checkpoint_best = session.execute(
        select(func.min(Checkpoint.val_loss)).where(
            Checkpoint.run_id == run_id,
            Checkpoint.val_loss.is_not(None),
        )
    ).scalar_one_or_none()

    run = session.execute(select(Run).where(Run.id == run_id)).scalar_one_or_none()
    run_eval = run.eval_loss if run else None

    candidates = [v for v in (metric_best, checkpoint_best, run_eval) if v is not None]
    return min(candidates) if candidates else None
