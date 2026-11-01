"""Utilities for measuring training execution speed."""

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


def measure_execution_time(function: Callable[[], T]) -> tuple[T, float]:
    """Execute a function and measure how long it takes."""
    start_time = perf_counter()

    try:
        return function(), perf_counter() - start_time
    except Exception:
        raise


def calculate_steps_per_second(steps: int, elapsed_seconds: float) -> float:
    """Calculate training throughput in steps per second."""
    if steps < 0:
        raise ValueError("Steps cannot be negative.")

    if elapsed_seconds <= 0:
        raise ValueError("Elapsed time must be greater than zero.")

    return steps / elapsed_seconds
