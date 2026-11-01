"""Tests for SP9.3 speed measurement utilities."""

import pytest

from forge.train.speed_test import (
    calculate_steps_per_second,
    measure_execution_time,
)


def test_measure_execution_time():
    """Check that execution time and results are returned."""
    result, elapsed = measure_execution_time(lambda: 42)

    assert result == 42
    assert elapsed >= 0


def test_calculate_steps_per_second():
    """Check that training throughput is calculated correctly."""
    speed = calculate_steps_per_second(100, 5.0)

    assert speed == 20.0


def test_zero_steps():
    """Zero training steps should produce zero throughput."""
    speed = calculate_steps_per_second(0, 5.0)

    assert speed == 0.0


def test_negative_steps():
    """Negative step counts should be rejected."""
    with pytest.raises(ValueError):
        calculate_steps_per_second(-10, 5.0)


def test_zero_elapsed_time():
    """Elapsed time must be greater than zero."""
    with pytest.raises(ValueError):
        calculate_steps_per_second(100, 0.0)


def test_negative_elapsed_time():
    """Negative elapsed time should be rejected."""
    with pytest.raises(ValueError):
        calculate_steps_per_second(100, -2.0)


def test_measure_execution_time_propagates_error():
    """Errors from the measured function must not be hidden."""

    def failing_function():
        raise RuntimeError("Training failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        measure_execution_time(failing_function)
