"""Trailing prediction smoothing utilities producing the prioritization FaultRisk signal."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_TIMESTEP_HOURS = 10.0 / 60.0
DEFAULT_HORIZON_HOURS = 6.0


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    """Return validated one-dimensional fault probabilities."""
    if probabilities is None:
        raise ValueError("probabilities is required")
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain numeric values") from exc

    if values.ndim != 1:
        raise ValueError(f"probabilities must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("probabilities must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must not contain NaN or infinite values")
    if not np.logical_and(values >= 0.0, values <= 1.0).all():
        raise ValueError("probabilities must be within [0, 1]")
    return values


def _validate_positive(value: Any, name: str) -> float:
    """Return a finite positive float, rejecting bool/NaN/inf/non-numeric input."""
    if not isinstance(value, (int, float, np.number)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def aggregate_predictions(
    probabilities: Any,
    *,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    timestep_hours: float = DEFAULT_TIMESTEP_HOURS,
) -> np.ndarray:
    """
    Smooth a fault-probability sequence into a trailing FaultRisk signal.

    Computes, for each index ``i``, the mean of ``probabilities[max(0, i - w + 1) : i + 1]``
    where ``w = round(horizon_hours / timestep_hours)`` is the number of samples spanned by
    the requested horizon (minimum 1). Only current and past samples are ever used, so no
    future/test information leaks into the aggregated value - this makes the result safe to
    align back onto the original ``window_end``/``timestep_idx`` ordering.

    Near the start of the sequence, fewer than ``w`` samples are available; the window
    expands from the first sample instead of assuming missing history (no zero-padding), so
    early outputs are the average of whatever real history exists so far. If the horizon
    resolves to fewer than two samples (``w <= 1``), smoothing has no effect and the input is
    returned unchanged (copied).

    ``timestep_hours`` defaults to this module's ``DEFAULT_TIMESTEP_HOURS``,
    the project's 10-minute GRU cadence. Callers working with hourly baseline
    windows should pass ``timestep_hours=1.0``.
    """
    values = _validate_probabilities(probabilities)
    horizon_hours = _validate_positive(horizon_hours, "horizon_hours")
    timestep_hours = _validate_positive(timestep_hours, "timestep_hours")

    window = max(int(round(horizon_hours / timestep_hours)), 1)
    if window == 1:
        return values.copy()

    n = values.size
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    indices = np.arange(n)
    window_start = np.maximum(0, indices - window + 1)
    counts = indices - window_start + 1
    totals = cumulative[indices + 1] - cumulative[window_start]
    return totals / counts
