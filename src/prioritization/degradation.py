"""Deterministic degradation-rate calculation from fault probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TIMESTEP_HOURS = 10.0 / 60.0


def _probability_array(probabilities: Any) -> np.ndarray:
    """Return validated one-dimensional fault probabilities."""
    if probabilities is None:
        raise ValueError("probabilities is required")
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain numeric values") from exc

    if values.ndim != 1:
        raise ValueError(f"probabilities must be one-dimensional, got shape {values.shape}")
    if len(values) < 2:
        raise ValueError("at least two probabilities are required to calculate a slope")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must not contain NaN or infinite values")
    if not np.logical_and(values >= 0.0, values <= 1.0).all():
        raise ValueError("probabilities must be within [0, 1]")
    return values


def _time_hours(time_points: Any, count: int, timestep_hours: float) -> np.ndarray:
    """Return strictly increasing elapsed hours from numeric or datetime points."""
    if time_points is None:
        if not isinstance(timestep_hours, (int, float, np.number)) or isinstance(
            timestep_hours, (bool, np.bool_)
        ):
            raise ValueError("timestep_hours must be a finite positive number")
        if not np.isfinite(timestep_hours) or timestep_hours <= 0:
            raise ValueError("timestep_hours must be a finite positive number")
        return np.arange(count, dtype=np.float64) * float(timestep_hours)

    try:
        raw_points = np.asarray(time_points)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_points must be a one-dimensional sequence") from exc

    if raw_points.ndim != 1:
        raise ValueError(f"time_points must be one-dimensional, got shape {raw_points.shape}")
    if len(raw_points) != count:
        raise ValueError(
            f"time_points length ({len(raw_points)}) does not match probabilities length ({count})"
        )

    if np.issubdtype(raw_points.dtype, np.number):
        try:
            hours = raw_points.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric time_points must be finite") from exc
        if not np.isfinite(hours).all():
            raise ValueError("numeric time_points must not contain NaN or infinite values")
        return hours - hours[0]

    try:
        timestamps = pd.to_datetime(time_points, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("time_points must contain valid numeric or datetime values") from exc
    if pd.isna(timestamps).any():
        raise ValueError("time_points must not contain missing timestamps")
    elapsed = (timestamps - timestamps[0]).total_seconds() / 3600.0
    return np.asarray(elapsed, dtype=np.float64)


def calculate_degradation_rate(
    probabilities: Any,
    time_points: Any | None = None,
    *,
    timestep_hours: float = DEFAULT_TIMESTEP_HOURS,
) -> float:
    """
    Estimate the linear probability trend, expressed as probability increase per hour.

    The result is the ordinary least-squares slope of ``probabilities`` against
    elapsed hours. Pass ordered ``window_end`` timestamps for prediction tables.
    If timestamps are unavailable, values are assumed to be equally spaced by
    ``timestep_hours``, which defaults to the project's 10-minute GRU cadence.
    Callers working with hourly baseline windows should pass their ``window_end``
    values or explicitly set ``timestep_hours=1.0``.

    The caller selects the trailing 6-12 hour probability window; this function
    deliberately does not choose or truncate that operational window.
    """
    values = _probability_array(probabilities)
    hours = _time_hours(time_points, len(values), timestep_hours)
    intervals = np.diff(hours)
    if not np.isfinite(hours).all() or not (intervals > 0.0).all():
        raise ValueError("time_points must be finite and strictly increasing")

    centered_hours = hours - hours.mean()
    centered_probabilities = values - values.mean()
    return float(np.dot(centered_hours, centered_probabilities) / np.dot(centered_hours, centered_hours))