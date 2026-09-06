"""
AccessCost model for wind-turbine maintenance prioritization.

Produces AccessCost, the ``w3`` term consumed by the existing PriorityScore
formula in ``core.py``:

    PriorityScore = w1*FaultRisk + w2*DegradationRate - w3*AccessCost + w4*Criticality

IMPORTANT - what this value represents:
AccessCost here is an interpretable ACCESS BURDEN / RELATIVE ACCESS-COST INDEX,
not an estimate of real monetary maintenance expense. It combines configurable
cost *components* (base access, transport/travel, logistics) with an offshore
premium and an optional multiplier into one deterministic number that is larger
for turbines that are harder/costlier to reach. All numeric defaults below are
illustrative model parameters chosen for interpretability and testability -
they are NOT empirical offshore/onshore cost ratios. Grounding this model in
real cost evidence is explicitly out of scope for this module (see project
task notes on AccessCost research).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


VALID_FARM_TYPES = ("onshore", "offshore")


@dataclass(frozen=True)
class AccessCostConfig:
    """
    Configurable components of the AccessCost model.

    Formula:
        raw = base_access_cost + transport_cost + logistics_cost
        raw *= offshore_premium   (only applied when farm_type == "offshore")
        AccessCost = raw * multiplier

    Note:
        Default values are illustrative model parameters for interpretability
        and testing. They are NOT claimed to reflect real onshore/offshore
        maintenance economics.
    """

    base_access_cost: float = 1.0
    transport_cost: float = 0.0
    logistics_cost: float = 0.0
    offshore_premium: float = 1.5
    multiplier: float = 1.0

    def validate(self) -> None:
        """Validate that all cost components and factors are sane numbers."""
        _validate_non_negative(self.base_access_cost, "base_access_cost")
        _validate_non_negative(self.transport_cost, "transport_cost")
        _validate_non_negative(self.logistics_cost, "logistics_cost")
        _validate_positive(self.offshore_premium, "offshore_premium")
        _validate_positive(self.multiplier, "multiplier")


def _validate_non_negative(value: Any, name: str) -> float:
    """Return a finite non-negative float, rejecting bool/NaN/inf/non-numeric input."""
    if not isinstance(value, (int, float, np.number)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value}")
    return float(value)


def _validate_positive(value: Any, name: str) -> float:
    """Return a finite positive float, rejecting bool/NaN/inf/non-numeric input."""
    if not isinstance(value, (int, float, np.number)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value}")
    return float(value)


def _normalize_farm_type(farm_type: Any) -> tuple[np.ndarray, bool]:
    """Return (normalized lowercase farm-type array, was_scalar_input)."""
    if farm_type is None:
        raise ValueError("farm_type is required")

    if isinstance(farm_type, str):
        raw_values = np.array([farm_type])
        was_scalar = True
    else:
        try:
            raw_values = np.asarray(farm_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("farm_type must be a string or an array-like of strings") from exc
        if raw_values.ndim != 1:
            raise ValueError(f"farm_type must be one-dimensional, got shape {raw_values.shape}")
        if raw_values.size == 0:
            raise ValueError("farm_type must not be empty")
        was_scalar = False

    normalized = np.array([str(value).strip().lower() for value in raw_values])
    invalid = sorted(set(normalized) - set(VALID_FARM_TYPES))
    if invalid:
        raise ValueError(
            f"farm_type must be one of {VALID_FARM_TYPES}, got invalid value(s): {invalid}"
        )
    return normalized, was_scalar


def calculate_access_cost(farm_type: Any, config: AccessCostConfig | None = None) -> Any:
    """
    Compute a deterministic AccessCost index for one or more turbines.

    ``farm_type`` is ``"onshore"`` or ``"offshore"`` (case-insensitive), or a
    1D array-like of such strings for a vectorized, per-turbine calculation.
    Returns a Python float for scalar input, or a ``numpy.ndarray`` aligned
    with the input order for vectorized input - consistent with the scalar-
    in/scalar-out convention used by ``core.calculate_priority_score``.
    """
    config = config or AccessCostConfig()
    config.validate()

    farm_types, was_scalar = _normalize_farm_type(farm_type)
    is_offshore = farm_types == "offshore"

    raw_cost = config.base_access_cost + config.transport_cost + config.logistics_cost
    premium = np.where(is_offshore, config.offshore_premium, 1.0)
    access_cost = raw_cost * premium * config.multiplier

    if was_scalar:
        return float(access_cost[0])
    return access_cost


class AccessCostCalculator:
    """Reusable AccessCost calculator holding a pre-configured AccessCostConfig."""

    def __init__(self, config: AccessCostConfig | None = None) -> None:
        self.config = config or AccessCostConfig()
        self.config.validate()

    def compute(self, farm_type: Any) -> Any:
        """Compute AccessCost for one turbine (str) or many (array-like of str)."""
        return calculate_access_cost(farm_type, config=self.config)
