"""
Criticality scoring model for wind-turbine maintenance prioritization.

Produces Criticality, the ``w4`` term consumed by the existing PriorityScore
formula in ``core.py``:

    PriorityScore = w1*FaultRisk + w2*DegradationRate - w3*AccessCost + w4*Criticality

WHAT THIS VALUE REPRESENTS:
Criticality is an interpretable OPERATIONAL-IMPORTANCE INDEX, not a direct
monetary value. It reflects how operationally significant a turbine is,
based on how much generation capacity/output it represents - independent of
its technical fault condition. A turbine with a higher rated capacity can
represent a greater potential generation impact if it fails; a turbine with
higher actual/current output can represent a greater current operational
contribution. Capacity is relatively stable over time, while output is
time-dependent and can be affected by weather, curtailment, maintenance
status, or grid conditions - so output should not automatically be treated
as a permanent measure of a turbine's importance.

WHY THIS IS SEPARATE FROM FaultRisk/DegradationRate:
FaultRisk and DegradationRate (see degradation.py and aggregation.py) already
describe a turbine's technical condition and fault trend from model
predictions. Criticality intentionally does NOT reuse fault probability or
any prediction signal - mixing them in would double-count technical risk
inside what is meant to be an independent "how much does this turbine
matter operationally" signal.

MODEL:
    capacity_factor = capacity / reference_capacity
    output_factor   = output   / reference_output
    Criticality = weight_capacity * capacity_factor + weight_output * output_factor

Terms for inputs that are not supplied are simply omitted (treated as zero
contribution), so the same formula covers capacity-only, output-only, and
combined calculation. ``reference_capacity``/``reference_output`` are
normalization parameters (not empirical universal constants) that make the
resulting factors unitless and comparable; ``weight_capacity``/
``weight_output`` control how much each factor contributes to the combined
score. All numeric defaults are illustrative model parameters chosen for
interpretability and testability, not measured industry values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CriticalityConfig:
    """
    Configurable parameters of the Criticality model.

    Formula:
        capacity_factor = capacity / reference_capacity
        output_factor   = output   / reference_output
        Criticality = weight_capacity * capacity_factor + weight_output * output_factor

    Note:
        ``reference_capacity`` and ``reference_output`` are normalization
        parameters, not empirical universal constants. Default values are
        illustrative model parameters for interpretability and testing.
    """

    reference_capacity: float = 1.0
    reference_output: float = 1.0
    weight_capacity: float = 1.0
    weight_output: float = 1.0

    def validate(self) -> None:
        """Validate references (finite, positive) and weights (finite, non-negative)."""
        _validate_positive(self.reference_capacity, "reference_capacity")
        _validate_positive(self.reference_output, "reference_output")
        _validate_non_negative(self.weight_capacity, "weight_capacity")
        _validate_non_negative(self.weight_output, "weight_output")
        if self.weight_capacity == 0.0 and self.weight_output == 0.0:
            raise ValueError("weight_capacity and weight_output cannot both be zero")


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


def _validate_measurement(value: Any, name: str) -> tuple[np.ndarray, bool]:
    """Return (validated non-negative float array, was_scalar_input) for capacity/output."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        array = np.array([float(value)])
        was_scalar = True
    else:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        was_scalar = False

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values")
    if (array < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    return array, was_scalar


def calculate_criticality(
    capacity: Any | None = None,
    output: Any | None = None,
    *,
    config: CriticalityConfig | None = None,
) -> Any:
    """
    Compute a deterministic Criticality index for one or more turbines.

    At least one of ``capacity`` or ``output`` must be supplied. Each may be
    a scalar number (one turbine) or a 1D array-like of numbers (vectorized,
    per-turbine calculation); if both are supplied as arrays their shapes
    must be broadcastable. Returns a Python float for scalar input, or a
    ``numpy.ndarray`` aligned with the input order for vectorized input -
    consistent with the scalar-in/scalar-out convention used elsewhere in
    this package (see ``core.calculate_priority_score`` and
    ``access_cost.calculate_access_cost``).
    """
    if capacity is None and output is None:
        raise ValueError("at least one of capacity or output must be provided")

    config = config or CriticalityConfig()
    config.validate()

    capacity_values: np.ndarray | None = None
    output_values: np.ndarray | None = None
    capacity_scalar = output_scalar = True

    if capacity is not None:
        capacity_values, capacity_scalar = _validate_measurement(capacity, "capacity")
    if output is not None:
        output_values, output_scalar = _validate_measurement(output, "output")

    if capacity_values is not None and output_values is not None:
        try:
            np.broadcast_shapes(capacity_values.shape, output_values.shape)
        except ValueError as exc:
            raise ValueError(
                f"Incompatible input shapes: capacity={capacity_values.shape}, "
                f"output={output_values.shape}"
            ) from exc
        was_scalar = capacity_scalar and output_scalar
    else:
        was_scalar = capacity_scalar if capacity_values is not None else output_scalar

    capacity_term = (
        config.weight_capacity * (capacity_values / config.reference_capacity)
        if capacity_values is not None
        else 0.0
    )
    output_term = (
        config.weight_output * (output_values / config.reference_output)
        if output_values is not None
        else 0.0
    )
    criticality = capacity_term + output_term

    if was_scalar:
        return float(np.asarray(criticality).reshape(-1)[0])
    return np.asarray(criticality, dtype=np.float64)


class CriticalityCalculator:
    """Reusable Criticality calculator holding a pre-configured CriticalityConfig."""

    def __init__(self, config: CriticalityConfig | None = None) -> None:
        self.config = config or CriticalityConfig()
        self.config.validate()

    def compute(self, capacity: Any | None = None, output: Any | None = None) -> Any:
        """Compute Criticality for one turbine (scalar) or many (array-like)."""
        return calculate_criticality(capacity=capacity, output=output, config=self.config)
