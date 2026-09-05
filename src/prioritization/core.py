"""
Core prioritization layer for risk-based maintenance scheduling.

Computes a deterministic PriorityScore for wind turbine maintenance based on:
    PriorityScore = w1 * FaultRisk + w2 * DegradationRate - w3 * AccessCost + w4 * Criticality

Default weights are provided as initial configurable references and are NOT claimed
to be optimal for any specific operational deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrioritizationWeights:
    """
    Configurable weights for the priority score calculation.

    Formula:
        PriorityScore = w1 * FaultRisk + w2 * DegradationRate - w3 * AccessCost + w4 * Criticality

    Note:
        Default values (all 1.0) serve as initial reference/configurable values
        and are NOT claimed to be optimal for any specific operational context.
    """

    w1_fault_risk: float = 1.0
    w2_degradation_rate: float = 1.0
    w3_access_cost: float = 1.0
    w4_criticality: float = 1.0

    def validate(self) -> None:
        """Validate that all weights are valid, finite, non-negative numbers."""
        weights_dict = {
            "w1_fault_risk": self.w1_fault_risk,
            "w2_degradation_rate": self.w2_degradation_rate,
            "w3_access_cost": self.w3_access_cost,
            "w4_criticality": self.w4_criticality,
        }
        for name, val in weights_dict.items():
            if val is None or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Weight '{name}' must be a numeric value, got {type(val).__name__ if val is not None else 'None'}")
            if np.isnan(val):
                raise ValueError(f"Weight '{name}' cannot be NaN")
            if np.isinf(val):
                raise ValueError(f"Weight '{name}' cannot be infinite")
            if val < 0:
                raise ValueError(f"Weight '{name}' must be non-negative, got {val}")


def _to_numpy_and_validate(val: Any, name: str) -> np.ndarray:
    """Convert input to a numpy array and perform NaN, inf, and type validation."""
    if val is None:
        raise ValueError(f"Required input '{name}' is missing (None)")

    if isinstance(val, (int, float, np.number)):
        arr = np.array(val, dtype=np.float64)
    elif isinstance(val, (pd.Series, pd.DataFrame)):
        if isinstance(val, pd.DataFrame):
            if val.shape[1] == 1:
                arr = val.iloc[:, 0].to_numpy(dtype=np.float64)
            else:
                raise ValueError(f"DataFrame input '{name}' must have exactly 1 column, got shape {val.shape}")
        else:
            arr = val.to_numpy(dtype=np.float64)
    elif isinstance(val, (np.ndarray, list, tuple)):
        try:
            arr = np.asarray(val, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Input '{name}' contains non-numeric data") from exc
    else:
        raise ValueError(f"Unsupported data type for '{name}': {type(val).__name__}")

    if arr.size == 0:
        raise ValueError(f"Input '{name}' cannot be empty")

    if np.isnan(arr).any():
        raise ValueError(f"Input '{name}' contains NaN values")

    if np.isinf(arr).any():
        raise ValueError(f"Input '{name}' contains infinite values")

    return arr


def validate_prioritization_inputs(
    fault_risk: Any,
    degradation_rate: Any,
    access_cost: Any,
    criticality: Any,
    weights: PrioritizationWeights | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PrioritizationWeights]:
    """
    Validate all inputs and return sanitized numpy arrays and validated weights.
    """
    if weights is None:
        weights = PrioritizationWeights()
    weights.validate()

    arr_fr = _to_numpy_and_validate(fault_risk, "fault_risk")
    arr_dr = _to_numpy_and_validate(degradation_rate, "degradation_rate")
    arr_ac = _to_numpy_and_validate(access_cost, "access_cost")
    arr_cr = _to_numpy_and_validate(criticality, "criticality")

    try:
        np.broadcast_shapes(arr_fr.shape, arr_dr.shape, arr_ac.shape, arr_cr.shape)
    except ValueError as exc:
        raise ValueError(
            f"Incompatible input shapes: fault_risk={arr_fr.shape}, "
            f"degradation_rate={arr_dr.shape}, access_cost={arr_ac.shape}, "
            f"criticality={arr_cr.shape}"
        ) from exc

    return arr_fr, arr_dr, arr_ac, arr_cr, weights


def calculate_priority_score(
    fault_risk: Any,
    degradation_rate: Any,
    access_cost: Any,
    criticality: Any,
    weights: PrioritizationWeights | None = None,
    *,
    w1: float | None = None,
    w2: float | None = None,
    w3: float | None = None,
    w4: float | None = None,
) -> Any:
    """
    Calculate PriorityScore given components and weights.

    Formula:
        PriorityScore = w1 * FaultRisk + w2 * DegradationRate - w3 * AccessCost + w4 * Criticality

    Supports scalar numbers, numpy arrays, and pandas Series/DataFrames.
    Preserves input container type (e.g. returns pd.Series if fault_risk is a pd.Series).
    """
    if any(w is not None for w in (w1, w2, w3, w4)):
        base = weights or PrioritizationWeights()
        weights = PrioritizationWeights(
            w1_fault_risk=w1 if w1 is not None else base.w1_fault_risk,
            w2_degradation_rate=w2 if w2 is not None else base.w2_degradation_rate,
            w3_access_cost=w3 if w3 is not None else base.w3_access_cost,
            w4_criticality=w4 if w4 is not None else base.w4_criticality,
        )

    arr_fr, arr_dr, arr_ac, arr_cr, weights = validate_prioritization_inputs(
        fault_risk, degradation_rate, access_cost, criticality, weights
    )

    score_arr = (
        weights.w1_fault_risk * arr_fr
        + weights.w2_degradation_rate * arr_dr
        - weights.w3_access_cost * arr_ac
        + weights.w4_criticality * arr_cr
    )

    if isinstance(fault_risk, pd.Series):
        return pd.Series(score_arr, index=fault_risk.index, name="priority_score")
    if isinstance(fault_risk, (int, float, np.number)) and score_arr.ndim == 0:
        return float(score_arr)
    if np.ndim(fault_risk) == 0 and score_arr.ndim == 0:
        return float(score_arr)

    return score_arr


class PriorityCalculator:
    """
    Reusable priority score calculator holding pre-configured weights.
    """

    def __init__(self, weights: PrioritizationWeights | None = None) -> None:
        self.weights = weights or PrioritizationWeights()
        self.weights.validate()

    def compute(
        self,
        fault_risk: Any,
        degradation_rate: Any,
        access_cost: Any,
        criticality: Any,
    ) -> Any:
        """Compute priority score using configured weights."""
        return calculate_priority_score(
            fault_risk=fault_risk,
            degradation_rate=degradation_rate,
            access_cost=access_cost,
            criticality=criticality,
            weights=self.weights,
        )

    def compute_dataframe(
        self,
        df: pd.DataFrame,
        fault_risk_col: str = "fault_risk",
        degradation_rate_col: str = "degradation_rate",
        access_cost_col: str = "access_cost",
        criticality_col: str = "criticality",
        output_col: str = "priority_score",
    ) -> pd.DataFrame:
        """
        Compute priority scores for a pandas DataFrame and return a copy with the result column.
        """
        missing = [
            c
            for c in (fault_risk_col, degradation_rate_col, access_cost_col, criticality_col)
            if c not in df.columns
        ]
        if missing:
            raise ValueError(f"DataFrame is missing required prioritization columns: {missing}")

        out = df.copy()
        out[output_col] = self.compute(
            fault_risk=out[fault_risk_col],
            degradation_rate=out[degradation_rate_col],
            access_cost=out[access_cost_col],
            criticality=out[criticality_col],
        )
        return out
