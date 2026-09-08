"""
End-to-end prioritization pipeline orchestration.

Combines the existing, independently-tested prioritization components into the
project's PriorityScore formula:

    raw model predictions
            v
    aggregate_predictions (aggregation.py)   -> smoothed FaultRisk signal
            v
    calculate_degradation_rate (degradation.py) -> DegradationRate
            v
    calculate_priority_score (core.py)       -> PriorityScore (+ AccessCost, Criticality)

This module does not reimplement smoothing, slope-fitting, or scoring: it only
wires the existing functions together. See core.py, aggregation.py, and
degradation.py for the respective algorithms and validation rules.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.prioritization.aggregation import (
    DEFAULT_HORIZON_HOURS,
    DEFAULT_TIMESTEP_HOURS,
    aggregate_predictions,
)
from src.prioritization.core import PrioritizationWeights, calculate_priority_score
from src.prioritization.degradation import calculate_degradation_rate


def calculate_priority(
    fault_risk: Any,
    degradation_rate: Any,
    access_cost: Any,
    criticality: Any,
    *,
    weights: PrioritizationWeights | None = None,
) -> Any:
    """
    Combine already-prepared components into a PriorityScore.

    Thin orchestration wrapper: delegates entirely to
    ``core.calculate_priority_score`` and adds no scoring logic of its own.
    AccessCost and Criticality are treated as already-prepared numeric inputs;
    this pipeline does not invent or ground their values.
    """
    return calculate_priority_score(
        fault_risk=fault_risk,
        degradation_rate=degradation_rate,
        access_cost=access_cost,
        criticality=criticality,
        weights=weights,
    )


class PrioritizationPipeline:
    """
    Stateful orchestrator wiring aggregation, degradation, and scoring together.

    Holds the weights and the aggregation horizon/timestep so FaultRisk and
    DegradationRate are derived from raw prediction sequences using the same
    time convention, then combined via the existing core scoring formula.
    """

    def __init__(
        self,
        weights: PrioritizationWeights | None = None,
        *,
        horizon_hours: float = DEFAULT_HORIZON_HOURS,
        timestep_hours: float = DEFAULT_TIMESTEP_HOURS,
    ) -> None:
        self.weights = weights or PrioritizationWeights()
        self.weights.validate()
        self.horizon_hours = horizon_hours
        self.timestep_hours = timestep_hours

    def fault_risk_from_predictions(self, probabilities: Any) -> float:
        """
        Derive a scalar FaultRisk as the most recent smoothed prediction.

        Delegates smoothing entirely to ``aggregation.aggregate_predictions``,
        which only ever uses current and past predictions, then reads the
        latest (rightmost) value of the trailing signal - never a future one.
        """
        signal = aggregate_predictions(
            probabilities, horizon_hours=self.horizon_hours, timestep_hours=self.timestep_hours
        )
        return float(signal[-1])

    def degradation_rate_from_predictions(self, probabilities: Any) -> float:
        """
        Derive DegradationRate as the slope of the smoothed FaultRisk signal.

        The same ``probabilities`` are first smoothed with
        ``aggregate_predictions`` using this pipeline's horizon/timestep, then
        ``degradation.calculate_degradation_rate`` fits the trend to that
        signal using the identical ``timestep_hours`` convention.
        """
        signal = aggregate_predictions(
            probabilities, horizon_hours=self.horizon_hours, timestep_hours=self.timestep_hours
        )
        return calculate_degradation_rate(signal, timestep_hours=self.timestep_hours)

    def calculate(
        self,
        fault_risk: Any,
        degradation_rate: Any,
        access_cost: Any,
        criticality: Any,
    ) -> Any:
        """Combine already-prepared components into a PriorityScore."""
        return calculate_priority_score(
            fault_risk=fault_risk,
            degradation_rate=degradation_rate,
            access_cost=access_cost,
            criticality=criticality,
            weights=self.weights,
        )

    def calculate_from_predictions(
        self,
        probabilities: Any,
        access_cost: Any,
        criticality: Any,
    ) -> Any:
        """Run the full raw-predictions-to-PriorityScore flow for one asset."""
        fault_risk = self.fault_risk_from_predictions(probabilities)
        degradation_rate = self.degradation_rate_from_predictions(probabilities)
        return self.calculate(fault_risk, degradation_rate, access_cost, criticality)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PriorityScore from real FaultRisk/DegradationRate/AccessCost/"
            "Criticality values via the existing prioritization pipeline. Does not "
            "invent or fabricate any component value - all four must be supplied."
        )
    )
    parser.add_argument("--fault-risk", type=float, required=True, help="FaultRisk value")
    parser.add_argument("--degradation-rate", type=float, required=True, help="DegradationRate value")
    parser.add_argument("--access-cost", type=float, required=True, help="AccessCost value")
    parser.add_argument("--criticality", type=float, required=True, help="Criticality value")
    parser.add_argument("--w-fault-risk", type=float, default=None, help="override w1 (default: PrioritizationWeights default)")
    parser.add_argument("--w-degradation-rate", type=float, default=None, help="override w2 (default: PrioritizationWeights default)")
    parser.add_argument("--w-access-cost", type=float, default=None, help="override w3 (default: PrioritizationWeights default)")
    parser.add_argument("--w-criticality", type=float, default=None, help="override w4 (default: PrioritizationWeights default)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args, validate, and delegate to calculate_priority."""
    args = _build_arg_parser().parse_args(argv)

    weights = None
    if any(
        w is not None
        for w in (args.w_fault_risk, args.w_degradation_rate, args.w_access_cost, args.w_criticality)
    ):
        defaults = PrioritizationWeights()
        weights = PrioritizationWeights(
            w1_fault_risk=args.w_fault_risk if args.w_fault_risk is not None else defaults.w1_fault_risk,
            w2_degradation_rate=(
                args.w_degradation_rate if args.w_degradation_rate is not None else defaults.w2_degradation_rate
            ),
            w3_access_cost=args.w_access_cost if args.w_access_cost is not None else defaults.w3_access_cost,
            w4_criticality=args.w_criticality if args.w_criticality is not None else defaults.w4_criticality,
        )

    # calculate_priority delegates to core.calculate_priority_score, which
    # performs all input/weight validation - no validation is duplicated here.
    score = calculate_priority(
        fault_risk=args.fault_risk,
        degradation_rate=args.degradation_rate,
        access_cost=args.access_cost,
        criticality=args.criticality,
        weights=weights,
    )
    print(f"PriorityScore={score}")


if __name__ == "__main__":
    main()
