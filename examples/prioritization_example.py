"""
Practical prioritization example: rank multiple turbines by PriorityScore.

Uses the existing public API from src/prioritization/ to compute
PriorityScore = w1*FaultRisk + w2*DegradationRate - w3*AccessCost + w4*Criticality
for a small set of synthetic turbines and ranks them.

Interpretation of each factor in the current formula:
  - higher FaultRisk        -> higher PriorityScore (w1 > 0)
  - higher DegradationRate  -> higher PriorityScore (w2 > 0)
  - higher Criticality      -> higher PriorityScore (w4 > 0)
  - higher AccessCost       -> lower PriorityScore  (AccessCost is subtracted, w3 > 0)

All input values in this example are SYNTHETIC/ILLUSTRATIVE - they are not
real farm measurements and are not empirical estimates. Run from the repo root:

    python examples/prioritization_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.prioritization import PrioritizationWeights, calculate_priority

# Hand-computed with the default weights (w1=w2=w3=w4=1.0):
#   T-01: 0.85 + 0.30 - 0.20 + 0.90 = 1.85
#   T-02: 0.75 + 0.10 - 0.60 + 0.40 = 0.65
#   T-03: 0.30 + 0.05 - 0.10 + 0.35 = 0.60
#   T-04: 0.50 + 0.40 - 0.80 + 0.50 = 0.60
#   T-05: 0.20 + 0.10 - 0.15 + 0.25 = 0.40
# T-03 and T-04 tie at 0.60; stable sort keeps T-03 ahead of T-04.
TURBINES = {
    "id": ["T-01", "T-02", "T-03", "T-04", "T-05"],
    "fault_risk": np.array([0.85, 0.75, 0.30, 0.50, 0.20]),
    "degradation_rate": np.array([0.30, 0.10, 0.05, 0.40, 0.10]),
    "access_cost": np.array([0.20, 0.60, 0.10, 0.80, 0.15]),
    "criticality": np.array([0.90, 0.40, 0.35, 0.50, 0.25]),
}


def main() -> None:
    weights = PrioritizationWeights()  # default illustrative weights (all 1.0)

    scores = calculate_priority(
        fault_risk=TURBINES["fault_risk"],
        degradation_rate=TURBINES["degradation_rate"],
        access_cost=TURBINES["access_cost"],
        criticality=TURBINES["criticality"],
        weights=weights,
    )

    ranked_indices = np.argsort(-scores, kind="stable")

    print("PriorityScore = w1*FaultRisk + w2*DegradationRate - w3*AccessCost + w4*Criticality")
    print("Default weights: w1=w2=w3=w4=1.0 | Synthetic inputs, illustrative only\n")

    header = f"{'Rank':<4} {'Turbine':<8} {'FaultRisk':>9} {'DegradRate':>10} {'AccessCost':>10} {'Criticality':>11} {'PriorityScore':>13}"
    print(header)
    print("-" * len(header))

    for rank, idx in enumerate(ranked_indices, start=1):
        print(
            f"{rank:<4} {TURBINES['id'][idx]:<8} "
            f"{TURBINES['fault_risk'][idx]:>9.2f} "
            f"{TURBINES['degradation_rate'][idx]:>10.2f} "
            f"{TURBINES['access_cost'][idx]:>10.2f} "
            f"{TURBINES['criticality'][idx]:>11.2f} "
            f"{scores[idx]:>13.2f}"
        )


if __name__ == "__main__":
    main()
