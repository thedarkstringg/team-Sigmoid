# Examples

## prioritization_example.py

Demonstrates how the existing prioritization system in `src/prioritization/`
computes a `PriorityScore` for multiple turbines and ranks them.

### What it shows

Given a small synthetic set of 5 turbines with four input factors each
(`FaultRisk`, `DegradationRate`, `AccessCost`, `Criticality`), the script
calls the public `calculate_priority(...)` API to evaluate:

```
PriorityScore = w1*FaultRisk + w2*DegradationRate - w3*AccessCost + w4*Criticality
```

using the default `PrioritizationWeights` (all `w = 1.0`), then prints the
turbines ranked by `PriorityScore` in descending order.

### The values are synthetic

All input numbers in this example are illustrative, hand-verifiable values
invented for demonstration only - they are **not** real farm measurements
and are **not** empirical estimates of risk, degradation, cost, or
criticality.

### How to run

From the repository root:

```
python examples/prioritization_example.py
```

### What the output means

The printed table shows, per turbine:

| Column | Meaning |
| --- | --- |
| `Rank` | Position when sorted by `PriorityScore`, highest first |
| `Turbine` | Synthetic turbine identifier |
| `FaultRisk` | Smoothed fault-probability signal |
| `DegradRate` | Slope of fault probability over time (per hour) |
| `AccessCost` | Relative access-burden index (higher = harder to reach) |
| `Criticality` | Relative operational-importance index (higher = more important) |
| `PriorityScore` | Final combined score |

Under the default formula, higher `FaultRisk`, `DegradationRate`, and
`Criticality` increase the score, while higher `AccessCost` decreases it.
