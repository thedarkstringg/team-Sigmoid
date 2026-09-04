# Project Pivot Note — Farm A → Farm C

**Date:** 2026-09-03/04

## What changed

Initial development (see commits through `[last Farm A commit hash]`) trained
and evaluated the GRU model on Wind Farm A (5 turbines, 54 sensors, 86
features). We are moving the core training/evaluation target to **Wind Farm
C** (22 turbines, 238 sensors, 957 features) instead.

## Why

Farm A results (see `docs/farm_a_results.md` or equivalent write-up) showed:
- Val ROC-AUC 0.77–0.84 across configurations, but test ROC-AUC 0.36–0.44
  (below the LogReg baseline's 0.66).
- This gap was consistent across 3 architecture/regularization variants
  (hidden size 32/64, with/without dropout), suggesting the cause was
  structural, not a tunable hyperparameter.
- Root cause: Farm A has only 5 turbines total. Val and test splits are
  single turbines each, giving high-variance, low-confidence estimates of
  generalization. Val turbine fault rate (5.3%) and test turbine fault rate
  (9.1%) differ substantially.

Farm C has **22 turbines** (vs. Farm A's 5), allowing a train/val/test split
with multiple turbines per split, which should give more stable and
meaningful generalization estimates.

## What carries over vs. what's being rebuilt

**Rebuilt from scratch (Ismayil):** data quality audit, sliding-window
construction, splits, baseline models, sequence export — all specific to
Farm C's 957 features / 238 sensors and its documented data quality
challenge (zero-filled missing values, per Gück et al. 2024).

**Reused as-is (Ziyad):** model architecture code (`TemporalRiskModel`),
training loop, masking logic, evaluation script — these are farm-agnostic
and only need `input_size` updated to match Farm C's sensor count once
confirmed.

**Superseded, kept for reference:** all Farm A results, the Farm A Method
section draft, and Farm A baseline comparisons remain in git history and
are referenced in the paper's [Discussion/Limitations or Appendix] as the
basis for this design decision, not deleted.