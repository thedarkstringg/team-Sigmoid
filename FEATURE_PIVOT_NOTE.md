# Feature Representation Pivot — Raw Sensors → Physical Subsystem Vectors

**Author:** Ziyad
**Date:** 2026-09-04 (follow-up to PIVOT_NOTE.md's Farm A → Farm C pivot)

## What I changed and why

I moved the model's input away from raw per-farm sensor channels, which
differ completely in count and schema across farms (54 / 63 / 238+ features
for Farms A/B/C). Training a model on one farm's raw sensors made it
impossible to evaluate that same model on a different farm without a
dimension mismatch — which directly blocked the generalization study our
proposal always intended.

The fix: I asked Ismayil to build a small set of **physically-meaningful,
farm-agnostic subsystem features**, computed identically regardless of
which farm the data comes from, using each farm's sensor description file
to map differently-named raw channels onto the same physical quantities.

I originally sketched five illustrative features (power residual, two
thermal deltas, a kinematic ratio, a vibration index) as a starting point.
Ismayil's actual audit and mapping work (PR #4, `ismayil/cross-farm-physical`)
resulted in a different, better-justified final schema — **10 features**,
none of them a raw vibration statistic, since Zenodo's own documentation
flags Min/Max/Std reliability problems for Farms B/C, and Ismayil dropped
that feature category rather than risk building on known-contaminated data:

| # | Feature | What It Captures |
|---|---|---|
| 1 | `wind_speed_mps` | Wind speed |
| 2 | `pitch_angle_deg` | Blade pitch angle |
| 3 | `yaw_misalignment_deg` | Nacelle-to-wind yaw misalignment |
| 4 | `gearbox_oil_rise_C` | Gearbox oil temp above ambient |
| 5 | `gearbox_bearing_hotspot_over_oil_C` | Hottest gearbox bearing above oil temp |
| 6 | `generator_rotor_speed_ratio` | Generator RPM / rotor RPM |
| 7 | `grid_power_factor` | Active power / apparent power |
| 8 | `grid_current_imbalance` | Max phase-current deviation from mean |
| 9 | `grid_voltage_imbalance` | Max phase-voltage deviation from mean |
| 10 | `grid_frequency_deviation_Hz` | Deviation from expected 50 Hz grid frequency |

Exact mappings, formulas, and per-farm sensor sources live in
`configs/physical_sensor_mapping.yaml` and
`artifacts/data/physical_feature_manifest.csv`.

I updated my model code (`model.py`, `train.py`, `evaluate.py`) so
`input_size` is **inferred directly from the data's shape at load time**,
not hardcoded — this is why the schema changing from my original 5-feature
sketch to Ismayil's final 10-feature version required zero changes on my
side.

## Why this is a better approach than the Farm A → Farm C pivot alone

Our first pivot (documented in `PIVOT_NOTE.md`) addressed the single-turbine
variance problem I found in Farm A's results, but it didn't address the
cross-farm schema mismatch. A Farm-C-trained model still couldn't be
evaluated on Farm A or B data, since raw feature counts don't match between
farms. Physical subsystem aggregation solves this properly: the same 10
features are computed for every farm, so I can evaluate one trained
checkpoint directly on a different farm's data with zero dimension
mismatch — a real, architecturally clean generalization test, not a
workaround.

## Architecture decision: 2-layer GRU

I'm keeping 2 layers, not the 3-layer variant I tested earlier on Farm A's
raw sensors — that experiment showed no meaningful improvement over 2
layers, consistent with my earlier finding (hidden=64 vs. 32 tied exactly)
that added capacity doesn't help on this dataset size. Simpler stays the
default until an actual experiment justifies otherwise.

## What I changed in the code (done and pushed to `main`)

- `model.py`: `input_size` is now a required, explicit argument (no
  hardcoded default), with a docstring explaining it depends on whatever
  Ismayil's pipeline produces.
- `train.py` / `evaluate.py`: `input_size` is inferred automatically from
  the loaded data's shape (`X.shape[-1]`).
- Defaults: `hidden_size=32`, `dropout=0.3`, `num_layers=2` — my
  best-performing configuration from Farm A tuning, kept as the Farm C
  starting point.
- Default `data_dir` points at Farm C as the primary training target.
- `evaluate.py` documents that pointing `--data_dir` at a *different*
  farm's exported sequences than the training farm is exactly how to run
  Kamal's cross-farm generalization study.

## Current blocker (as of this writing)

Ismayil's `audit_crossfarm.py` is reading Farm C's raw CSVs with the wrong
delimiter — the files are semicolon-delimited, not comma-delimited despite
the `comma_*.csv` filenames, so the audit silently found zero usable
columns across all 58 events. I found and reported this to him; waiting on
his fix before the split/export/validate steps can run for real.

## What this means for Kamal's generalization study

This resolves the open framing question from `KAMAL_ROADMAP.md` (Task 3):
since the 10 subsystem features are farm-agnostic by construction, his
generalization experiment is straightforwardly: train on Farm C, evaluate
the same checkpoint directly on Farm A and/or Farm B's exported sequences.
No shared-feature-subset workaround needed anymore.

## What this means for Emin's evaluation work

No change to his metric logic. `prob`, `label`, `mask`, `asset_id`,
`event_id`, `window_end` per timestep are unaffected by whether the
underlying features are raw sensors or the 10 physical subsystem features.
His work continues unblocked against the existing Farm A
checkpoint/predictions table.