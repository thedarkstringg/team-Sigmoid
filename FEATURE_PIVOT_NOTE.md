# Feature Representation Pivot — Raw Sensors → Physical Subsystem Vectors

**Date:** 2026-09-04 (follow-up to PIVOT_NOTE.md's Farm A → Farm C pivot)

## What changed

The model's input is no longer raw per-farm sensor channels (which differ
completely in count and schema across farms: 54 / 63 / 238+ features for
Farms A/B/C). Instead, the data pipeline maps each farm's raw sensors into
a small set of **physically-meaningful, farm-agnostic subsystem features**,
computed identically regardless of which farm the data comes from:

| Subsystem Vector | What It Aggregates |
|---|---|
| Power Residual | Actual Active Power − Theoretical Power(Wind Speed) |
| Thermal Delta (gearbox) | Gearbox Bearing Temp − Ambient Temp |
| Thermal Delta (generator) | Generator Bearing Temp − Ambient Temp |
| Kinematic Ratio | Generator RPM / Rotor RPM |
| Vibration Index | Rolling mean of available bearing vibration std channels |

**The exact final set/count of subsystem vectors is not fixed at 5** — it
is determined by Ismayil based on what can actually be reliably computed
across all farms using the sensor description file (see "Open items"
below). The model code has been updated so `input_size` is **inferred
directly from the data's shape at load time**, not hardcoded anywhere, so
this can change without requiring a code update on Ziyad's side.

## Why this is a better approach than the raw-sensor pivot alone

The team's first pivot (Farm A → Farm C, see `PIVOT_NOTE.md`) addressed the
single-turbine variance problem but did not address the cross-farm schema
mismatch — a Farm-C-trained model could not be evaluated directly on Farm
A or B data, since the raw feature counts don't match. Physical subsystem
aggregation solves this: the same 5 (or however many) subsystem features
are computed for every farm, so **one trained model can be evaluated
directly on any farm without a dimension mismatch** — this is a real,
architecturally clean way to run the generalization study the project's
proposal always intended, not a workaround.

## Architecture decision: 2-layer GRU (not 3-layer)

Confirmed: staying with **2 layers**, not the 3-layer variant that was
tested earlier. The 3-layer experiment (on Farm A raw sensors) did not
show a meaningful improvement over 2 layers, consistent with the earlier
finding that added capacity did not help on this dataset size. Simpler
architecture, same reasoning as before.

## What changed in the code (already done)

- `model.py`: `input_size` is now a required, explicit argument (no
  hardcoded default) with a docstring explaining it depends on the final
  subsystem-vector count.
- `train.py` / `evaluate.py`: `input_size` is inferred automatically from
  the loaded data's shape (`X.shape[-1]`) rather than hardcoded — this
  means the code does not need to change again if the subsystem-vector
  count changes.
- Default `hidden_size=32`, `dropout=0.3`, `num_layers=2` — the
  best-performing configuration found during Farm A tuning, kept as the
  starting point for Farm C.
- Default `data_dir` updated to point at Farm C as the primary training
  target.
- `evaluate.py` now explicitly documents that pointing `--data_dir` at a
  *different* farm's exported sequences than the training farm is exactly
  how to run the cross-farm generalization study.

## Open items — for Ismayil

1. **Confirm the sensor description file's exact location/format** for
   all three farms, and share it with the team.
2. **Build the sensor-to-physical-type mapping** per farm using that file
   (which raw sensor is "gearbox bearing temperature," "ambient
   temperature," "active power," "wind speed," "generator RPM," "rotor
   RPM," and bearing vibration channels, for each farm).
3. **Power Residual requires fitting a power curve** (expected power as a
   function of wind speed) — this is not a raw sensor, it must be built,
   typically via binning wind speed and taking the median power per bin
   from *normal-operation training data only*, computed **separately per
   farm** (turbine models/curves differ).
4. **Check the Vibration Index against known data-quality issues** —
   Zenodo's own documentation states Min/Max/Std statistics are largely
   unreliable for Farm B (recommends "Avg signals only") and flags ~17
   specific sensors with high contamination in Farm C. If the vibration
   channels needed for this subsystem vector overlap with flagged/unreliable
   sensors, either exclude those specific sensors or fall back to
   average-only signals for that farm.
5. **Report back the final, confirmed subsystem-vector list and count**
   once built — this determines `input_size` automatically on Ziyad's
   side, no separate sync needed, but the paper's Method section needs
   the exact final list documented.

## What this means for Kamal's generalization study

This directly resolves the open framing question from `KAMAL_ROADMAP.md`
(Task 3): since subsystem vectors are farm-agnostic by construction, the
generalization experiment is now straightforwardly: train on Farm C,
evaluate the same trained checkpoint directly on Farm A and/or Farm B's
exported sequences, no separate "shared feature subset" workaround needed.
This is a cleaner, more direct test of transfer than what was possible
under the raw-sensor approach.

## What this means for Emin's evaluation work

No change to his metric logic — `prob`, `label`, `mask`, `asset_id`,
`event_id`, `window_end` per timestep are unaffected by whether the
underlying features are raw sensors or subsystem vectors. His work
continues unblocked against the existing Farm A checkpoint/predictions
table as before.