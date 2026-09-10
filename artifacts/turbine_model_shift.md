# Turbine-Model Shift Analysis (Task 18)

Reuses Task 16's (Emin) raw feature statistics rather than recomputing them - see script docstring.

## Farm configuration (documented, from Gueck & Roelofs 2024)

| Farm | Location | Turbines | Raw sensors |
|---|---|---:|---:|
| A | Onshore, Portugal | 5 | 86 |
| B | Offshore, Germany | 9 | 257 |
| C | Offshore, Germany | 22 | 957 |

## Gearbox ratio evidence (generator_rotor_speed_ratio, raw units)

| Farm | Raw mean | Raw std |
|---|---:|---:|
| C (source) | 118.785 | 0.971 |
| A (target) | 113.002 | 0.433 |
| B (target) | 115.516 | 0.200 |

Each farm's generator/rotor speed ratio clusters tightly around a distinct mean (std < 1.0 in all three farms) - consistent with each farm using a fixed gearbox ratio specific to its turbine model, not a shared design. This matches the sensor mapping's own quality note: 'nominal gear ratios may differ' (configs/physical_sensor_mapping.yaml).


Turbine count and raw sensor count differ by roughly an order of magnitude across farms (5/86 for A vs 22/957 for C) - this alone does not prove different turbine models, but combined with the tightly-clustered, farm-specific gearbox ratio below, it supports treating each farm as a genuinely distinct turbine population rather than the same hardware at different sites.

