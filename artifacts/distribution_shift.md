# Distribution Shift Analysis (Task 16)

Source: Farm C (train split, the scaler's own fit population). Targets: Farm A test, Farm B test (zero-shot, frozen Farm C scaler).

Compression factor (not standardized mean difference) is the primary shift statistic reported here. See module docstring: raw-value SMD would understate the shift for features where Farm C's own raw variance is itself inflated by an apparent data-quality issue, since that inflation dominates the pooled denominator. Compression factor instead measures what the model actually receives after the frozen scaler is applied, which has no such blind spot.


## Features ranked by worst compression (most shifted first)

| Feature | Farm A compression | Farm B compression |
|---|---:|---:|
| gearbox_bearing_hotspot_over_oil_C | 33.73x | 66.31x |
| gearbox_oil_rise_C | 16.12x | 17.95x |
| grid_frequency_deviation_Hz | 6.98x | 1.51x |
| generator_rotor_speed_ratio | 2.24x | 4.86x |
| grid_voltage_imbalance | 0.69x | 3.35x |
| grid_current_imbalance | 0.37x | 2.42x |
| pitch_angle_deg | 1.78x | 1.09x |
| wind_speed_mps | 1.2x | 0.98x |
| grid_power_factor | 0.85x | 0.5x |
| yaw_misalignment_deg | 0.29x | 0.48x |

![Compression summary](distribution_shift_compression.png)


![Distribution comparison](distribution_shift_features.png)


## Full per-feature statistics (raw units)

| Feature | Farm C mean | Farm C std | Farm A mean | Farm A std | Farm B mean | Farm B std |
|---|---:|---:|---:|---:|---:|---:|
| wind_speed_mps | 8.499 | 3.912 | 7.984 | 3.266 | 9.321 | 3.997 |
| pitch_angle_deg | 3.050 | 7.570 | 0.180 | 4.251 | 4.341 | 6.933 |
| yaw_misalignment_deg | 3.029 | 2.339 | 7.633 | 8.097 | 4.915 | 4.892 |
| gearbox_oil_rise_C | 44.689 | 91.550 | 29.697 | 5.679 | 42.271 | 5.102 |
| gearbox_bearing_hotspot_over_oil_C | -12.287 | 91.147 | 6.193 | 2.703 | 6.743 | 1.375 |
| generator_rotor_speed_ratio | 118.785 | 0.971 | 113.002 | 0.433 | 115.516 | 0.200 |
| grid_power_factor | 0.976 | 0.092 | 0.934 | 0.108 | 0.934 | 0.184 |
| grid_current_imbalance | 0.014 | 0.016 | 0.045 | 0.042 | 0.005 | 0.006 |
| grid_voltage_imbalance | 0.001 | 0.001 | 0.004 | 0.001 | 0.000 | 0.000 |
| grid_frequency_deviation_Hz | 0.087 | 0.084 | 0.001 | 0.012 | 0.101 | 0.056 |
