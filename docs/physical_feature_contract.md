# Cross-farm physical feature contract

`physical-v1` is the strict domain-generalization representation for CARE
Farms A, B, and C. Farm C is the source domain. Farms A and B are external
test domains and never participate in feature selection, scaler fitting, or
threshold selection.

## Strict ordered schema

1. `wind_speed_mps`
2. `pitch_angle_deg`
3. `yaw_misalignment_deg`
4. `gearbox_oil_rise_C`
5. `gearbox_bearing_hotspot_over_oil_C`
6. `generator_rotor_speed_ratio`
7. `grid_power_factor`
8. `grid_current_imbalance`
9. `grid_voltage_imbalance`
10. `grid_frequency_deviation_Hz`

The authoritative mappings, formulas, units, quality notes, and validity
conditions are in `configs/physical_sensor_mapping.yaml`. The ordered tabular
manifest at `artifacts/data/physical_feature_manifest.csv` is generated with:

```bash
python -m src.data.build_physical_manifest
```

All inputs are 10-minute **Avg** channels. A `max()` in the gearbox hotspot
means the maximum across simultaneous bearing Avg channels; it never means a
raw maximum-statistic channel. The strict schema does not include transformer
temperature, vibration, generator thermal features, or a power residual.

## Validity and filling

The speed ratio is invalid below 0.5 rotor rpm. Power factor and phase
imbalances are invalid below their configurable load/voltage floors. Frequency
deviation is disabled until the per-farm audit proves that the selected signal
is centered within 0.5 Hz of 50 Hz.

An exported timestep mask is one only when the timestamp was observed and all
physical features passed their validity conditions. Values may be forward/back
filled only inside a window that already passes the 95% coverage and 30-minute
maximum-gap rules. Filled or physically invalid timesteps remain mask zero.
Windows that cannot be filled to finite values are rejected.

Farm B/C zero runs are audited but zero is not globally converted to missing:
zero is a legitimate value for power, speed, current, and several other
signals. A raw-data review must approve suspicious runs plus pitch/yaw ranges.

## Temporal and metadata contract

- 10-minute interval
- 24-hour lookback / 144 timesteps
- one-hour sequence stride
- per-timestep positive label when `0 < hours_to_fault <= 48`
- no sequence crosses a timestamp gap longer than 30 minutes
- `X=(N,144,F)` with dynamic `F`; strict `physical-v1` has `F=10`
- `y` and `mask` are `(N,144)` binary arrays
- `{split}_metadata.parquet` contains exactly `N*144` sequence-major rows
- `window_end` is the timestamp of each timestep, not the enclosing sequence end

## Normalization and leakage

The z-score scaler is fit on valid Farm C training timesteps only. Its saved
provenance is `source_farm=C`, `source_split=train`. The identical frozen
scaler is applied to Farm C validation/test and both external farms. External
Farm A/B statistics do not alter scaling, features, thresholds, or model
selection.

## Optional power-residual ablation

The power residual is disabled by default and appends an eleventh feature only
when `--include-power-residual` is supplied. Its expected curve uses wind-speed
bins, median active power per bin, and interpolation.

For Farm C the exporter fits the curve only from `train_test=train` rows of
normal events belonging to Farm C training assets. Farm A/B fitting is refused unless
`--allow-target-power-calibration` is explicitly supplied. Such a run is a
calibrated target-domain ablation, not zero-shot domain generalization.
