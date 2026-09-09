# Cross-Farm Generalization Results (Task 15)

**Checkpoint:** `/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/checkpoints/farm_c_lr5e4/best.pt` (farm_c_lr5e4)

**Config:** GRU, input_size=10, hidden_size=32, num_layers=2, trained_epoch=0, best_val_loss=1.3349884762821427

**Frozen threshold:** 0.47500000000000003 (selected via max-F1 on Farm C validation only; F1 at threshold = 0.1597. NOT re-tuned on Farm A or Farm B.)

**Preprocessing:** scaler fit on C (train split) only, features: wind_speed_mps, pitch_angle_deg, yaw_misalignment_deg, gearbox_oil_rise_C, gearbox_bearing_hotspot_over_oil_C, generator_rotor_speed_ratio, grid_power_factor, grid_current_imbalance, grid_voltage_imbalance, grid_frequency_deviation_Hz. Reused unchanged on all splits.


## Superseded prior result (kept for provenance)

An earlier cross-farm attempt in this repository used `checkpoints/farm_c_final/best.pt` (**farm_c_final**), verified input_size=10, config {'hidden_size': 32, 'num_layers': 2, 'dropout': 0.3, 'bidirectional': False}.

> **Correction to the original writeup:** The original writeup for this run stated farm_c_final was 'also previously referred to as' the 11-feature farm_c_power_residual checkpoint. This is incorrect: farm_c_final's checkpoint weights were directly verified (infer_model_shape) to have input_size=10, not 11. Corrected here rather than repeated.

| Split | Role | ROC-AUC | PR-AUC | Precision | Recall | F1 | threshold |
|---|---|---|---|---|---|---|---|
| Farm C test | Source-domain test | 0.47513 | 0.037999 | 0.028658 | 0.624541 | 0.054801 | 0.475 |

*Not evaluated for this checkpoint; superseded before Farm A/B were run.*


Source: artifacts/cross_farm_results.md as committed in PR #23/#24 (commits ddf0264, 1b49625), prior to this run.


## Results table

| Split | Role | ROC-AUC | PR-AUC | Precision | Recall | F1 | n valid | n positive | pos. rate |
|---|---|---|---|---|---|---|---|---|---|
| Farm C test | Source-domain test | 0.336 | 0.0273 | 0.0275 | 0.5795 | 0.0524 | 290793 | 10611 | 0.0365 |
| Farm A test | Zero-shot target | 0.4672 | 0.0526 | 0.0608 | 0.2431 | 0.0972 | 622715 | 34934 | 0.0561 |
| Farm B test | Zero-shot target | 0.486 | 0.0217 | 0.1018 | 0.0044 | 0.0084 | 1214700 | 25244 | 0.0208 |

## Per-asset breakdown (appendix)

| Split | Asset | n | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Farm C test (source) | 16 | 104636 | 0.4512 | 0.0386 | 0.994 | 0.0744 |
| Farm C test (source) | 23 | 122704 | 0.4142 | 0.0174 | 0.9943 | 0.0341 |
| Farm C test (source) | 56 | 63453 | 0.5667 | 0.0744 | 0.0188 | 0.0301 |
| Farm C test (source) | ALL | 290793 | 0.336 | 0.0275 | 0.5795 | 0.0524 |
| Farm A test (zero-shot) | 0 | 138848 | 0.3722 | 0.0601 | 0.0876 | 0.0713 |
| Farm A test (zero-shot) | 10 | 158420 | 0.5935 | 0.0731 | 0.6346 | 0.1311 |
| Farm A test (zero-shot) | 11 | 109828 | 0.5553 | 0.0346 | 0.4253 | 0.0641 |
| Farm A test (zero-shot) | 13 | 82843 | 0.5278 | 0.0763 | 0.219 | 0.1132 |
| Farm A test (zero-shot) | 21 | 132776 | 0.5019 | 0.0279 | 0.0066 | 0.0107 |
| Farm A test (zero-shot) | ALL | 622715 | 0.4672 | 0.0608 | 0.2431 | 0.0972 |
| Farm B test (zero-shot) | 0 | 22809 | nan | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 2 | 230832 | nan | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 5 | 33144 | nan | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 6 | 161044 | 0.4479 | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 7 | 226590 | 0.1909 | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 11 | 105675 | 0.4827 | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 12 | 203549 | 0.416 | 0.2207 | 0.0371 | 0.0635 |
| Farm B test (zero-shot) | 13 | 125094 | 0.4471 | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | 14 | 105963 | 0.5592 | 0.0 | 0.0 | 0.0 |
| Farm B test (zero-shot) | ALL | 1214700 | 0.486 | 0.1018 | 0.0044 | 0.0084 |
