# Cross-Farm Zero-Shot Evaluation Results (Task #14)

> **STATUS: results below are for a SUPERSEDED checkpoint. A rerun is required
> for the designated checkpoint before these numbers can be cited as final.**

## 0. Designated checkpoint (correction)

The cross-farm generalization experiment **must** use:

- **Checkpoint:** `checkpoints/farm_c_lr5e4/best.pt`
- **Config:** `hidden_size=32, num_layers=2, dropout=0.4, input_size=10, bidirectional=False`
- **Approx. validation ROC-AUC:** ~0.645 (best-by-validation 10-feature-only configuration)

This checkpoint has **no power_residual feature and is not bidirectional**.
Farm A and Farm B were re-exported using the strict 10-feature physical
schema; `power_residual` was intentionally excluded because it requires
target-domain calibration that would break zero-shot validity. The older
11-feature `checkpoints/farm_c_power_residual/best.pt` (also previously
referred to as `farm_c_final`) is **incompatible** with this 10-feature
external data and must **not** be used as the designated cross-farm
checkpoint.

**The results in Sections 1-6 below were generated with the wrong checkpoint
(`farm_c_final`, 11-feature) and predate this correction.** They are kept
here for provenance only, clearly marked as superseded. This repository does
not yet contain Farm A/B evaluation results for the corrected
`farm_c_lr5e4` checkpoint. Rerun the commands below to produce real numbers
before reporting a final cross-farm result:

```
python3 src/model/evaluate.py \
  --checkpoint checkpoints/farm_c_lr5e4/best.pt \
  --dropout 0.4 \
  --data_dir data/processed/CARE_Farm_A/physical_sequences

python3 src/model/evaluate.py \
  --checkpoint checkpoints/farm_c_lr5e4/best.pt \
  --dropout 0.4 \
  --data_dir data/processed/CARE_Farm_B/physical_sequences
```

The evaluation must remain zero-shot: the threshold stays frozen from Farm C
validation, no tuning occurs on Farm A/B, no scaler is fit on Farm A/B, and
Farm A/B labels are never used to modify the model.

---

## Superseded results (checkpoint: `farm_c_final`, 11-feature - DO NOT reuse as the designated result)

## 1. Experiment setup

- **Training farm:** Farm C
- **Checkpoint used for the results below:** `checkpoints/farm_c_final/best.pt` (superseded - see Section 0)
- **Farm A and Farm B are external, test-only, zero-shot evaluations.** No retraining occurred on either farm, and no threshold tuning was performed on Farm A or Farm B.
- Farm A/B use the frozen Farm C preprocessing/scaler (not independently fitted).
- Same clipping policy applied everywhere: `|x| > 10` clipped to `[-10, 10]`.
- Valid timesteps are selected using the dataset mask (gap-filled rows excluded).

## 2. Checkpoint / model configuration (superseded)

| Parameter | Value |
| --- | --- |
| Architecture | GRU (unidirectional) |
| `input_size` | 10 |
| `hidden_size` | 32 |
| `num_layers` | 2 |
| `dropout` | 0.3 |
| `bidirectional` | false |

## 3. Threshold selection protocol (superseded run)

- The classification threshold was selected on the **Farm C validation split** using **maximum F1**.
- The resulting threshold, **0.475**, was **frozen** and reused unchanged for every test evaluation below (Farm C test, Farm A external test, Farm B external test).
- No threshold was tuned or re-selected on Farm A or Farm B data.

## 4. Results (superseded - wrong checkpoint, kept for provenance only)

| Split | Role | n_timesteps | n_positive | positive_rate | roc_auc | pr_auc | threshold | precision | recall | f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Farm C test | source-domain test | 290793 | 10611 | 0.036490 | 0.475130 | 0.037999 | 0.475 | 0.028658 | 0.624541 | 0.054801 |
| Farm A external test | zero-shot external | 622715 | 34934 | 0.056099 | 0.427255 | 0.048552 | 0.475 | 0.039987 | 0.045772 | 0.042684 |
| Farm B external test | zero-shot external | 1214700 | 25244 | 0.020782 | 0.489700 | 0.020006 | 0.475 | 0.031170 | 0.008160 | 0.012934 |

## 5. Interpretation (superseded run - see Section 0 for the required rerun)

- On its own source domain (Farm C test), the model already shows weak discrimination (ROC-AUC ≈ 0.475, near chance) under the frozen max-F1 threshold, though recall is high (0.625) at the cost of very low precision.
- **Farm A and Farm B are zero-shot external evaluations**: the checkpoint was trained exclusively on Farm C and evaluated on Farm A/B without any retraining or threshold re-tuning.
- On Farm A, ROC-AUC drops further to 0.427 (below chance level), and recall collapses to 0.046 (vs. 0.625 on Farm C) under the same frozen threshold.
- On Farm B, ROC-AUC (0.490) is close to chance, and recall is extremely low (0.008), with precision (0.031) also weaker than on Farm C.
- Across both external farms, F1 (0.043 and 0.013) is substantially lower than on Farm C (0.055), and the pattern is consistent: the frozen Farm-C threshold and scaler do not transfer usefully to either external farm.

## 6. Limitations / generalization note

**These superseded results indicate substantial cross-farm performance degradation, not successful generalization**, and this pattern is the reason a rerun on the designated `farm_c_lr5e4` checkpoint is required rather than assumed to look better:

- ROC-AUC on both external farms is at or below chance level (0.427 and 0.490), indicating the model's ranking of fault risk does not transfer.
- Recall drops drastically on both external farms relative to Farm C, meaning the frozen threshold that was tuned for Farm C's score distribution is poorly calibrated for Farm A/B's score distributions.
- Because no threshold or scaler recalibration was performed on Farm A/B (by design, to keep this a strict zero-shot test), these results should be read as evidence of a domain-shift/generalization gap, not as a deployable cross-farm model.
- Any future work extending this evaluation should not reuse these numbers as a baseline for a "working" cross-farm model without addressing this generalization gap first (e.g. via domain adaptation, farm-specific calibration, or additional cross-farm training data).
