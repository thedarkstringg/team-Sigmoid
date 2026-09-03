# CARE Farm A Data Contract

Raw CARE Farm A CSV files and generated NumPy arrays are not tracked by Git.

## Sequence export

Run:

    python src/data/export_sequences.py \
      --raw-dir /path/to/CARE_Farm_A \
      --output-dir data/processed/CARE_Farm_A/sequences

Validate:

    python src/data/validate_sequences.py \
      --data-dir data/processed/CARE_Farm_A/sequences

## Sequence definition

- 54 average SCADA channels
- 10-minute SCADA resolution
- 24-hour lookback
- 144 timesteps per sequence
- 1-hour stride
- 48-hour fault horizon

Asset-disjoint split:

- Train: assets 0, 10, 11
- Validation: asset 21
- Test: asset 13

## Generated arrays

For each split:

- `*_X.npy`: `(N, 144, 54)`, float32
- `*_y.npy`: `(N, 144)`, uint8
- `*_mask.npy`: `(N, 144)`, uint8

The mask is 1 for real observations and 0 for approved short gap-filled timesteps.

Sequences with less than 95% coverage or an internal timestamp gap greater than 30 minutes are rejected.

## Scaling

Features are standardized using per-feature mean and standard deviation fitted only on real training timesteps.

The same training statistics are applied to validation and test data.

Scaler metadata is saved in `scaler_stats.npz`.

## Class imbalance

The exporter does not resample data.

The GRU training code computes positive-class weighting from the training split only and ignores masked timesteps.
