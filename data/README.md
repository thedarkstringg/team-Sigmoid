# CARE data contracts

Raw CARE CSVs, generated NumPy arrays, Parquet metadata, scalers, and model
files are not tracked by Git. See `SERVER_RUNBOOK.md` for the server commands.

## Strict cross-farm physical export

Farm C is the training/source domain. Its output is:

```text
data/processed/CARE_Farm_C/sequences/
  train_X.npy  train_y.npy  train_mask.npy  train_metadata.parquet
  val_X.npy    val_y.npy    val_mask.npy    val_metadata.parquet
  test_X.npy   test_y.npy   test_mask.npy   test_metadata.parquet
  scaler_stats.npz
  export_summary.json
```

External Farm A/B outputs contain the same `test_*` files under
`physical_sequences/`. All three use the exact ten-feature order in
`artifacts/data/physical_feature_manifest.csv`. The Farm C scaler is fit on
valid training timesteps only and reused unchanged everywhere else.

Sequences have 144 ten-minute timesteps, a 24-hour lookback, one-hour stride,
and per-timestep labels for a 48-hour early-fault horizon. The timestep mask is
one only for observed, physically valid feature rows.

## Legacy Farm A export

The old Farm A arrays remain a separate 54-channel contract under
`data/processed/CARE_Farm_A/sequences`. They can receive aligned per-timestep
metadata through `src/data/reconstruct_farm_a_metadata.py`; the utility does
not regenerate their `X` arrays and fails unless it reproduces their y/mask
ordering exactly.
