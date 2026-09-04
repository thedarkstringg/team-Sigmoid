# Shared Linux server runbook

These commands are for the shared server only. They have not been run on the
local Windows checkout. This runbook assumes the checkout is
`/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm`; change `REPO` once if the
existing checkout uses another directory name.

## 1. Pull the branch and install runtime dependencies

The first server command is:

```bash
cd /sdb-disk/notebooks/team12/
```

Then run:

```bash
export REPO=/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm
cd "$REPO"
git fetch origin
git switch ismayil/cross-farm-physical
git pull --ff-only origin ismayil/cross-farm-physical
python -m pip install -r requirements.txt
export RAW_ROOT="$REPO/data/raw"
mkdir -p "$RAW_ROOT" "$REPO/data/audits"
```

## 2. Download only Farms B and C

Kaggle credentials must already be configured for `kagglehub`.

```bash
python scripts/download_care_farm.py --farm B --output-root "$RAW_ROOT"
python scripts/download_care_farm.py --farm C --output-root "$RAW_ROOT"
```

Farm A is assumed to exist at `$RAW_ROOT/CARE_Farm_A`. If it does not, download
only that farm with the same script:

```bash
python scripts/download_care_farm.py --farm A --output-root "$RAW_ROOT"
```

## 3. Audit the selected Avg channels

```bash
python src/data/audit_crossfarm.py --farm A --raw-dir "$RAW_ROOT/CARE_Farm_A" --output data/audits/farm_a.json
python src/data/audit_crossfarm.py --farm B --raw-dir "$RAW_ROOT/CARE_Farm_B" --output data/audits/farm_b.json
python src/data/audit_crossfarm.py --farm C --raw-dir "$RAW_ROOT/CARE_Farm_C" --output data/audits/farm_c.json
```

Before export, inspect all three JSON reports. Confirm:

- `feature_availability_failures` is empty;
- `frequency_validation.passed` is true;
- suspicious zero runs are understood, particularly for Farms B/C;
- pitch and raw yaw ranges/sign conventions match their descriptions;
- generator/rotor ratio valid fractions are plausible;
- no selected channel is constant or dominated by non-finite values.

Do not continue merely because `strict_export_ready` is true if the human
pitch/yaw or zero-run review fails.

## 4. Generate and freeze the real Farm C split

This command reads the actual event files to discover assets. It does not
fabricate asset IDs locally.

```bash
python src/data/generate_farm_c_split.py \
  --raw-dir "$RAW_ROOT/CARE_Farm_C" \
  --seed 42 \
  --output configs/farm_c_split.json
```

Review `configs/farm_c_split.json`: validation and test must each have multiple
assets and anomaly/normal event coverage. Once approved, preserve this exact
file with the experiment artifacts (and commit it separately if the team wants
the server-derived split frozen in Git).

## 5. Export Farm C source sequences

```bash
python src/data/export_physical_sequences.py \
  --farm C \
  --raw-dir "$RAW_ROOT/CARE_Farm_C" \
  --audit-report data/audits/farm_c.json \
  --split-config configs/farm_c_split.json \
  --output-dir data/processed/CARE_Farm_C/sequences
```

This creates the Farm-C-train scaler. Do not train a model before the strict
validator in step 7 passes.

## 6. Export external Farm A and B tests with the frozen C scaler

```bash
python src/data/export_physical_sequences.py \
  --farm A \
  --raw-dir "$RAW_ROOT/CARE_Farm_A" \
  --audit-report data/audits/farm_a.json \
  --scaler data/processed/CARE_Farm_C/sequences/scaler_stats.npz \
  --output-dir data/processed/CARE_Farm_A/physical_sequences

python src/data/export_physical_sequences.py \
  --farm B \
  --raw-dir "$RAW_ROOT/CARE_Farm_B" \
  --audit-report data/audits/farm_b.json \
  --scaler data/processed/CARE_Farm_C/sequences/scaler_stats.npz \
  --output-dir data/processed/CARE_Farm_B/physical_sequences
```

Do not pass `--include-power-residual`; these commands are the strict
10-feature zero-shot DG exports.

## 7. Validate all strict exports

```bash
python src/data/validate_physical_pipeline.py \
  --farm-c-dir data/processed/CARE_Farm_C/sequences \
  --farm-a-dir data/processed/CARE_Farm_A/physical_sequences \
  --farm-b-dir data/processed/CARE_Farm_B/physical_sequences \
  --split-config configs/farm_c_split.json
```

Expected final line: `ALL STRICT PHYSICAL-V1 EXPORT CHECKS PASSED`.

## 8. Reconstruct metadata for the old 54-feature Farm A arrays

The existing arrays must be at `data/processed/CARE_Farm_A/sequences`. This
utility does not regenerate `X`; it replays legacy selection and proves every
old `y` and `mask` row before writing metadata.

```bash
python src/data/reconstruct_farm_a_metadata.py \
  --raw-dir "$RAW_ROOT/CARE_Farm_A" \
  --arrays-dir data/processed/CARE_Farm_A/sequences \
  --output-dir data/processed/CARE_Farm_A/sequences
```

It must report exact sequence counts: train 4357, validation 1092, test 778.
Any mismatch is a hard failure and must not be bypassed.

## 9. Export old Farm A checkpoint predictions for Emin

The repository documentation identifies the old checkpoint as
`checkpoints/h32_dropout/best.pt`. The exporter infers input size, hidden size,
and layer count from its weights and writes probabilities without fitting a
threshold.

```bash
python src/eval/export_gru_predictions.py \
  --checkpoint checkpoints/h32_dropout/best.pt \
  --data-dir data/processed/CARE_Farm_A/sequences \
  --metadata-parquet data/processed/CARE_Farm_A/sequences/val_metadata.parquet \
  --split val \
  --output-parquet artifacts/predictions/val_predictions.parquet

python src/eval/export_gru_predictions.py \
  --checkpoint checkpoints/h32_dropout/best.pt \
  --data-dir data/processed/CARE_Farm_A/sequences \
  --metadata-parquet data/processed/CARE_Farm_A/sequences/test_metadata.parquet \
  --split test \
  --output-parquet artifacts/predictions/test_predictions.parquet
```

If a different old checkpoint is authoritative, change only `--checkpoint`.
Do not fabricate outputs when the checkpoint or reconstructed metadata is
absent.
