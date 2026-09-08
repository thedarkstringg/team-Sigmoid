#!/bin/bash
set -e  # fail fast - don't continue past a broken stage

CHECKPOINT="checkpoints/farm_c_power_residual/best.pt"
DATA_DIR="data/processed/CARE_Farm_C/sequences_v2"
HIDDEN_SIZE=32
NUM_LAYERS=2
DROPOUT=0.3

echo "=== Stage 1: Farm C baseline (Ismayil) ==="
python3 src/baseline/train_farm_c_baseline.py --data-dir "$DATA_DIR"

echo "=== Stage 2: GRU training (Ziyad) ==="
python3 src/model/train.py --use_real_data --epochs 50 --patience 8 \
  --lr 0.0001 --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --dropout $DROPOUT \
  --data_dir "$DATA_DIR" \
  --checkpoint_dir "$(dirname "$CHECKPOINT")"

echo "=== Stage 3: GRU evaluation on Farm C (Ziyad) ==="
python3 src/model/evaluate.py \
  --checkpoint "$CHECKPOINT" --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --dropout $DROPOUT \
  --data_dir "$DATA_DIR"

echo "=== Stage 4: Export per-timestep predictions for evaluation/prioritization (Ziyad/Emin) ==="
for SPLIT in val test; do
  python3 -m src.eval.export_gru_predictions \
    --checkpoint "$CHECKPOINT" \
    --data-dir "$DATA_DIR" \
    --metadata-parquet "$DATA_DIR/${SPLIT}_metadata.parquet" \
    --split "$SPLIT" \
    --output-parquet "artifacts/predictions/farm_c_${SPLIT}_predictions.parquet" \
    --input-size 11 --hidden-size $HIDDEN_SIZE --num-layers $NUM_LAYERS --dropout $DROPOUT
done

echo "=== Stage 5: Cross-farm generalization check (Kamal) ==="
# Uses the 10-feature checkpoint specifically - Farm A/B were re-exported as
# the strict zero-shot schema (power residual excluded on purpose, since it
# needs target-domain calibration that would break zero-shot validity). The
# 11-feature power-residual checkpoint cannot load against this data - see
# FEATURE_PIVOT_NOTE.md.
python3 -m src.eval.cross_farm_evaluate \
  --checkpoint checkpoints/farm_c_lr5e4/best.pt \
  --data_dir data/processed/CARE_Farm_A/physical_sequences \
  --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --dropout 0.4
python3 -m src.eval.cross_farm_evaluate \
  --checkpoint checkpoints/farm_c_lr5e4/best.pt \
  --data_dir data/processed/CARE_Farm_B/physical_sequences \
  --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --dropout 0.4

echo ""
echo "=== Stages 6-7: Evaluation figures (Emin) and prioritization ranking (Kamal) ==="
echo "Emin's metrics.py / plots.py currently run interactively rather than via a"
echo "standalone CLI script: import functions directly against"
echo "artifacts/predictions/farm_c_{val,test}_predictions.parquet"
echo "(see src/eval/metrics.py docstring for the required table schema)"
echo ""
echo "Kamal's prioritization pipeline now has a real, non-interactive CLI."
echo "It requires real FaultRisk/DegradationRate/AccessCost/Criticality values"
echo "(no synthetic example values); substitute the actual numbers for a given"
echo "turbine/asset, e.g.:"
echo "  python3 -m src.prioritization.pipeline \\"
echo "    --fault-risk <value> --degradation-rate <value> \\"
echo "    --access-cost <value> --criticality <value>"
echo ""
echo "=== CLI STAGES COMPLETE ==="