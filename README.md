# Team Sigmoid: cross-farm wind-turbine fault detection

The project uses Farm C as the source domain and evaluates transfer to CARE
Farms A and B through a strict ten-feature physical representation. The
authoritative feature mapping is `configs/physical_sensor_mapping.yaml` and
the server workflow is documented in `SERVER_RUNBOOK.md`.

No raw CARE data, generated NumPy/Parquet data, checkpoints, or Kaggle caches
belong in Git.
