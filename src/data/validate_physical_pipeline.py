"""Validate strict source/external physical-v1 exports and leakage boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.data.physical_features import load_mapping
    from src.data.sequence_utils import HORIZON_HOURS, SEQ_LEN, sha256_file
except ModuleNotFoundError:  # Support direct script invocation from repo root.
    from physical_features import load_mapping
    from sequence_utils import HORIZON_HOURS, SEQ_LEN, sha256_file


def validate_metadata(metadata: pd.DataFrame, y: np.ndarray, mask: np.ndarray, farm: str) -> None:
    required = {
        "farm", "split", "sequence_idx", "timestep_idx", "asset_id", "event_id",
        "window_end", "event_end", "fault_time", "hours_to_fault", "label", "mask",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Farm {farm} metadata missing columns: {sorted(missing)}")
    n = len(y)
    if len(metadata) != n * SEQ_LEN:
        raise ValueError(f"Farm {farm} metadata rows {len(metadata)} != {n} * 144")
    expected_sequence = np.repeat(np.arange(n), SEQ_LEN)
    expected_timestep = np.tile(np.arange(SEQ_LEN), n)
    if not np.array_equal(metadata["sequence_idx"].to_numpy(), expected_sequence):
        raise ValueError(f"Farm {farm} metadata sequence order is not array order")
    if not np.array_equal(metadata["timestep_idx"].to_numpy(), expected_timestep):
        raise ValueError(f"Farm {farm} timestep_idx is not 0..143 per sequence")
    timestamps = pd.to_datetime(metadata["window_end"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Farm {farm} metadata contains invalid timestamps")
    diffs = timestamps.groupby(metadata["sequence_idx"]).diff().dropna()
    if not (diffs == pd.Timedelta(minutes=10)).all():
        raise ValueError(f"Farm {farm} timestamps are not monotonic 10-minute timesteps")
    flat_y = y.reshape(-1).astype(np.uint8)
    flat_mask = mask.reshape(-1).astype(np.uint8)
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.uint8), flat_y):
        raise ValueError(f"Farm {farm} metadata labels do not align with y")
    if not np.array_equal(metadata["mask"].to_numpy(dtype=np.uint8), flat_mask):
        raise ValueError(f"Farm {farm} metadata masks do not align with mask array")
    fault_time = pd.to_datetime(metadata["fault_time"], errors="coerce")
    hours = (fault_time - timestamps).dt.total_seconds() / 3600.0
    expected_y = ((hours > 0) & (hours <= HORIZON_HOURS)).astype(np.uint8).to_numpy()
    if not np.array_equal(flat_y, expected_y):
        raise ValueError(f"Farm {farm} labels do not match the 48-hour fault horizon")


def validate_split(directory: Path, split: str, expected_f: int, farm: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(directory / f"{split}_X.npy", mmap_mode="r")
    y = np.load(directory / f"{split}_y.npy")
    mask = np.load(directory / f"{split}_mask.npy")
    if x.ndim != 3 or x.shape[1:] != (SEQ_LEN, expected_f):
        raise ValueError(f"Farm {farm} {split} X shape {x.shape}; expected (N,144,{expected_f})")
    if y.shape != x.shape[:2] or mask.shape != x.shape[:2]:
        raise ValueError(f"Farm {farm} {split} y/mask shapes do not match X")
    if not np.isfinite(x).all():
        raise ValueError(f"Farm {farm} {split} X contains NaN/Inf")
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(f"Farm {farm} {split} y/mask are not binary")
    metadata = pd.read_parquet(directory / f"{split}_metadata.parquet")
    validate_metadata(metadata, y, mask, farm)
    return np.asarray(x), mask


def validate_summary(directory: Path, farm: str, expected_order: list[str], scaler_hash: str) -> dict:
    summary = json.loads((directory / "export_summary.json").read_text(encoding="utf-8"))
    if summary.get("farm") != farm or summary.get("feature_order") != expected_order:
        raise ValueError(f"Farm {farm} summary feature/farm contract mismatch")
    if summary.get("num_features") != len(expected_order) or not summary.get("strict_schema"):
        raise ValueError(f"Farm {farm} is not a strict {len(expected_order)}-feature export")
    if summary.get("mask_policy") != "observed_timestamp_and_all_physical_features_valid":
        raise ValueError(f"Farm {farm} does not attest the physical validity mask policy")
    normalization = summary.get("normalization", {})
    if normalization.get("source_farm") != "C" or normalization.get("source_split") != "train":
        raise ValueError(f"Farm {farm} scaler provenance is not Farm C train")
    if normalization.get("scaler_sha256") != scaler_hash:
        raise ValueError(f"Farm {farm} did not use the frozen Farm C scaler")
    leakage = summary.get("leakage_contract", {})
    if leakage.get("external_farms_affect_scaler") is not False:
        raise ValueError(f"Farm {farm} allows external data into scaler fitting")
    if leakage.get("external_farms_affect_threshold") is not False:
        raise ValueError(f"Farm {farm} allows external data into threshold selection")
    if leakage.get("feature_selection_source") != "committed physical-v1 mapping":
        raise ValueError(f"Farm {farm} feature selection provenance is not fixed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-c-dir", required=True, type=Path)
    parser.add_argument("--farm-a-dir", type=Path)
    parser.add_argument("--farm-b-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/physical_feature_manifest.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/physical_sensor_mapping.yaml"))
    parser.add_argument("--split-config", required=True, type=Path)
    args = parser.parse_args()
    config = load_mapping(args.config)
    expected_order = list(config["strict_feature_order"])
    manifest = pd.read_csv(args.manifest).sort_values("feature_index")
    manifest_order = manifest["feature_name"].tolist()
    if manifest_order != expected_order or len(manifest) != 10:
        raise ValueError("Committed manifest does not match strict physical-v1 order")
    scaler_path = args.farm_c_dir / "scaler_stats.npz"
    scaler_hash = sha256_file(scaler_path)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        if scaler["feature_names"].astype(str).tolist() != expected_order:
            raise ValueError("Farm C scaler feature order differs from the manifest")
        if str(scaler["source_farm"].item()) != "C" or str(scaler["source_split"].item()) != "train":
            raise ValueError("Farm C scaler provenance is not train-only")
    c_arrays = {}
    for split in ("train", "val", "test"):
        c_arrays[split] = validate_split(args.farm_c_dir, split, len(expected_order), "C")
    c_summary = validate_summary(args.farm_c_dir, "C", expected_order, scaler_hash)
    train_x, train_mask = c_arrays["train"]
    valid_train = train_x.reshape(-1, len(expected_order))[train_mask.reshape(-1).astype(bool)]
    if not np.allclose(valid_train.mean(axis=0), 0.0, atol=2e-4):
        raise ValueError("Normalized Farm C train means do not confirm train-fitted scaling")
    varying = valid_train.std(axis=0) > 1e-5
    if not np.allclose(valid_train.std(axis=0)[varying], 1.0, atol=2e-4):
        raise ValueError("Normalized Farm C train standard deviations are not one")
    asset_sets = {split: set(c_summary["splits"][split]["assets"]) for split in ("train", "val", "test")}
    if asset_sets["train"] & asset_sets["val"] or asset_sets["train"] & asset_sets["test"] or asset_sets["val"] & asset_sets["test"]:
        raise ValueError("Farm C train/val/test assets overlap")
    frozen_split = json.loads(args.split_config.read_text(encoding="utf-8"))
    for split in asset_sets:
        if asset_sets[split] != set(map(str, frozen_split["splits"][split]["assets"])):
            raise ValueError(f"Farm C exported {split} assets differ from frozen split")
    for farm, directory in (("A", args.farm_a_dir), ("B", args.farm_b_dir)):
        if directory is None:
            continue
        validate_split(directory, "test", len(expected_order), farm)
        validate_summary(directory, farm, expected_order, scaler_hash)
    print("ALL STRICT PHYSICAL-V1 EXPORT CHECKS PASSED")


if __name__ == "__main__":
    main()
