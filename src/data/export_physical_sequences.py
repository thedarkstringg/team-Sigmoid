"""Generic CARE Farm A/B/C physical sequence exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.data.physical_features import (
        PowerCurve, compute_physical_features, feature_order, fit_binned_power_curve,
        load_mapping, rename_to_sensor_ids, required_sensor_ids, resolve_average_columns,
    )
    from src.data.sequence_utils import (
        HORIZON_HOURS, INTERVAL_MINUTES, LOOKBACK_HOURS, MAX_GAP_MINUTES,
        MIN_COVERAGE, SEQ_LEN, STRIDE_HOURS, apply_scaler, build_timestep_metadata,
        fit_train_scaler, load_passing_frequency_audit, load_scaler, save_scaler,
        read_care_csv, resolve_event_boundaries, sha256_file, timestep_labels,
    )
except ModuleNotFoundError:  # Support direct script invocation from repo root.
    from physical_features import (
        PowerCurve, compute_physical_features, feature_order, fit_binned_power_curve,
        load_mapping, rename_to_sensor_ids, required_sensor_ids, resolve_average_columns,
    )
    from sequence_utils import (
        HORIZON_HOURS, INTERVAL_MINUTES, LOOKBACK_HOURS, MAX_GAP_MINUTES,
        MIN_COVERAGE, SEQ_LEN, STRIDE_HOURS, apply_scaler, build_timestep_metadata,
        fit_train_scaler, load_passing_frequency_audit, load_scaler, save_scaler,
        read_care_csv, resolve_event_boundaries, sha256_file, timestep_labels,
    )


def load_split_config(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("farm") != "C" or payload.get("schema_version") != "farm-c-asset-split-v1":
        raise ValueError("Expected a generated farm-c-asset-split-v1 configuration")
    mapping = {}
    for split in ("train", "val", "test"):
        assets = payload.get("splits", {}).get(split, {}).get("assets", [])
        if len(assets) < (2 if split in {"val", "test"} else 1):
            raise ValueError(f"Farm C {split} split has too few assets: {assets}")
        for asset in assets:
            key = str(asset)
            if key in mapping:
                raise ValueError(f"Asset {asset} appears in multiple Farm C splits")
            mapping[key] = split
    return mapping, payload


def read_event_raw(
    path: Path, sensor_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    header = read_care_csv(path, nrows=0).columns.tolist()
    resolved = resolve_average_columns(header, sensor_ids)
    required = ["time_stamp", "asset_id"]
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"{path} misses required columns: {sorted(missing)}")
    frame = read_care_csv(path, usecols=required + list(resolved.values()))
    frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], errors="coerce")
    frame = (
        frame.dropna(subset=["time_stamp"])
        .sort_values("time_stamp")
        .drop_duplicates("time_stamp")
        .set_index("time_stamp")
    )
    logical = rename_to_sensor_ids(frame, resolved).apply(pd.to_numeric, errors="coerce")
    logical["asset_id"] = frame["asset_id"]
    return logical, resolved


def event_asset(frame: pd.DataFrame, event_id: int) -> Any:
    assets = frame["asset_id"].dropna().unique()
    if len(assets) != 1:
        raise ValueError(f"Event {event_id} does not contain exactly one asset: {assets}")
    return assets[0]


def read_asset_only(path: Path, event_id: int) -> Any:
    values = read_care_csv(path, usecols=["asset_id"])["asset_id"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Event {event_id} does not contain exactly one asset: {values}")
    return values[0]


def collect_power_curve_data(
    raw_dir: Path,
    event_info: pd.DataFrame,
    farm: str,
    config: dict,
    allowed_assets: set[str] | None,
) -> tuple[pd.Series, pd.Series, list[int]]:
    definition = config["optional_features"]["power_residual"]["farms"][farm]
    sensors = [definition["wind_speed"], definition["active_power"]]
    winds, powers, used_events = [], [], []
    normal = event_info[event_info["event_label"] == "normal"].sort_values("event_id")
    for row in normal.itertuples(index=False):
        path = raw_dir / f"comma_{int(row.event_id)}.csv"
        header = read_care_csv(path, nrows=0).columns.tolist()
        if "train_test" not in header:
            raise ValueError(
                f"{path} has no train_test column; refusing to fit a power curve "
                "without proving TRAIN-row scope"
            )
        resolved = resolve_average_columns(header, sensors)
        frame = read_care_csv(
            path,
            usecols=["asset_id", "train_test"] + list(resolved.values()),
        )
        asset_values = frame["asset_id"].dropna().unique()
        if len(asset_values) != 1:
            raise ValueError(f"Event {int(row.event_id)} does not contain exactly one asset")
        asset = str(asset_values[0])
        if allowed_assets is not None and asset not in allowed_assets:
            continue
        train_rows = frame["train_test"].astype(str).str.strip().str.lower().eq("train")
        if not train_rows.any():
            raise ValueError(f"Normal event {int(row.event_id)} has no train_test=train rows")
        logical = rename_to_sensor_ids(frame.loc[train_rows], resolved).apply(
            pd.to_numeric, errors="coerce"
        )
        winds.append(logical[definition["wind_speed"]])
        powers.append(logical[definition["active_power"]])
        used_events.append(int(row.event_id))
    if not winds:
        raise ValueError("No normal-event observations satisfy the power-curve fit policy")
    return pd.concat(winds, ignore_index=True), pd.concat(powers, ignore_index=True), used_events


def prepare_power_curve(
    args: argparse.Namespace,
    config: dict,
    event_info: pd.DataFrame,
    split_assets: dict[str, str],
) -> PowerCurve | None:
    if not args.include_power_residual:
        return None
    if args.power_curve:
        curve = PowerCurve.load(args.power_curve)
        if args.farm in {"A", "B"} and not args.allow_target_power_calibration:
            raise ValueError(
                "Using any Farm A/B power curve requires --allow-target-power-calibration; "
                "this is not zero-shot DG"
            )
        if curve.provenance.get("farm") != args.farm:
            raise ValueError("Power-curve provenance farm does not match the export farm")
        if args.farm == "C" and curve.provenance.get("setting") != "source-domain-train-normal-only":
            raise ValueError("Farm C power curve must attest train-normal-only fitting")
        return curve
    if args.farm == "C":
        allowed = {asset for asset, split in split_assets.items() if split == "train"}
        setting = "source-domain-train-normal-only"
    else:
        if not args.allow_target_power_calibration:
            raise ValueError(
                "Farm A/B power residual requires an existing approved curve or "
                "--allow-target-power-calibration; target calibration is not zero-shot DG"
            )
        allowed = None
        setting = "calibrated-target-domain-not-zero-shot"
    wind, power, events = collect_power_curve_data(
        args.raw_dir, event_info, args.farm, config, allowed
    )
    curve = fit_binned_power_curve(
        wind,
        power,
        provenance={
            "farm": args.farm,
            "setting": setting,
            "event_scope": "normal events only",
            "event_ids": events,
            "row_scope": "train_test=train",
            "split_scope": "source train assets" if args.farm == "C" else "target calibration",
        },
    )
    curve.save(args.output_dir / "power_curve.json")
    return curve


def build_event_sequences(
    *,
    raw_path: Path,
    event_row: Any,
    farm: str,
    split: str,
    config: dict,
    sensor_ids: list[str],
    power_curve: PowerCurve | None,
) -> list[dict[str, Any]]:
    event_id = int(event_row.event_id)
    event_label = str(event_row.event_label)
    boundaries = resolve_event_boundaries(raw_path, event_row, farm)
    event_start = boundaries.event_start
    event_end = boundaries.event_end
    raw, _ = read_event_raw(raw_path, sensor_ids)
    asset_id = event_asset(raw, event_id)
    physical, feature_valid = compute_physical_features(
        raw.drop(columns="asset_id"), farm, config, frequency_validated=True, power_curve=power_curve
    )
    final_candidate = event_end - pd.Timedelta(minutes=INTERVAL_MINUTES)
    candidates = pd.date_range(event_start, final_candidate, freq=f"{STRIDE_HOURS}h")
    sequences = []
    rejected = {"endpoint": 0, "coverage": 0, "gap": 0, "no_valid_timestep": 0}
    for window_end in candidates:
        if window_end not in raw.index:
            rejected["endpoint"] += 1
            continue
        window_start = window_end - pd.Timedelta(hours=LOOKBACK_HOURS) + pd.Timedelta(minutes=INTERVAL_MINUTES)
        regular_index = pd.date_range(window_start, window_end, freq=f"{INTERVAL_MINUTES}min")
        if len(regular_index) != SEQ_LEN:
            raise RuntimeError(f"Expected {SEQ_LEN} timestamps, got {len(regular_index)}")
        observed_index = raw.index[(raw.index >= window_start) & (raw.index <= window_end)]
        coverage = len(observed_index) / SEQ_LEN
        if coverage < MIN_COVERAGE:
            rejected["coverage"] += 1
            continue
        diffs = observed_index.to_series().diff().dropna()
        max_gap = diffs.max().total_seconds() / 60.0 if len(diffs) else 0.0
        if max_gap > MAX_GAP_MINUTES:
            rejected["gap"] += 1
            continue
        aligned = physical.reindex(regular_index)
        aligned_valid = feature_valid.reindex(regular_index, fill_value=False)
        mask = (aligned.notna().all(axis=1) & aligned_valid.all(axis=1)).astype(np.uint8).to_numpy()
        if not mask.any():
            rejected["no_valid_timestep"] += 1
            continue
        # At most two missing 10-minute rows can exist inside an approved
        # <=30-minute timestamp gap. Longer physical-invalid runs remain NaN
        # here and are later replaced by the Farm-C-train mean (scaled zero).
        filled = aligned.ffill(limit=2).bfill(limit=2)
        labels = timestep_labels(regular_index, event_label, event_end)
        sequences.append(
            {
                "X": filled.to_numpy(dtype=np.float32),
                "y": labels,
                "mask": mask,
                "asset_id": asset_id,
                "event_id": event_id,
                "event_label": event_label,
                "event_end": event_end,
                "index": regular_index,
                "split": split,
            }
        )
    print(
        f"Farm {farm} event {event_id:>3} asset={asset_id} split={split} "
        f"kept={len(sequences)} rejected={rejected}"
    )
    return sequences


def save_outputs(
    grouped: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    names: list[str],
    split_payload: dict[str, Any] | None,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split, items in grouped.items():
        if not items:
            raise RuntimeError(f"No sequences generated for {split}")
        arrays[f"{split}_X"] = np.stack([item["X"] for item in items]).astype(np.float32)
        arrays[f"{split}_y"] = np.stack([item["y"] for item in items]).astype(np.uint8)
        arrays[f"{split}_mask"] = np.stack([item["mask"] for item in items]).astype(np.uint8)
        frames = []
        for sequence_idx, item in enumerate(items):
            frames.append(
                build_timestep_metadata(
                    farm=args.farm,
                    split=split,
                    sequence_idx=sequence_idx,
                    index=item["index"],
                    asset_id=item["asset_id"],
                    event_id=item["event_id"],
                    event_label=item["event_label"],
                    event_end=item["event_end"],
                    labels=item["y"],
                    mask=item["mask"],
                )
            )
        metadata[split] = pd.concat(frames, ignore_index=True)
    if args.farm == "C":
        mean, std = fit_train_scaler(arrays["train_X"], arrays["train_mask"])
        save_scaler(args.output_dir / "scaler_stats.npz", mean, std, names)
        scaler_path = args.output_dir / "scaler_stats.npz"
    else:
        if args.scaler is None:
            raise ValueError("Farm A/B external export requires --scaler from Farm C train")
        mean, std = load_scaler(args.scaler, names)
        scaler_path = args.scaler
    for split in grouped:
        values = arrays[f"{split}_X"]
        valid_rows = arrays[f"{split}_mask"].astype(bool)
        finite = np.isfinite(values)
        if ((~finite) & valid_rows[..., None]).any():
            raise ValueError(f"Farm {args.farm} {split} has non-finite values at valid timesteps")
        values = np.where(finite, values, mean.reshape(1, 1, -1))
        arrays[f"{split}_X"] = apply_scaler(values, mean, std)
        np.save(args.output_dir / f"{split}_X.npy", arrays[f"{split}_X"])
        np.save(args.output_dir / f"{split}_y.npy", arrays[f"{split}_y"])
        np.save(args.output_dir / f"{split}_mask.npy", arrays[f"{split}_mask"])
        if len(metadata[split]) != len(arrays[f"{split}_X"]) * SEQ_LEN:
            raise RuntimeError("Metadata row count is not N * 144")
        metadata[split].to_parquet(
            args.output_dir / f"{split}_metadata.parquet", index=False, compression="zstd"
        )
    summary = {
        "export_version": "physical-sequences-v1",
        "farm": args.farm,
        "feature_order": names,
        "num_features": len(names),
        "strict_schema": not args.include_power_residual,
        "temporal_contract": {
            "interval_minutes": INTERVAL_MINUTES,
            "lookback_hours": LOOKBACK_HOURS,
            "sequence_length": SEQ_LEN,
            "stride_hours": STRIDE_HOURS,
            "fault_horizon_hours": HORIZON_HOURS,
            "minimum_coverage": MIN_COVERAGE,
            "maximum_gap_minutes": MAX_GAP_MINUTES,
        },
        "mask_policy": "observed_timestamp_and_all_physical_features_valid",
        "normalization": {
            "type": "z-score",
            "source_farm": "C",
            "source_split": "train",
            "fit_timesteps": "real_and_physical_valid_only",
            "scaler_sha256": sha256_file(scaler_path),
        },
        "leakage_contract": {
            "feature_selection_source": "committed physical-v1 mapping",
            "external_farms_affect_scaler": False,
            "external_farms_affect_threshold": False,
        },
        "splits": {
            split: {
                "num_sequences": len(items),
                "assets": sorted({str(item["asset_id"]) for item in items}),
                "events": sorted({int(item["event_id"]) for item in items}),
                "valid_timesteps": int(arrays[f"{split}_mask"].sum()),
            }
            for split, items in grouped.items()
        },
        "farm_c_split": split_payload,
        "power_residual_setting": (
            "disabled-strict-zero-shot-dg"
            if not args.include_power_residual
            else "see power_curve.json provenance"
        ),
    }
    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/physical_sensor_mapping.yaml"))
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--split-config", type=Path, help="Required generated JSON for Farm C")
    parser.add_argument("--scaler", type=Path, help="Required Farm-C-train scaler for Farm A/B")
    parser.add_argument("--include-power-residual", action="store_true")
    parser.add_argument("--power-curve", type=Path)
    parser.add_argument("--allow-target-power-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_mapping(args.config)
    load_passing_frequency_audit(args.audit_report, args.farm)
    event_info = read_care_csv(args.raw_dir / "comma_event_info.csv")
    if args.farm in {"A", "B"}:
        event_info["event_start"] = pd.to_datetime(event_info["event_start"])
        event_info["event_end"] = pd.to_datetime(event_info["event_end"])
    split_payload = None
    if args.farm == "C":
        if args.split_config is None:
            raise ValueError("Farm C export requires --split-config generated from raw data")
        split_assets, split_payload = load_split_config(args.split_config)
    else:
        if args.split_config is not None:
            raise ValueError("External Farm A/B export must not use a source-domain split config")
        split_assets = {}
    curve = prepare_power_curve(args, config, event_info, split_assets)
    names = feature_order(config, include_power_residual=curve is not None)
    sensors = required_sensor_ids(config, args.farm, include_power_residual=curve is not None)
    grouped: dict[str, list[dict[str, Any]]] = (
        {"train": [], "val": [], "test": []} if args.farm == "C" else {"test": []}
    )
    for event_row in event_info.sort_values("event_id").itertuples(index=False):
        path = args.raw_dir / f"comma_{int(event_row.event_id)}.csv"
        asset = str(read_asset_only(path, int(event_row.event_id)))
        split = split_assets.get(asset) if args.farm == "C" else "test"
        if split is None:
            raise ValueError(f"Farm C asset {asset} is absent from split config")
        grouped[split].extend(
            build_event_sequences(
                raw_path=path,
                event_row=event_row,
                farm=args.farm,
                split=split,
                config=config,
                sensor_ids=sensors,
                power_curve=curve,
            )
        )
    save_outputs(grouped, args, names, split_payload)
    print(f"Saved Farm {args.farm} physical sequences to {args.output_dir}")


if __name__ == "__main__":
    main()
