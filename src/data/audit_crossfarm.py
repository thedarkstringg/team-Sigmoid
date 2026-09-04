"""Audit selected physical-v1 raw channels on a server with CARE data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.data.physical_features import (
        compute_physical_features,
        load_mapping,
        rename_to_sensor_ids,
        required_sensor_ids,
        resolve_average_columns,
    )
except ModuleNotFoundError:  # Support direct script invocation from repo root.
    from physical_features import (
        compute_physical_features,
        load_mapping,
        rename_to_sensor_ids,
        required_sensor_ids,
        resolve_average_columns,
    )


def max_zero_run(values: pd.Series) -> int:
    zero = values.eq(0).fillna(False).to_numpy()
    best = current = 0
    for item in zero:
        current = current + 1 if item else 0
        best = max(best, current)
    return int(best)


def safe_number(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/physical_sensor_mapping.yaml"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zero-run-threshold", type=int, default=6)
    parser.add_argument("--sample-per-event", type=int, default=2000)
    args = parser.parse_args()
    config = load_mapping(args.config)
    sensors = required_sensor_ids(config, args.farm)
    event_info = pd.read_csv(args.raw_dir / "comma_event_info.csv")
    raw_stats: dict[str, dict[str, Any]] = {
        sensor: {"count": 0, "missing": 0, "nonfinite": 0, "zeros": 0, "negative": 0, "min": np.inf, "max": -np.inf, "samples": []}
        for sensor in sensors
    }
    continuity = []
    zero_runs = []
    feature_valid = defaultdict(lambda: [0, 0])
    feature_samples = defaultdict(list)
    frequency_samples: list[np.ndarray] = []
    frequency_outside_plausible = 0
    availability_failures = []
    for event_id in sorted(event_info["event_id"].astype(int)):
        path = args.raw_dir / f"comma_{event_id}.csv"
        if not path.exists():
            availability_failures.append({"event_id": event_id, "reason": "missing event file"})
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        try:
            resolved = resolve_average_columns(header, sensors)
        except ValueError as exc:
            availability_failures.append({"event_id": event_id, "reason": str(exc)})
            continue
        usecols = ["time_stamp"] + (["asset_id"] if "asset_id" in header else []) + list(resolved.values())
        frame = pd.read_csv(path, usecols=usecols)
        timestamp = pd.to_datetime(frame["time_stamp"], errors="coerce")
        valid_time = timestamp.dropna().sort_values()
        diffs = valid_time.diff().dropna()
        continuity.append(
            {
                "event_id": event_id,
                "rows": len(frame),
                "invalid_timestamps": int(timestamp.isna().sum()),
                "duplicate_timestamps": int(timestamp.duplicated().sum()),
                "gaps_over_10m": int((diffs > pd.Timedelta(minutes=10)).sum()),
                "max_gap_minutes": safe_number(diffs.max().total_seconds() / 60.0) if len(diffs) else 0.0,
            }
        )
        logical = rename_to_sensor_ids(frame, resolved).apply(pd.to_numeric, errors="coerce")
        for sensor in sensors:
            series = logical[sensor]
            numeric = series.to_numpy(dtype=float)
            finite = numeric[np.isfinite(numeric)]
            stats = raw_stats[sensor]
            stats["count"] += len(series)
            stats["missing"] += int(series.isna().sum())
            stats["nonfinite"] += int(np.isinf(numeric).sum())
            stats["zeros"] += int(np.sum(finite == 0))
            stats["negative"] += int(np.sum(finite < 0))
            if len(finite):
                stats["min"] = min(stats["min"], float(finite.min()))
                stats["max"] = max(stats["max"], float(finite.max()))
                step = max(1, len(finite) // args.sample_per_event)
                stats["samples"].extend(finite[::step][: args.sample_per_event].tolist())
            run = max_zero_run(series)
            if run >= args.zero_run_threshold:
                zero_runs.append({"event_id": event_id, "sensor": sensor, "max_consecutive_zeros": run})
        feature_frame, validity = compute_physical_features(
            logical, args.farm, config, frequency_validated=True
        )
        for name in feature_frame.columns:
            feature_valid[name][0] += int(validity[name].sum())
            feature_valid[name][1] += len(validity)
            finite = feature_frame.loc[validity[name], name].to_numpy(dtype=float)
            if len(finite):
                step = max(1, len(finite) // args.sample_per_event)
                feature_samples[name].extend(finite[::step][: args.sample_per_event].tolist())
        frequency_sensor = config["features"]["grid_frequency_deviation_Hz"]["farms"][args.farm]["grid_frequency"]
        frequency = logical[frequency_sensor].to_numpy(dtype=float)
        frequency_outside_plausible += int(
            np.sum(np.isfinite(frequency) & ((frequency < 45) | (frequency > 55)))
        )
        frequency_samples.append(frequency[np.isfinite(frequency) & (frequency >= 45) & (frequency <= 55)])

    raw_report = {}
    for sensor, stats in raw_stats.items():
        sample = np.asarray(stats.pop("samples"), dtype=float)
        raw_report[sensor] = {
            **stats,
            "min": safe_number(stats["min"]),
            "max": safe_number(stats["max"]),
            "constant": bool(np.isfinite(stats["min"]) and stats["min"] == stats["max"]),
            "median_sample": safe_number(np.median(sample)) if len(sample) else None,
            "q01_sample": safe_number(np.quantile(sample, 0.01)) if len(sample) else None,
            "q99_sample": safe_number(np.quantile(sample, 0.99)) if len(sample) else None,
        }
    derived_report = {}
    for name in config["strict_feature_order"]:
        sample = np.asarray(feature_samples[name], dtype=float)
        valid_count, total = feature_valid[name]
        derived_report[name] = {
            "valid_count": valid_count,
            "total_count": total,
            "valid_fraction": valid_count / total if total else 0.0,
            "min_sample": safe_number(sample.min()) if len(sample) else None,
            "median_sample": safe_number(np.median(sample)) if len(sample) else None,
            "max_sample": safe_number(sample.max()) if len(sample) else None,
        }
    all_frequency = np.concatenate(frequency_samples) if frequency_samples else np.asarray([])
    median_frequency = float(np.median(all_frequency)) if len(all_frequency) else np.nan
    expected = float(config["global_validity"]["expected_grid_frequency_hz"])
    tolerance = float(config["global_validity"]["allowed_frequency_median_error_hz"])
    frequency_passed = bool(np.isfinite(median_frequency) and abs(median_frequency - expected) <= tolerance)
    wind_sensor = config["features"]["wind_speed_mps"]["farms"][args.farm]["wind_speed"]
    report = {
        "audit_version": "crossfarm-physical-v1",
        "farm": args.farm,
        "raw_dir": str(args.raw_dir.resolve()),
        "selected_statistic": "average",
        "selected_sensor_count": len(sensors),
        "events_expected": len(event_info),
        "events_audited": len(continuity),
        "feature_availability_failures": availability_failures,
        "timestamp_continuity": continuity,
        "raw_channel_summary": raw_report,
        "suspicious_zero_runs": zero_runs,
        "derived_feature_summary": derived_report,
        "impossible_value_summary": {
            "negative_wind_speed_count": int(raw_report[wind_sensor]["negative"]),
            "power_factor_outside_0_1_sample_count": int(
                np.sum(
                    (np.asarray(feature_samples["grid_power_factor"], dtype=float) < 0)
                    | (np.asarray(feature_samples["grid_power_factor"], dtype=float) > 1 + 1e-9)
                )
            ),
            "frequency_outside_45_55_hz_count": frequency_outside_plausible,
            "nonfinite_raw_value_count": int(sum(item["nonfinite"] for item in raw_report.values())),
        },
        "operating_range_checks": {
            "wind_speed_nonnegative_enforced": True,
            "yaw_output_range_deg": [0, 180],
            "pitch_range_requires_human_review": True,
            "speed_ratio_minimum_rotor_rpm": config["global_validity"]["minimum_rotor_speed_rpm"],
        },
        "frequency_validation": {
            "expected_hz": expected,
            "plausible_observation_count": len(all_frequency),
            "median_hz": safe_number(median_frequency),
            "allowed_median_error_hz": tolerance,
            "passed": frequency_passed,
        },
        "strict_export_ready": not availability_failures and frequency_passed,
        "limitations": [
            "Zero is reported rather than globally treated as missing because several selected channels legitimately reach zero.",
            "Sample quantiles are deterministic diagnostics, not full-distribution estimates.",
            "Pitch and yaw sign/range conventions require human review before accepting the audit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "farm": args.farm,
        "events_audited": len(continuity),
        "availability_failures": len(availability_failures),
        "frequency_validation": report["frequency_validation"],
        "strict_export_ready": report["strict_export_ready"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
