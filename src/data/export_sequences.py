from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


LOOKBACK_HOURS = 24
HORIZON_HOURS = 48
STRIDE_HOURS = 1

INTERVAL_MINUTES = 10
SEQ_LEN = 144
MIN_COVERAGE = 0.95
MAX_GAP_MINUTES = 30

ASSET_SPLIT = {
    0: "train",
    10: "train",
    11: "train",
    21: "val",
    13: "test",
}


def get_sensor_columns(meta, available_columns):
    """Keeps exactly one average signal for each Farm A sensor."""
    columns = []

    for sensor in meta["sensor_name"]:
        avg_name = f"{sensor}_avg"

        if avg_name in available_columns:
            columns.append(avg_name)
        elif sensor in available_columns:
            columns.append(sensor)
        else:
            raise ValueError(f"Missing sensor column: {sensor}")

    if len(columns) != 54:
        raise ValueError(f"Expected 54 sensor channels, found {len(columns)}")

    return columns


def timestep_labels(index, event_label, event_end):
    """Marks each real timestep positive when failure is within 48 hours."""
    if event_label != "anomaly":
        return np.zeros(len(index), dtype=np.uint8)

    hours_to_fault = (
        event_end - index
    ).total_seconds() / 3600.0

    return (
        (hours_to_fault > 0)
        & (hours_to_fault <= HORIZON_HOURS)
    ).astype(np.uint8)


def build_event_sequences(path, event_row, sensor_cols):
    """Builds 24-hour tensors while masking small SCADA gaps."""
    event_id = int(event_row["event_id"])
    event_label = event_row["event_label"]
    event_start = pd.Timestamp(event_row["event_start"])
    event_end = pd.Timestamp(event_row["event_end"])

    usecols = [
        "time_stamp",
        "asset_id",
    ] + sensor_cols

    df = pd.read_csv(path, usecols=usecols)

    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["time_stamp"])
        .sort_values("time_stamp")
        .drop_duplicates("time_stamp")
        .set_index("time_stamp")
    )

    asset_ids = df["asset_id"].dropna().unique()

    if len(asset_ids) != 1:
        raise ValueError(
            f"Event {event_id} has unexpected assets: {asset_ids}"
        )

    asset_id = int(asset_ids[0])
    split = ASSET_SPLIT[asset_id]

    final_candidate = (
        event_end - pd.Timedelta(minutes=INTERVAL_MINUTES)
    )

    candidate_times = pd.date_range(
        event_start,
        final_candidate,
        freq=f"{STRIDE_HOURS}h",
    )

    sequences = []

    rejected_endpoint = 0
    rejected_coverage = 0
    rejected_gap = 0

    for window_end in candidate_times:
        if window_end not in df.index:
            rejected_endpoint += 1
            continue

        window_start = (
            window_end
            - pd.Timedelta(hours=LOOKBACK_HOURS)
            + pd.Timedelta(minutes=INTERVAL_MINUTES)
        )

        regular_index = pd.date_range(
            window_start,
            window_end,
            freq=f"{INTERVAL_MINUTES}min",
        )

        if len(regular_index) != SEQ_LEN:
            raise RuntimeError(
                f"Expected {SEQ_LEN} timestamps, got {len(regular_index)}"
            )

        observed = df.loc[
            (df.index >= window_start)
            & (df.index <= window_end)
        ]

        coverage = len(observed) / SEQ_LEN

        if coverage < MIN_COVERAGE:
            rejected_coverage += 1
            continue

        diffs = observed.index.to_series().diff().dropna()

        if len(diffs):
            max_gap = diffs.max().total_seconds() / 60.0
        else:
            max_gap = 0.0

        if max_gap > MAX_GAP_MINUTES:
            rejected_gap += 1
            continue

        aligned = observed[sensor_cols].reindex(regular_index)

        mask = (
            aligned.notna()
            .all(axis=1)
            .astype(np.uint8)
            .to_numpy()
        )

        # Fill only inside already-approved short gaps.
        aligned = aligned.ffill().bfill()

        if aligned.isna().any().any():
            raise ValueError(
                f"Unfilled NaN remained in event {event_id}"
            )

        x = aligned.to_numpy(dtype=np.float32)

        y = timestep_labels(
            regular_index,
            event_label,
            event_end,
        )

        sequences.append(
            {
                "X": x,
                "y": y,
                "mask": mask,
                "event_id": event_id,
                "asset_id": asset_id,
                "split": split,
            }
        )

    print(
        f"Event {event_id:>2} | "
        f"asset={asset_id:>2} | "
        f"{event_label:7s} | "
        f"split={split:5s} | "
        f"kept={len(sequences):>4} | "
        f"missing_end={rejected_endpoint:>3} | "
        f"coverage={rejected_coverage:>3} | "
        f"gap={rejected_gap:>3}"
    )

    return sequences


def fit_train_scaler(train_x, train_mask):
    """Fits per-sensor mean/std from real training timesteps only."""
    flat_x = train_x.reshape(-1, train_x.shape[-1])
    flat_mask = train_mask.reshape(-1).astype(bool)

    real_values = flat_x[flat_mask]

    mean = real_values.mean(axis=0).astype(np.float32)
    std = real_values.std(axis=0).astype(np.float32)

    std[std < 1e-6] = 1.0

    return mean, std


def apply_scaler(x, mean, std):
    """Applies train-only normalization without changing tensor shape."""
    return ((x - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    event_info = pd.read_csv(
        raw_dir / "comma_event_info.csv"
    )

    event_info["event_start"] = pd.to_datetime(
        event_info["event_start"]
    )

    event_info["event_end"] = pd.to_datetime(
        event_info["event_end"]
    )

    feature_meta = pd.read_csv(
        raw_dir / "comma_feature_description.csv"
    )

    sample_path = raw_dir / "comma_0.csv"

    available_columns = pd.read_csv(
        sample_path,
        nrows=0,
    ).columns.tolist()

    sensor_cols = get_sensor_columns(
        feature_meta,
        available_columns,
    )

    grouped = {
        "train": [],
        "val": [],
        "test": [],
    }

    print("=== EXPORTING FARM A SEQUENCES ===")
    print(f"Sensors: {len(sensor_cols)}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Lookback: {LOOKBACK_HOURS}h")
    print(f"Horizon: {HORIZON_HOURS}h")
    print()

    for _, event_row in (
        event_info.sort_values("event_id").iterrows()
    ):
        event_id = int(event_row["event_id"])
        path = raw_dir / f"comma_{event_id}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        sequences = build_event_sequences(
            path,
            event_row,
            sensor_cols,
        )

        for item in sequences:
            grouped[item["split"]].append(item)

    arrays = {}

    for split, items in grouped.items():
        if not items:
            raise RuntimeError(f"No sequences for split: {split}")

        arrays[f"{split}_X"] = np.stack(
            [item["X"] for item in items]
        ).astype(np.float32)

        arrays[f"{split}_y"] = np.stack(
            [item["y"] for item in items]
        ).astype(np.uint8)

        arrays[f"{split}_mask"] = np.stack(
            [item["mask"] for item in items]
        ).astype(np.uint8)

    mean, std = fit_train_scaler(
        arrays["train_X"],
        arrays["train_mask"],
    )

    for split in ("train", "val", "test"):
        arrays[f"{split}_X"] = apply_scaler(
            arrays[f"{split}_X"],
            mean,
            std,
        )

    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)

    np.savez(
        output_dir / "scaler_stats.npz",
        mean=mean,
        std=std,
        sensor_columns=np.array(sensor_cols),
    )

    split_metadata = {}

    for split, items in grouped.items():
        split_metadata[split] = {
            "num_sequences": len(items),
            "assets": sorted(
                {int(item["asset_id"]) for item in items}
            ),
            "events": sorted(
                {int(item["event_id"]) for item in items}
            ),
            "positive_timesteps": int(
                arrays[f"{split}_y"][
                    arrays[f"{split}_mask"].astype(bool)
                ].sum()
            ),
            "real_timesteps": int(
                arrays[f"{split}_mask"].sum()
            ),
        }

    summary = {
        "lookback_hours": LOOKBACK_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "stride_hours": STRIDE_HOURS,
        "sequence_length": SEQ_LEN,
        "num_features": len(sensor_cols),
        "normalization": "per-feature z-score fitted on real train timesteps only",
        "asset_split": ASSET_SPLIT,
        "splits": split_metadata,
    }

    with open(
        output_dir / "export_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== SAVED ARRAYS ===")

    for split in ("train", "val", "test"):
        print(
            split,
            arrays[f"{split}_X"].shape,
            arrays[f"{split}_X"].dtype,
            arrays[f"{split}_y"].shape,
            arrays[f"{split}_y"].dtype,
            arrays[f"{split}_mask"].shape,
            arrays[f"{split}_mask"].dtype,
        )

    print("\nOutput:", output_dir)


if __name__ == "__main__":
    main()
