from pathlib import Path
import os

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.environ.get(
        "CARE_FARM_A_DIR",
        str(REPO_ROOT / "data" / "raw" / "CARE_Farm_A"),
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "CARE_FARM_A_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "processed" / "CARE_Farm_A"),
    )
)

LOOKBACK_HOURS = 24
HORIZON_HOURS = 48
STRIDE_HOURS = 1

EXPECTED_INTERVAL_MINUTES = 10
MIN_WINDOW_COVERAGE = 0.95
MAX_ALLOWED_GAP_MINUTES = 30

# Asset-disjoint split.
# No turbine appears in more than one split.
ASSET_SPLIT = {
    0: "train",
    10: "train",
    11: "train",
    21: "val",
    13: "test",
}


def get_sensor_columns(meta, available_columns):
    """Keeps one average signal per physical sensor."""
    columns = []

    for sensor in meta["sensor_name"]:
        avg_name = f"{sensor}_avg"

        if avg_name in available_columns:
            columns.append(avg_name)
        elif sensor in available_columns:
            columns.append(sensor)
        else:
            raise ValueError(f"Missing sensor column: {sensor}")

    return columns


def extract_features(window, sensor_cols):
    """Turns one 24-hour SCADA window into compact baseline features."""
    values = window[sensor_cols].copy()

    # Sensor-level NaNs are extremely rare in Farm A.
    # Fill only existing timestamp rows; missing timestamps are never created.
    values = values.ffill().bfill()

    features = {}

    for col in sensor_cols:
        series = values[col].astype(float)

        features[f"{col}__mean"] = series.mean()
        features[f"{col}__std"] = series.std(ddof=0)
        features[f"{col}__min"] = series.min()
        features[f"{col}__max"] = series.max()
        features[f"{col}__last"] = series.iloc[-1]
        features[f"{col}__change"] = series.iloc[-1] - series.iloc[0]

    return features


def build_event_windows(path, event_row, sensor_cols):
    """Builds valid hourly prediction samples without bridging broken SCADA periods."""
    event_id = int(event_row["event_id"])
    event_label = event_row["event_label"]

    usecols = [
        "time_stamp",
        "asset_id",
        "train_test",
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
            f"Event {event_id} has unexpected asset IDs: {asset_ids}"
        )

    asset_id = int(asset_ids[0])

    if asset_id not in ASSET_SPLIT:
        raise ValueError(f"No split configured for asset {asset_id}")

    split = ASSET_SPLIT[asset_id]

    event_start = pd.Timestamp(event_row["event_start"])
    event_end = pd.Timestamp(event_row["event_end"])

    # Do not create a sample exactly at fault onset.
    final_candidate = event_end - pd.Timedelta(
        minutes=EXPECTED_INTERVAL_MINUTES
    )

    candidate_times = pd.date_range(
        start=event_start,
        end=final_candidate,
        freq=f"{STRIDE_HOURS}h",
    )

    expected_points = int(
        LOOKBACK_HOURS * 60 / EXPECTED_INTERVAL_MINUTES
    )

    rows = []

    rejected_missing_endpoint = 0
    rejected_coverage = 0
    rejected_gap = 0

    for window_end in candidate_times:

        # An alert requires a real SCADA observation at its endpoint.
        if window_end not in df.index:
            rejected_missing_endpoint += 1
            continue

        window_start = (
            window_end
            - pd.Timedelta(hours=LOOKBACK_HOURS)
            + pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES)
        )

        window = df.loc[
            (df.index >= window_start)
            & (df.index <= window_end)
        ]

        coverage = len(window) / expected_points

        if coverage < MIN_WINDOW_COVERAGE:
            rejected_coverage += 1
            continue

        timestamp_diffs = (
            window.index
            .to_series()
            .diff()
            .dropna()
        )

        if len(timestamp_diffs):
            max_gap_minutes = (
                timestamp_diffs.max().total_seconds() / 60
            )
        else:
            max_gap_minutes = 0.0

        if max_gap_minutes > MAX_ALLOWED_GAP_MINUTES:
            rejected_gap += 1
            continue

        if event_label == "anomaly":
            hours_to_fault = (
                event_end - window_end
            ).total_seconds() / 3600

            label = int(
                0 < hours_to_fault <= HORIZON_HOURS
            )
        else:
            hours_to_fault = np.nan
            label = 0

        feature_row = extract_features(
            window,
            sensor_cols,
        )

        feature_row.update(
            {
                "event_id": event_id,
                "asset_id": asset_id,
                "split": split,
                "event_label": event_label,
                "window_end": window_end,
                "hours_to_fault": hours_to_fault,
                "window_coverage": coverage,
                "label": label,
            }
        )

        rows.append(feature_row)

    print(
        f"Event {event_id:>2} | "
        f"asset={asset_id:>2} | "
        f"{event_label:7s} | "
        f"split={split:5s} | "
        f"kept={len(rows):>4} | "
        f"missing_end={rejected_missing_endpoint:>3} | "
        f"coverage={rejected_coverage:>3} | "
        f"gap={rejected_gap:>3}"
    )

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_info = pd.read_csv(
        DATA_DIR / "comma_event_info.csv"
    )

    event_info["event_start"] = pd.to_datetime(
        event_info["event_start"]
    )

    event_info["event_end"] = pd.to_datetime(
        event_info["event_end"]
    )

    feature_meta = pd.read_csv(
        DATA_DIR / "comma_feature_description.csv"
    )

    sample_file = next(
        p
        for p in DATA_DIR.glob("comma_*.csv")
        if p.name not in {
            "comma_event_info.csv",
            "comma_feature_description.csv",
        }
    )

    available_columns = pd.read_csv(
        sample_file,
        nrows=0,
    ).columns.tolist()

    sensor_cols = get_sensor_columns(
        feature_meta,
        available_columns,
    )

    print("=== BUILDING FARM A WINDOWS ===")
    print(f"Sensor channels: {len(sensor_cols)}")
    print(f"Lookback: {LOOKBACK_HOURS} hours")
    print(f"Fault horizon: {HORIZON_HOURS} hours")
    print(f"Stride: {STRIDE_HOURS} hour")
    print()

    all_rows = []

    for _, event_row in event_info.sort_values(
        "event_id"
    ).iterrows():

        event_id = int(event_row["event_id"])

        path = DATA_DIR / f"comma_{event_id}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing event dataset: {path}"
            )

        rows = build_event_windows(
            path,
            event_row,
            sensor_cols,
        )

        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)

    if result.empty:
        raise RuntimeError("No valid windows were generated.")

    metadata_cols = [
        "event_id",
        "asset_id",
        "split",
        "event_label",
        "window_end",
        "hours_to_fault",
        "window_coverage",
        "label",
    ]

    feature_cols = [
        c for c in result.columns
        if c not in metadata_cols
    ]

    result = result[
        metadata_cols + feature_cols
    ]

    output_file = OUTPUT_DIR / "farm_a_windows.csv"

    result.to_csv(
        output_file,
        index=False,
    )

    split_table = pd.DataFrame(
        [
            {
                "asset_id": asset_id,
                "split": split,
            }
            for asset_id, split in ASSET_SPLIT.items()
        ]
    ).sort_values("asset_id")

    split_table.to_csv(
        OUTPUT_DIR / "splits.csv",
        index=False,
    )

    pd.DataFrame(
        {"feature": feature_cols}
    ).to_csv(
        OUTPUT_DIR / "feature_manifest.csv",
        index=False,
    )

    print("\n=== FINAL DATASET ===")
    print("Windows:", len(result))
    print("Features:", len(feature_cols))

    print("\nSplit counts:")
    print(result["split"].value_counts())

    print("\nClass counts by split:")
    print(
        pd.crosstab(
            result["split"],
            result["label"],
        )
    )

    print("\nEvents by split:")
    print(
        result.groupby("split")["event_id"]
        .nunique()
    )

    print("\nAssets by split:")
    print(
        result.groupby("split")["asset_id"]
        .unique()
    )

    print("\nSaved:")
    print(output_file)
    print(OUTPUT_DIR / "splits.csv")
    print(OUTPUT_DIR / "feature_manifest.csv")


if __name__ == "__main__":
    main()
