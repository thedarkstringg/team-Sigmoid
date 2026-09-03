from pathlib import Path
import os
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "CARE_FARM_A_DIR",
        str(REPO_ROOT / "data" / "raw" / "CARE_Farm_A"),
    )
)

meta = pd.read_csv(ROOT / "comma_feature_description.csv")

dataset_files = sorted(
    [
        p for p in ROOT.glob("comma_*.csv")
        if p.name not in {
            "comma_event_info.csv",
            "comma_feature_description.csv"
        }
    ],
    key=lambda p: int(p.stem.replace("comma_", ""))
)

sample_cols = pd.read_csv(dataset_files[0], nrows=0).columns.tolist()

sensor_cols = []

for sensor in meta["sensor_name"]:
    avg_name = f"{sensor}_avg"

    if avg_name in sample_cols:
        sensor_cols.append(avg_name)
    elif sensor in sample_cols:
        sensor_cols.append(sensor)
    else:
        print("MISSING SENSOR COLUMN:", sensor)

print("\n=== FEATURE SELECTION ===")
print("Selected average sensor channels:", len(sensor_cols))

overall_missing = {c: [0, 0] for c in sensor_cols}

print("\n=== DATA QUALITY BY EVENT ===")

for path in dataset_files:
    event_id = int(path.stem.replace("comma_", ""))

    df = pd.read_csv(
        path,
        usecols=["time_stamp"] + sensor_cols
    )

    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"],
        errors="coerce"
    )

    df = df.sort_values("time_stamp")

    duplicate_ts = int(df["time_stamp"].duplicated().sum())

    diffs = df["time_stamp"].diff().dropna()
    expected = pd.Timedelta(minutes=10)

    large_gaps = diffs[diffs > expected]

    missing_pct = (
        df[sensor_cols]
        .isna()
        .mean()
        .mul(100)
    )

    for col in sensor_cols:
        overall_missing[col][0] += int(df[col].isna().sum())
        overall_missing[col][1] += len(df)

    max_gap = (
        large_gaps.max()
        if len(large_gaps)
        else pd.Timedelta(0)
    )

    print(
        f"Event {event_id:>2}: "
        f"rows={len(df):>6}, "
        f"duplicates={duplicate_ts:>3}, "
        f"gaps>10m={len(large_gaps):>4}, "
        f"max_gap={max_gap}, "
        f"worst_missing={missing_pct.max():.2f}%"
    )

print("\n=== OVERALL MISSINGNESS ===")

overall = []

for col, (missing, total) in overall_missing.items():
    pct = 100 * missing / total if total else 0
    overall.append((col, pct))

for col, pct in sorted(
    overall,
    key=lambda x: x[1],
    reverse=True
):
    print(f"{col:30s} {pct:8.3f}%")
