"""Prove and reconstruct metadata alignment for the legacy Farm A arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.data.sequence_utils import (
        INTERVAL_MINUTES, LOOKBACK_HOURS, MAX_GAP_MINUTES, MIN_COVERAGE,
        SEQ_LEN, STRIDE_HOURS, build_timestep_metadata, timestep_labels,
    )
except ModuleNotFoundError:  # Support direct script invocation from repo root.
    from sequence_utils import (
        INTERVAL_MINUTES, LOOKBACK_HOURS, MAX_GAP_MINUTES, MIN_COVERAGE,
        SEQ_LEN, STRIDE_HOURS, build_timestep_metadata, timestep_labels,
    )


ASSET_SPLIT = {0: "train", 10: "train", 11: "train", 21: "val", 13: "test"}
EXPECTED_COUNTS = {"train": 4357, "val": 1092, "test": 778}


def legacy_sensor_columns(meta: pd.DataFrame, available: list[str]) -> list[str]:
    result = []
    for sensor in meta["sensor_name"]:
        avg = f"{sensor}_avg"
        average = f"{sensor}_average"
        if avg in available:
            result.append(avg)
        elif average in available:
            result.append(average)
        elif sensor in available:
            result.append(sensor)
        else:
            raise ValueError(f"Missing legacy Avg sensor column: {sensor}")
    if len(result) != 54:
        raise ValueError(f"Expected 54 legacy Farm A sensors, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--arrays-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    info = pd.read_csv(args.raw_dir / "comma_event_info.csv")
    info["event_start"] = pd.to_datetime(info["event_start"])
    info["event_end"] = pd.to_datetime(info["event_end"])
    meta = pd.read_csv(args.raw_dir / "comma_feature_description.csv")
    first_path = args.raw_dir / f"comma_{int(info.iloc[0]['event_id'])}.csv"
    columns = legacy_sensor_columns(meta, pd.read_csv(first_path, nrows=0).columns.tolist())
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[split] = {
            "y": np.load(args.arrays_dir / f"{split}_y.npy"),
            "mask": np.load(args.arrays_dir / f"{split}_mask.npy"),
        }
        shape = arrays[split]["y"].shape
        if shape != (EXPECTED_COUNTS[split], SEQ_LEN):
            raise ValueError(f"Legacy {split}_y shape {shape}; expected {(EXPECTED_COUNTS[split], SEQ_LEN)}")
        if arrays[split]["mask"].shape != shape:
            raise ValueError(f"Legacy {split} y/mask shapes differ")
    offsets = {split: 0 for split in EXPECTED_COUNTS}
    metadata = {split: [] for split in EXPECTED_COUNTS}
    for row in info.sort_values("event_id").itertuples(index=False):
        event_id = int(row.event_id)
        event_end = pd.Timestamp(row.event_end)
        path = args.raw_dir / f"comma_{event_id}.csv"
        frame = pd.read_csv(path, usecols=["time_stamp", "asset_id"] + columns)
        frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], errors="coerce")
        frame = (
            frame.dropna(subset=["time_stamp"])
            .sort_values("time_stamp")
            .drop_duplicates("time_stamp")
            .set_index("time_stamp")
        )
        assets = frame["asset_id"].dropna().unique()
        if len(assets) != 1 or int(assets[0]) not in ASSET_SPLIT:
            raise ValueError(f"Legacy event {event_id} has unexpected assets {assets}")
        asset_id = int(assets[0])
        split = ASSET_SPLIT[asset_id]
        candidates = pd.date_range(
            pd.Timestamp(row.event_start),
            event_end - pd.Timedelta(minutes=INTERVAL_MINUTES),
            freq=f"{STRIDE_HOURS}h",
        )
        for sequence_end in candidates:
            if sequence_end not in frame.index:
                continue
            start = sequence_end - pd.Timedelta(hours=LOOKBACK_HOURS) + pd.Timedelta(minutes=INTERVAL_MINUTES)
            regular = pd.date_range(start, sequence_end, freq=f"{INTERVAL_MINUTES}min")
            observed = frame.loc[(frame.index >= start) & (frame.index <= sequence_end)]
            if len(observed) / SEQ_LEN < MIN_COVERAGE:
                continue
            diffs = observed.index.to_series().diff().dropna()
            if len(diffs) and diffs.max().total_seconds() / 60.0 > MAX_GAP_MINUTES:
                continue
            aligned = observed[columns].reindex(regular)
            replay_mask = aligned.notna().all(axis=1).astype(np.uint8).to_numpy()
            replay_y = timestep_labels(regular, str(row.event_label), event_end)
            sequence_idx = offsets[split]
            if sequence_idx >= len(arrays[split]["y"]):
                raise ValueError(f"Replay generated too many {split} sequences")
            if not np.array_equal(replay_y, arrays[split]["y"][sequence_idx]):
                raise ValueError(f"Legacy y alignment failed at {split} sequence {sequence_idx}, event {event_id}")
            if not np.array_equal(replay_mask, arrays[split]["mask"][sequence_idx]):
                raise ValueError(f"Legacy mask alignment failed at {split} sequence {sequence_idx}, event {event_id}")
            metadata[split].append(
                build_timestep_metadata(
                    farm="A",
                    split=split,
                    sequence_idx=sequence_idx,
                    index=regular,
                    asset_id=asset_id,
                    event_id=event_id,
                    event_label=str(row.event_label),
                    event_end=event_end,
                    labels=replay_y,
                    mask=replay_mask,
                )
            )
            offsets[split] += 1
    if offsets != EXPECTED_COUNTS:
        raise ValueError(f"Legacy replay counts {offsets}; expected {EXPECTED_COUNTS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frames in metadata.items():
        result = pd.concat(frames, ignore_index=True)
        if len(result) != EXPECTED_COUNTS[split] * SEQ_LEN:
            raise RuntimeError(f"Legacy {split} metadata is not N * 144")
        output = args.output_dir / f"{split}_metadata.parquet"
        result.to_parquet(output, index=False, compression="zstd")
        print(f"Proved alignment and wrote {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
