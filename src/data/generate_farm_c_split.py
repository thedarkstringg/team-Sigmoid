"""Generate a deterministic, asset-disjoint Farm C split from raw event files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.data.sequence_utils import read_care_csv
except ModuleNotFoundError:  # Support direct script invocation from repo root.
    from sequence_utils import read_care_csv


def event_assets(raw_dir: Path, event_info: pd.DataFrame) -> dict[int, str]:
    result: dict[int, str] = {}
    for event_id in sorted(event_info["event_id"].astype(int)):
        path = raw_dir / f"comma_{event_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        values = read_care_csv(path, usecols=["asset_id"])["asset_id"].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Event {event_id} does not prove one asset_id: {values}")
        result[event_id] = str(values[0])
    return result


def choose_split(
    assets: list[str], events: pd.DataFrame, seed: int, trials: int
) -> dict[str, list[str]]:
    if len(assets) < 6:
        raise ValueError("Farm C needs at least six assets for multi-asset val/test splits")
    n_val = max(2, round(0.15 * len(assets)))
    n_test = max(2, round(0.15 * len(assets)))
    n_train = len(assets) - n_val - n_test
    rng = np.random.default_rng(seed)
    global_counts = events["event_label"].value_counts()
    target_asset = {"train": 0.70, "val": 0.15, "test": 0.15}
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    best: tuple[float, dict[str, list[str]]] | None = None
    for _ in range(trials):
        shuffled = list(rng.permutation(assets))
        candidate = {
            "train": sorted(shuffled[:n_train]),
            "val": sorted(shuffled[n_train:n_train + n_val]),
            "test": sorted(shuffled[n_train + n_val:]),
        }
        score = 0.0
        for split, split_assets in candidate.items():
            subset = events[events["asset_id"].isin(split_assets)]
            score += 10.0 * (sizes[split] / len(assets) - target_asset[split]) ** 2
            for label in ("anomaly", "normal"):
                count = int((subset["event_label"] == label).sum())
                expected = global_counts.get(label, 0) * target_asset[split]
                score += ((count - expected) / max(1.0, expected)) ** 2
                if count == 0:
                    score += 1000.0
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/farm_c_split.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()
    info_path = args.raw_dir / "comma_event_info.csv"
    events = read_care_csv(info_path)
    asset_by_event = event_assets(args.raw_dir, events)
    events["asset_id"] = events["event_id"].astype(int).map(asset_by_event)
    assets = sorted(events["asset_id"].unique())
    split = choose_split(assets, events, args.seed, args.trials)
    event_summaries = {}
    for name, split_assets in split.items():
        subset = events[events["asset_id"].isin(split_assets)]
        missing_labels = {
            label for label in ("anomaly", "normal")
            if not (subset["event_label"] == label).any()
        }
        if missing_labels:
            raise RuntimeError(
                f"Deterministic search did not find {name} coverage for {sorted(missing_labels)}; "
                "increase --trials or review whether asset-level coverage is feasible"
            )
        event_summaries[name] = {
            "assets": split_assets,
            "event_ids": sorted(subset["event_id"].astype(int).tolist()),
            "anomaly_events": int((subset["event_label"] == "anomaly").sum()),
            "normal_events": int((subset["event_label"] == "normal").sum()),
        }
    payload = {
        "schema_version": "farm-c-asset-split-v1",
        "farm": "C",
        "seed": args.seed,
        "trials": args.trials,
        "target_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "source_event_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "splits": event_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
