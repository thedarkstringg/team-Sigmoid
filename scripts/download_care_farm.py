"""Download exactly one CARE farm from Kaggle; never invoked automatically."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import pandas as pd


DEFAULT_DATASET = "azizkasimov/wind-turbine-scada-data-for-early-fault-detection"


def download_one(dataset: str, remote: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Already present: {destination}")
        return
    downloaded = Path(kagglehub.dataset_download(dataset, path=remote, force_download=force))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    print(f"Downloaded {remote} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--event-id", type=int, action="append", help="Repeat to download a subset; default is all listed events")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    farm_dir = args.output_root / f"CARE_Farm_{args.farm}"
    remote_dir = f"Wind Farm {args.farm}"
    for name in ("comma_event_info.csv", "comma_feature_description.csv"):
        download_one(args.dataset, f"{remote_dir}/{name}", farm_dir / name, args.force)
    info = pd.read_csv(farm_dir / "comma_event_info.csv")
    available = set(info["event_id"].astype(int))
    requested = sorted(set(args.event_id) if args.event_id else available)
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"Requested event IDs are absent from event metadata: {sorted(unknown)}")
    for event_id in requested:
        name = f"comma_{event_id}.csv"
        download_one(args.dataset, f"{remote_dir}/{name}", farm_dir / name, args.force)


if __name__ == "__main__":
    main()

