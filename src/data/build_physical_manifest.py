"""Build the committed strict physical feature manifest from the YAML mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.physical_features import load_mapping


def build_manifest(config: dict) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(config["strict_feature_order"]):
        definition = config["features"][name]
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "unit": definition["unit"],
                "formula": definition["formula"],
                "statistic": config["statistic"],
                "farm_a_sources": json.dumps(definition["farms"]["A"], sort_keys=True),
                "farm_b_sources": json.dumps(definition["farms"]["B"], sort_keys=True),
                "farm_c_sources": json.dumps(definition["farms"]["C"], sort_keys=True),
                "validity": definition["validity"],
                "quality_note": definition["quality_note"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/physical_sensor_mapping.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/physical_feature_manifest.csv"))
    args = parser.parse_args()
    manifest = build_manifest(load_mapping(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} strict features to {args.output}")


if __name__ == "__main__":
    main()

