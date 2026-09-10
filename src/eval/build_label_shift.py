"""
Task 17 - label shift analysis (Farm C source vs. Farm A/B targets).

Distinguishes differences in fault LABELS across farms, as opposed to
Task 16 (build_distribution_shift.py), which covers sensor FEATURE shift.
A model can face two separate kinds of domain gap: the inputs can look
different (Task 16), or what counts as a "fault" and how often it occurs
can differ (this task). Both matter for explaining cross-farm performance,
and conflating them would hide which one actually drives the gap.

Compares, per farm test split:
  1. Positive rate (fraction of valid timesteps labelled within the 48h
     fault horizon) - a large gap here means the model is calibrated to
     one base rate and evaluated against a very different one.
  2. Fault event count and duration statistics - farms with few, long
     events vs. many, short events pose a different learning problem even
     at matched positive rate.
  3. Label derivation methodology - documented, not computed: Farm A labels
     derive from EDP's fault logbook (explicit onset timestamp); Farms B/C
     labels derive from operator-provided operating-mode codes and service
     reports (per Gueck & Roelofs 2024, the CARE-to-Compare paper). A model
     trained on one labelling philosophy encodes assumptions about what an
     "onset" looks like that may not transfer to a differently-derived
     label on another farm - a real, citable limitation, not something a
     script can quantify directly, so it is reported as a fixed note
     rather than a computed statistic.

Author: Ziyad (Kamal's Task 17, delegated to Emin, not completed in time -
see contribution_report.pdf for the handoff note).

Usage (from repo root):
    python src/eval/build_label_shift.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


LABEL_METHODOLOGY_NOTE = {
    "A": "EDP fault logbook: explicit onset timestamp per event, curated "
         "by the site operator from SCADA alarms and maintenance records.",
    "B": "Operator-provided operating-mode codes plus service reports; "
         "onset inferred from mode-code transitions, not a single "
         "curated timestamp (per Gueck & Roelofs 2024).",
    "C": "Operator-provided operating-mode codes plus service reports; "
         "same derivation method as Farm B, distinct from Farm A's "
         "logbook-based labels (per Gueck & Roelofs 2024).",
}


def load_metadata(data_dir: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir, f"{split}_metadata.parquet"))


def load_mask(data_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, f"{split}_mask.npy"))


def load_labels(data_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, f"{split}_y.npy"))


def summarize_farm(data_dir: str, split: str, farm_name: str) -> dict:
    y = load_labels(data_dir, split)
    mask = load_mask(data_dir, split)
    meta = load_metadata(data_dir, split)

    valid = mask.astype(bool)
    y_flat = y.flatten()
    valid_flat = valid.flatten()

    positive_rate = float(y_flat[valid_flat].mean())

    if "event_id" in meta.columns and "asset_id" in meta.columns:
        meta = meta.copy()
        meta["label"] = y_flat
        meta["valid"] = valid_flat
        fault_events = (
            meta[meta["valid"] & (meta["label"] == 1)]
            .groupby(["asset_id", "event_id"])
            .size()
        )
        n_fault_events = int(len(fault_events))
        event_duration_timesteps = fault_events.to_numpy()
        duration_stats = {
            "mean_timesteps": float(event_duration_timesteps.mean()) if n_fault_events else None,
            "median_timesteps": float(np.median(event_duration_timesteps)) if n_fault_events else None,
            "min_timesteps": int(event_duration_timesteps.min()) if n_fault_events else None,
            "max_timesteps": int(event_duration_timesteps.max()) if n_fault_events else None,
        }
    else:
        n_fault_events = None
        duration_stats = None

    return {
        "farm": farm_name,
        "split": split,
        "n_valid_timesteps": int(valid_flat.sum()),
        "positive_rate": positive_rate,
        "n_fault_events": n_fault_events,
        "fault_event_duration_timesteps": duration_stats,
        "label_methodology": LABEL_METHODOLOGY_NOTE[farm_name],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-c-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_C/sequences_v2")
    parser.add_argument("--farm-c-split", default="test")
    parser.add_argument("--farm-a-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_A/physical_sequences")
    parser.add_argument("--farm-b-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_B/physical_sequences")
    parser.add_argument("--out-json", default="artifacts/label_shift.json")
    parser.add_argument("--out-md", default="artifacts/label_shift.md")
    args = parser.parse_args()

    results = {
        "C": summarize_farm(args.farm_c_dir, args.farm_c_split, "C"),
        "A": summarize_farm(args.farm_a_dir, "test", "A"),
        "B": summarize_farm(args.farm_b_dir, "test", "B"),
    }

    rates = {f: r["positive_rate"] for f, r in results.items()}
    max_rate, min_rate = max(rates.values()), min(rates.values())
    results["positive_rate_ratio_max_to_min"] = (
        float(max_rate / min_rate) if min_rate > 0 else float("inf")
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = ["# Label Shift Analysis (Task 17)\n"]
    lines.append("Compares fault-label characteristics across farms - distinct from "
                 "Task 16's sensor-feature shift analysis.\n")
    lines.append("| Farm | Split | Positive rate | # fault events | Median event duration (timesteps) |")
    lines.append("|---|---|---:|---:|---:|")
    for f in ["A", "B", "C"]:
        r = results[f]
        med = r["fault_event_duration_timesteps"]["median_timesteps"] if r["fault_event_duration_timesteps"] else "n/a"
        lines.append(f"| {f} | {r['split']} | {r['positive_rate']:.4f} | {r['n_fault_events']} | {med} |")
    lines.append(f"\nRatio of highest to lowest positive rate across farms: "
                 f"**{results['positive_rate_ratio_max_to_min']:.2f}x**\n")
    lines.append("\n## Label derivation methodology (documented, not computed)\n")
    for f in ["A", "B", "C"]:
        lines.append(f"- **Farm {f}:** {LABEL_METHODOLOGY_NOTE[f]}")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"\nPositive rates: {rates}")
    print(f"Max/min ratio: {results['positive_rate_ratio_max_to_min']:.2f}x")


if __name__ == "__main__":
    main()
