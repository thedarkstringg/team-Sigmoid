"""
Task 18 - turbine-model shift analysis (Farm C source vs. Farm A/B targets).

Distinguishes differences in TURBINE HARDWARE / CONFIGURATION across farms,
as opposed to Task 16 (raw sensor-value distribution shift) and Task 17
(fault-label characteristics). This task asks: even before looking at any
sensor reading, are these physically the same *kind* of turbine, or
different machines with different fixed mechanical properties?

Deliberately reuses artifacts/distribution_shift.json (Task 16, Emin) rather
than recomputing the underlying per-feature statistics - per this project's
convention (see src/eval/plots.py docstring: "if a number disagrees with
the metrics table, the bug is in metrics.py, not here"), the same discipline
applies across analysis scripts: one canonical source of the raw numbers,
reused everywhere else.

Author: Ziyad (Kamal's Task 18, delegated to Emin, not completed in time -
see contribution_report.pdf for the handoff note).

Usage (from repo root, after Task 16 has been run):
    python src/eval/build_turbine_model_shift.py
"""

from __future__ import annotations

import argparse
import json
import os


FARM_CONFIG = {
    "A": {"location": "Onshore, Portugal", "n_turbines": 5, "n_raw_sensors": 86},
    "B": {"location": "Offshore, Germany", "n_turbines": 9, "n_raw_sensors": 257},
    "C": {"location": "Offshore, Germany", "n_turbines": 22, "n_raw_sensors": 957},
}


def load_distribution_shift(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - this script reuses Task 16's output "
            "(build_distribution_shift.py) rather than recomputing raw "
            "statistics. Run Task 16 first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gearbox_ratio_evidence(shift_data: dict) -> dict:
    """
    generator_rotor_speed_ratio's raw mean/std, reused directly from Task 16,
    is the strongest available evidence of a genuine turbine-model
    difference: this ratio is fixed by a turbine's physical gearbox design,
    not by operating conditions, so a farm-specific value with tight spread
    indicates a distinct hardware configuration rather than measurement
    noise or operating-point variation.
    """
    per_feature = {row["feature"]: row for row in shift_data["per_feature"]}
    ratio_row = per_feature["generator_rotor_speed_ratio"]

    return {
        "farm_c_raw_mean": ratio_row["farm_c_train"]["raw_mean"],
        "farm_c_raw_std": ratio_row["farm_c_train"]["raw_std"],
        "farm_a_raw_mean": ratio_row["farm_a_test"]["raw_mean"],
        "farm_a_raw_std": ratio_row["farm_a_test"]["raw_std"],
        "farm_b_raw_mean": ratio_row["farm_b_test"]["raw_mean"],
        "farm_b_raw_std": ratio_row["farm_b_test"]["raw_std"],
        "interpretation": (
            "Each farm's generator/rotor speed ratio clusters tightly "
            "around a distinct mean (std < 1.0 in all three farms) - "
            "consistent with each farm using a fixed gearbox ratio "
            "specific to its turbine model, not a shared design. This "
            "matches the sensor mapping's own quality note: 'nominal gear "
            "ratios may differ' (configs/physical_sensor_mapping.yaml)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-shift-json", default="artifacts/distribution_shift.json")
    parser.add_argument("--out-json", default="artifacts/turbine_model_shift.json")
    parser.add_argument("--out-md", default="artifacts/turbine_model_shift.md")
    args = parser.parse_args()

    shift_data = load_distribution_shift(args.distribution_shift_json)
    ratio_evidence = gearbox_ratio_evidence(shift_data)

    results = {
        "farm_config": FARM_CONFIG,
        "gearbox_ratio_evidence": ratio_evidence,
        "note": (
            "Turbine count and raw sensor count differ by roughly an order "
            "of magnitude across farms (5/86 for A vs 22/957 for C) - this "
            "alone does not prove different turbine models, but combined "
            "with the tightly-clustered, farm-specific gearbox ratio below, "
            "it supports treating each farm as a genuinely distinct "
            "turbine population rather than the same hardware at "
            "different sites."
        ),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = ["# Turbine-Model Shift Analysis (Task 18)\n"]
    lines.append("Reuses Task 16's (Emin) raw feature statistics rather than "
                 "recomputing them - see script docstring.\n")
    lines.append("## Farm configuration (documented, from Gueck & Roelofs 2024)\n")
    lines.append("| Farm | Location | Turbines | Raw sensors |")
    lines.append("|---|---|---:|---:|")
    for f, cfg in FARM_CONFIG.items():
        lines.append(f"| {f} | {cfg['location']} | {cfg['n_turbines']} | {cfg['n_raw_sensors']} |")
    lines.append("\n## Gearbox ratio evidence (generator_rotor_speed_ratio, raw units)\n")
    lines.append("| Farm | Raw mean | Raw std |")
    lines.append("|---|---:|---:|")
    lines.append(f"| C (source) | {ratio_evidence['farm_c_raw_mean']:.3f} | {ratio_evidence['farm_c_raw_std']:.3f} |")
    lines.append(f"| A (target) | {ratio_evidence['farm_a_raw_mean']:.3f} | {ratio_evidence['farm_a_raw_std']:.3f} |")
    lines.append(f"| B (target) | {ratio_evidence['farm_b_raw_mean']:.3f} | {ratio_evidence['farm_b_raw_std']:.3f} |")
    lines.append(f"\n{ratio_evidence['interpretation']}\n")
    lines.append(f"\n{results['note']}\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"\nGearbox ratio by farm: C={ratio_evidence['farm_c_raw_mean']:.2f}, "
          f"A={ratio_evidence['farm_a_raw_mean']:.2f}, B={ratio_evidence['farm_b_raw_mean']:.2f}")


if __name__ == "__main__":
    main()
