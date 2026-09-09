"""
Task 16 - distribution shift analysis (Farm C source vs. Farm A/B targets).

Compares the 10 physical features on two axes, deliberately kept separate
because they answer different questions and one can mislead if reported
alone:

  1. RAW physical values - interpretable in real units (m/s, degrees C, etc).
  2. SCALED (model input) values, using Farm C's FROZEN train-fit scaler
     applied unchanged to every farm, per the roadmap's zero-shot
     constraints (no refit on target domains).

Why not just a standardized-mean-difference on raw values: for at least two
features (gearbox_oil_rise_C, gearbox_bearing_hotspot_over_oil_C), Farm C's
own raw standard deviation is itself inflated by what looks like a Farm
C-specific data-quality issue (echoes the grid_power_factor unit-mismatch
bug already documented in this repo). Pooling with that inflated std would
make the shift look artificially SMALL for exactly the two features where
it matters most. The scaled-space compression ratio below does not have
this blind spot: it directly measures what actually happens to each
feature's real physical variation once it passes through the frozen scaler
into the shape the model actually sees.

Does not invent numbers - every value here traces back to test_X.npy,
test_mask.npy and scaler_stats.npz for each farm.

Usage (from repo root):
    python src/eval/build_distribution_shift.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_valid_raw_and_scaled(data_dir: str, split: str,
                              fit_mean: np.ndarray, fit_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(os.path.join(data_dir, f"{split}_X.npy"))
    mask = np.load(os.path.join(data_dir, f"{split}_mask.npy"))
    valid = mask.astype(bool)
    scaled = X[valid]
    raw = scaled * fit_std + fit_mean
    return raw, scaled


def summarize_feature(raw: np.ndarray, scaled: np.ndarray) -> dict:
    return {
        "n_valid_timesteps": int(raw.shape[0]),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std()),
        "raw_median": float(np.median(raw)),
        "raw_q01": float(np.quantile(raw, 0.01)),
        "raw_q99": float(np.quantile(raw, 0.99)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "scaled_mean": float(scaled.mean()),
        "scaled_std": float(scaled.std()),
        "scaled_min": float(scaled.min()),
        "scaled_max": float(scaled.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_C/sequences/scaler_stats.npz")
    parser.add_argument("--farm-c-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_C/sequences")
    parser.add_argument("--farm-c-split", default="train",
                        help="the split the scaler was actually fit on - use this as the source reference")
    parser.add_argument("--farm-a-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_A/physical_sequences")
    parser.add_argument("--farm-b-dir", default=
        "/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_B/physical_sequences")
    parser.add_argument("--out-json", default="artifacts/distribution_shift.json")
    parser.add_argument("--out-md", default="artifacts/distribution_shift.md")
    parser.add_argument("--fig1", default="figures/eval/distribution_shift_features.png")
    parser.add_argument("--fig2", default="figures/eval/distribution_shift_compression.png")
    args = parser.parse_args()

    scaler = np.load(args.scaler, allow_pickle=False)
    feature_names = scaler["feature_names"].tolist()
    fit_mean = scaler["mean"]
    fit_std = scaler["std"]

    farm_c_raw, farm_c_scaled = load_valid_raw_and_scaled(
        args.farm_c_dir, args.farm_c_split, fit_mean, fit_std)
    farm_a_raw, farm_a_scaled = load_valid_raw_and_scaled(
        args.farm_a_dir, "test", fit_mean, fit_std)
    farm_b_raw, farm_b_scaled = load_valid_raw_and_scaled(
        args.farm_b_dir, "test", fit_mean, fit_std)

    per_feature = []
    for i, name in enumerate(feature_names):
        row = {
            "feature": name,
            "farm_c_train": summarize_feature(farm_c_raw[:, i], farm_c_scaled[:, i]),
            "farm_a_test": summarize_feature(farm_a_raw[:, i], farm_a_scaled[:, i]),
            "farm_b_test": summarize_feature(farm_b_raw[:, i], farm_b_scaled[:, i]),
        }
        # Compression factor: how many times NARROWER this feature's scaled
        # spread is on the target farm than on Farm C (which is ~1.0 by
        # construction, since the scaler was fit on it). A factor of 16
        # means the target farm's real variation in this feature occupies
        # only 1/16th of the input range the model was trained to expect.
        row["farm_a_compression_factor"] = (
            1.0 / row["farm_a_test"]["scaled_std"] if row["farm_a_test"]["scaled_std"] > 0 else float("inf")
        )
        row["farm_b_compression_factor"] = (
            1.0 / row["farm_b_test"]["scaled_std"] if row["farm_b_test"]["scaled_std"] > 0 else float("inf")
        )
        per_feature.append(row)

    # Rank by worst (max) compression across either target farm - this is
    # the ranking that answers "which physical variables change most".
    ranked = sorted(
        per_feature,
        key=lambda r: max(r["farm_a_compression_factor"], r["farm_b_compression_factor"]),
        reverse=True,
    )

    results = {
        "source_farm": "C",
        "source_split_used_for_reference": args.farm_c_split,
        "note_on_method": (
            "Compression factor (not standardized mean difference) is the "
            "primary shift statistic reported here. See module docstring: "
            "raw-value SMD would understate the shift for features where "
            "Farm C's own raw variance is itself inflated by an apparent "
            "data-quality issue, since that inflation dominates the pooled "
            "denominator. Compression factor instead measures what the "
            "model actually receives after the frozen scaler is applied, "
            "which has no such blind spot."
        ),
        "per_feature": per_feature,
        "ranked_by_worst_compression": [
            {"feature": r["feature"],
             "farm_a_compression_factor": round(r["farm_a_compression_factor"], 2),
             "farm_b_compression_factor": round(r["farm_b_compression_factor"], 2)}
            for r in ranked
        ],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- Figure 2: compression factor per feature (the headline figure) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.fig2) or ".", exist_ok=True)
    names = [r["feature"] for r in ranked]
    a_vals = [r["farm_a_compression_factor"] for r in ranked]
    b_vals = [r["farm_b_compression_factor"] for r in ranked]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(names))
    ax.barh(y - 0.2, a_vals, height=0.4, label="Farm A", color="#3b7dd8")
    ax.barh(y + 0.2, b_vals, height=0.4, label="Farm B", color="#d87d3b")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="no compression (Farm C baseline)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Compression factor (log scale) - higher = more of this feature's\nreal variation is lost under Farm C's frozen scaler")
    ax.set_title("Task 16: Feature-wise distribution-shift summary")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.fig2, dpi=150)
    plt.close(fig)

    # --- Figure 1: raw distribution comparison for the 3 worst features ---
    os.makedirs(os.path.dirname(args.fig1) or ".", exist_ok=True)
    worst_3 = ranked[:3]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, r in zip(axes, worst_3):
        idx = feature_names.index(r["feature"])
        ax.hist(farm_c_raw[:, idx], bins=60, alpha=0.5, density=True, label="Farm C (train)", color="#555555")
        ax.hist(farm_a_raw[:, idx], bins=60, alpha=0.5, density=True, label="Farm A (test)", color="#3b7dd8")
        ax.hist(farm_b_raw[:, idx], bins=60, alpha=0.5, density=True, label="Farm B (test)", color="#d87d3b")
        ax.set_title(r["feature"], fontsize=9)
        ax.set_xlabel("raw value")
    axes[0].legend(fontsize=7)
    fig.suptitle("Task 16: Feature distribution comparison, three most-shifted features (raw units)")
    fig.tight_layout()
    fig.savefig(args.fig1, dpi=150)
    plt.close(fig)

    # --- Markdown summary ---
    lines = ["# Distribution Shift Analysis (Task 16)\n"]
    lines.append(f"Source: Farm C ({args.farm_c_split} split, the scaler's own fit population). "
                 f"Targets: Farm A test, Farm B test (zero-shot, frozen Farm C scaler).\n")
    lines.append(results["note_on_method"] + "\n")
    lines.append("\n## Features ranked by worst compression (most shifted first)\n")
    lines.append("| Feature | Farm A compression | Farm B compression |")
    lines.append("|---|---:|---:|")
    for r in results["ranked_by_worst_compression"]:
        lines.append(f"| {r['feature']} | {r['farm_a_compression_factor']}x | {r['farm_b_compression_factor']}x |")
    lines.append(f"\n![Compression summary]({os.path.basename(args.fig2)})\n")
    lines.append(f"\n![Distribution comparison]({os.path.basename(args.fig1)})\n")
    lines.append("\n## Full per-feature statistics (raw units)\n")
    lines.append("| Feature | Farm C mean | Farm C std | Farm A mean | Farm A std | Farm B mean | Farm B std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in per_feature:
        c, a, b = r["farm_c_train"], r["farm_a_test"], r["farm_b_test"]
        lines.append(f"| {r['feature']} | {c['raw_mean']:.3f} | {c['raw_std']:.3f} | "
                     f"{a['raw_mean']:.3f} | {a['raw_std']:.3f} | {b['raw_mean']:.3f} | {b['raw_std']:.3f} |")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.fig1}")
    print(f"Wrote {args.fig2}")
    print("\nWorst-compressed features:")
    for r in results["ranked_by_worst_compression"][:3]:
        print(f"  {r['feature']}: Farm A {r['farm_a_compression_factor']}x, Farm B {r['farm_b_compression_factor']}x")


if __name__ == "__main__":
    main()
