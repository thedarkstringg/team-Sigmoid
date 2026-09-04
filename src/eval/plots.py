"""
The three figures the brief requires alongside the metrics table: reliability
diagram, lead-time distribution, per-turbine breakdown.

Each function takes the output of the matching metrics.py function directly
and writes a PNG. No plot here recomputes anything - if a number in a figure
disagrees with the metrics table, the bug is in metrics.py, not here.

Uses only matplotlib (already in requirements.txt) with the Agg backend, so
these run headlessly in a Jupyter kernel or a container with no display.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import LABEL_HORIZON_HOURS

FIGSIZE = (7, 5)
DPI = 150


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 4 figure - reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(reliability_table: pd.DataFrame, out_path: str,
                             title: str = "Reliability diagram",
                             brier: float | None = None,
                             ece: float | None = None) -> None:
    """
    Predicted vs observed fault rate per bin, against the diagonal.

    Marker area scales with bin count so a reader can see which bins are
    well-populated and which are one or two windows - important here, since
    several bins have single-digit counts (see the Task 4 verification run).
    Empty bins are plotted as open circles on the x-axis so gaps in coverage
    are visible rather than silently skipped.
    """
    filled = reliability_table[reliability_table["count"] > 0]
    empty = reliability_table[reliability_table["count"] == 0]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="perfect calibration")

    if not filled.empty:
        sizes = 30 + 400 * (filled["count"] / filled["count"].max())
        ax.scatter(filled["mean_predicted"], filled["observed_rate"],
                   s=sizes, alpha=0.75, edgecolor="black", linewidth=0.5,
                   label="model (marker area ~ bin count)")
        for _, row in filled.iterrows():
            ax.annotate(f"n={int(row['count'])}",
                       (row["mean_predicted"], row["observed_rate"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=7)

    if not empty.empty:
        midpoints = (empty["bin_lower"] + empty["bin_upper"]) / 2
        ax.scatter(midpoints, np.zeros(len(empty)), marker="o",
                   facecolors="none", edgecolors="lightgray", s=40,
                   label="empty bin")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fault rate")

    subtitle_parts = []
    if brier is not None:
        subtitle_parts.append(f"Brier = {brier:.4f}")
    if ece is not None:
        subtitle_parts.append(f"ECE = {ece:.4f}")
    full_title = title + ("\n" + ", ".join(subtitle_parts) if subtitle_parts else "")

    ax.set_title(full_title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Task 1 figure - lead-time distribution
# ---------------------------------------------------------------------------

def plot_lead_time_distribution(per_event: pd.DataFrame, out_path: str,
                                title: str = "Lead time by fault event",
                                benchmark_hours: float = 48.0) -> None:
    """
    One bar per fault event, not a histogram.

    With as few as 2-4 fault events (Farm A val+test combined), a histogram
    implies a sample size this data doesn't have. A labelled bar per event is
    honest about n and still shows the spread the roadmap asks for. Missed
    events (detected == False) are drawn as a red bar at the axis to keep
    them visible in the same figure rather than silently excluded from the
    average.
    """
    df = per_event.sort_values(["asset_id", "event_id"]).reset_index(drop=True)
    labels = [f"asset {r.asset_id}\nevent {r.event_id}" for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(df))

    detected = df["detected"].to_numpy()
    heights = df["lead_time_hours"].fillna(0).to_numpy()
    colors = ["#2a7f2a" if in_h else "#c9a227"
              for in_h in df["within_label_horizon"]]
    colors = [c if d else "#c0392b" for c, d in zip(colors, detected)]

    ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)

    for i, row in df.iterrows():
        if row["detected"]:
            ax.annotate(f"{row['lead_time_hours']:.0f}h",
                       (i, row["lead_time_hours"]),
                       textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=8)
        else:
            ax.annotate("missed", (i, 0),
                       textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=8, color="#c0392b")

    ax.axhline(benchmark_hours, linestyle="--", color="gray", linewidth=1,
              label=f"reference lead time ({benchmark_hours:.0f} h)")

    if not np.isclose(benchmark_hours, LABEL_HORIZON_HOURS):
        ax.axhline(LABEL_HORIZON_HOURS, linestyle=":", color="black", linewidth=1,
                  label=f"label horizon ({LABEL_HORIZON_HOURS:.0f} h)")
    else:
        # In this dataset the CARE-to-Compare reference lead time and the
        # sequence-export label horizon are numerically identical (both 48h)
        # - a genuine property of the data, not a plotting coincidence. Draw
        # one line, and say so rather than overlapping two indistinguishable
        # dashes.
        ax.annotate("reference lead time = label horizon",
                   (0.02, benchmark_hours), xycoords=("axes fraction", "data"),
                   ha="left", va="top", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Lead time (hours before fault)")
    ax.set_title(f"{title}  (n = {len(df)} fault events)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Task 2 figure - per-turbine / per-farm breakdown
# ---------------------------------------------------------------------------

def plot_turbine_breakdown(breakdown_df: pd.DataFrame, out_path: str,
                           group_col: str = "asset_id",
                           metric_cols: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "recall"),
                           title: str = "Per-turbine performance") -> None:
    """
    Grouped bars, one cluster per turbine (or farm), one bar per metric.

    The 'ALL' pooled row that breakdown() appends is drawn in a lighter shade
    and separated from the individual groups with a vertical divider, so a
    reader can see at a glance how far any one turbine departs from the
    pooled figure. NaN bars (single-class groups) are left as gaps, not
    zeros - a zero there would misreport a metric that could not be computed.
    """
    df = breakdown_df.copy()
    is_pooled = df[group_col].astype(str) == "ALL"

    n_groups = len(df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(FIGSIZE[0], n_groups * 1.4), FIGSIZE[1]))

    for i, metric in enumerate(metric_cols):
        offsets = x + (i - (n_metrics - 1) / 2) * width
        values = df[metric].to_numpy()
        colors = ["lightgray" if pooled else None for pooled in is_pooled]
        bars = ax.bar(offsets, np.nan_to_num(values, nan=0.0), width=width,
                      label=metric, alpha=0.9)
        for bar, v, pooled in zip(bars, values, is_pooled):
            if np.isnan(v):
                bar.set_visible(False)
            elif pooled:
                bar.set_alpha(0.5)
                bar.set_hatch("//")

    if is_pooled.any():
        divider_x = np.flatnonzero(is_pooled.to_numpy())[0] - 0.5
        ax.axvline(divider_x, color="black", linewidth=1, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels(df[group_col].astype(str), fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=n_metrics)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# convenience: combine per-event tables across splits for a fuller n
# ---------------------------------------------------------------------------

def combine_splits(per_split: dict) -> pd.DataFrame:
    """
    Stack lead_time() outputs from multiple splits (e.g. val and test) into
    one table for plotting. Farm A has only 2 fault events per split; the
    combined val+test view gives n=4 instead of n=2, still small but a more
    honest picture than either split alone.
    """
    frames = []
    for split_name, df in per_split.items():
        tagged = df.copy()
        tagged["split"] = split_name
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True)