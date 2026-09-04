"""
Evaluation metrics for the CARE fault-prediction models.

All four metrics consume ONE table with the following columns:

    event_id        int    identifies a contiguous fault or normal event
    asset_id        int    turbine identifier
    window_end      str    timestamp of the last timestep in the window
    hours_to_fault  float  hours from window_end to the fault; NaN for normal events
    label           int    1 if the fault is within the labelled horizon (48 h)
    probability     float  predicted fault probability in [0, 1]

Optional:

    mask            int    1 for real observations, 0 for gap-filled. Rows with
                           mask == 0 are dropped by prepare(). Absent -> all real.
    split           str    'val' / 'test'
    model           str    used to select one model from a multi-model table

The baseline table (artifacts/baseline/baseline_predictions.csv) already has
this shape. The GRU export must match it, one row per timestep instead of one
row per window. Nothing below depends on torch or on which farm produced the
predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REQUIRED_COLUMNS = (
    "event_id",
    "asset_id",
    "hours_to_fault",
    "label",
    "probability",
)

LABEL_HORIZON_HOURS = 48.0


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, model: str | None = None,
            split: str | None = None) -> pd.DataFrame:
    """Validate, filter and return a clean copy of the prediction table."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")

    out = df.copy()

    if model is not None:
        if "model" not in out.columns:
            raise ValueError("model= was given but the table has no 'model' column")
        out = out[out["model"] == model]

    if split is not None:
        if "split" not in out.columns:
            raise ValueError("split= was given but the table has no 'split' column")
        out = out[out["split"] == split]

    if "mask" in out.columns:
        out = out[out["mask"].astype(bool)]

    if out.empty:
        raise ValueError("no rows left after filtering")

    if not out["probability"].between(0.0, 1.0).all():
        raise ValueError("probability column contains values outside [0, 1]")

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 1 - lead time
# ---------------------------------------------------------------------------

def lead_time(df: pd.DataFrame, threshold: float,
              min_consecutive: int = 1,
              search_window_hours: float | None = None) -> pd.DataFrame:
    """
    Hours between the first alarm and the fault, one row per fault event.

    An event contributes a row only if it has a fault (hours_to_fault present).
    Windows are ordered from earliest (largest hours_to_fault) to latest, and
    the first run of `min_consecutive` windows above `threshold` counts as the
    alarm. Lead time is hours_to_fault at the START of that run.

    min_consecutive > 1 suppresses single-window spikes that would otherwise
    report an implausibly long lead time off one noisy prediction.

    search_window_hours caps how early an alarm may be and still count. This
    matters: an isolated high-confidence burst a week before a fault, with
    nothing during the labelled window, is not an early detection, but an
    uncapped search reports it as the best lead time in the set. Farm A test
    event 45 does exactly this. Pass None for the uncapped view and a finite
    value (e.g. 72.0) for the operationally meaningful one.

    detected == False means the model never crossed the threshold inside the
    search window; lead_time_hours is NaN for those rows and must be reported
    as a miss rather than dropped.

    alarmed_in_horizon is the stricter check: did the model alarm at any point
    inside the labelled 48 h horizon? An event can be 'detected' with a long
    lead time and still be False here, which is the signature of a spurious
    early spike rather than a genuine early warning.
    """
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")

    faults = df[df["hours_to_fault"].notna()]
    rows = []

    for (asset_id, event_id), event in faults.groupby(["asset_id", "event_id"]):
        event = event.sort_values("hours_to_fault", ascending=False)

        if search_window_hours is not None:
            event = event[event["hours_to_fault"] <= search_window_hours]
            if event.empty:
                continue

        probs = event["probability"].to_numpy()
        hours = event["hours_to_fault"].to_numpy()
        above = (probs > threshold).astype(int)

        alarm_idx = _first_run(above, min_consecutive)

        in_horizon = event["hours_to_fault"] <= LABEL_HORIZON_HOURS
        horizon_probs = event.loc[in_horizon, "probability"].to_numpy()

        rows.append({
            "asset_id": asset_id,
            "event_id": event_id,
            "n_windows": len(event),
            "detected": alarm_idx is not None,
            "lead_time_hours": float(hours[alarm_idx]) if alarm_idx is not None else np.nan,
            "within_label_horizon": (
                bool(hours[alarm_idx] <= LABEL_HORIZON_HOURS)
                if alarm_idx is not None else False
            ),
            "alarmed_in_horizon": bool(
                _first_run((horizon_probs > threshold).astype(int), min_consecutive) is not None
            ) if horizon_probs.size else False,
            "max_probability": float(probs.max()),
            "max_probability_in_horizon": float(horizon_probs.max()) if horizon_probs.size else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["asset_id", "event_id"]).reset_index(drop=True)


def _first_run(flags: np.ndarray, length: int) -> int | None:
    """Index of the first position starting a run of `length` ones."""
    if length == 1:
        hits = np.flatnonzero(flags)
        return int(hits[0]) if hits.size else None

    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run == length:
            return i - length + 1
    return None


def lead_time_summary(per_event: pd.DataFrame) -> dict:
    """Aggregate lead_time() output. Reports n so a tiny sample is visible."""
    detected = per_event[per_event["detected"]]
    return {
        "n_fault_events": int(len(per_event)),
        "n_detected": int(len(detected)),
        "detection_rate": float(len(detected) / len(per_event)) if len(per_event) else np.nan,
        "mean_lead_time_hours": float(detected["lead_time_hours"].mean()) if len(detected) else np.nan,
        "median_lead_time_hours": float(detected["lead_time_hours"].median()) if len(detected) else np.nan,
        "min_lead_time_hours": float(detected["lead_time_hours"].min()) if len(detected) else np.nan,
        "max_lead_time_hours": float(detected["lead_time_hours"].max()) if len(detected) else np.nan,
        "n_beyond_label_horizon": int((~detected["within_label_horizon"]).sum()) if len(detected) else 0,
    }


def false_alarm_rate(df: pd.DataFrame, threshold: float,
                     min_consecutive: int = 1,
                     hours_per_window: float = 1.0) -> dict:
    """
    Alarm behaviour on events that contain no fault at all.

    Lead time on its own rewards a model that alarms constantly, so it must be
    read next to this. Normal events are those with hours_to_fault NaN.

    Two views are reported:

      window rate   - fraction of normal windows above threshold. Comparable to
                      1 - specificity.
      episodes      - runs of consecutive alarm windows collapsed to one. This
                      is the maintenance-relevant count, because a crew is
                      dispatched once per sustained alarm, not once per
                      10-minute timestep.

    hours_per_window converts episode counts into a per-1000-hour rate. The
    Farm A export uses a 1 h stride, so the default of 1.0 is correct there;
    pass 1/6 for per-timestep GRU predictions at 10-minute resolution.
    """
    normal = df[df["hours_to_fault"].isna()]

    if normal.empty:
        return {
            "n_normal_events": 0,
            "n_normal_windows": 0,
            "n_alarm_windows": 0,
            "window_alarm_rate": np.nan,
            "n_alarm_episodes": 0,
            "episodes_per_1000h": np.nan,
            "n_events_with_alarm": 0,
            "event_alarm_rate": np.nan,
        }

    total_windows = 0
    total_alarm_windows = 0
    total_episodes = 0
    events_with_alarm = 0

    for _, event in normal.groupby(["asset_id", "event_id"]):
        flags = (event["probability"].to_numpy() > threshold).astype(int)
        episodes = _count_runs(flags, min_consecutive)

        total_windows += len(flags)
        total_alarm_windows += int(flags.sum())
        total_episodes += episodes
        events_with_alarm += int(episodes > 0)

    exposure_hours = total_windows * hours_per_window

    return {
        "n_normal_events": int(normal.groupby(["asset_id", "event_id"]).ngroups),
        "n_normal_windows": int(total_windows),
        "n_alarm_windows": int(total_alarm_windows),
        "window_alarm_rate": float(total_alarm_windows / total_windows),
        "n_alarm_episodes": int(total_episodes),
        "episodes_per_1000h": float(total_episodes * 1000.0 / exposure_hours) if exposure_hours else np.nan,
        "n_events_with_alarm": int(events_with_alarm),
        "event_alarm_rate": float(events_with_alarm / normal.groupby(["asset_id", "event_id"]).ngroups),
    }


def _count_runs(flags: np.ndarray, length: int) -> int:
    """Number of distinct runs of at least `length` consecutive ones."""
    count = 0
    run = 0
    for flag in flags:
        if flag:
            run += 1
            if run == length:
                count += 1
        else:
            run = 0
    return count


def operating_point(df: pd.DataFrame, threshold: float,
                    min_consecutive: int = 1,
                    hours_per_window: float = 1.0) -> dict:
    """
    Lead time and false-alarm behaviour at one threshold, in a single row.

    Sweeping this across thresholds is what belongs in the Results section:
    it shows the earliness/nuisance trade-off directly, which neither metric
    shows alone.
    """
    per_event = lead_time(df, threshold, min_consecutive=min_consecutive)
    lead = lead_time_summary(per_event)
    false_alarms = false_alarm_rate(
        df, threshold,
        min_consecutive=min_consecutive,
        hours_per_window=hours_per_window,
    )

    return {
        "threshold": float(threshold),
        "min_consecutive": int(min_consecutive),
        "n_fault_events": lead["n_fault_events"],
        "detection_rate": lead["detection_rate"],
        "median_lead_time_hours": lead["median_lead_time_hours"],
        "mean_lead_time_hours": lead["mean_lead_time_hours"],
        "window_alarm_rate": false_alarms["window_alarm_rate"],
        "episodes_per_1000h": false_alarms["episodes_per_1000h"],
        "event_alarm_rate": false_alarms["event_alarm_rate"],
    }


def operating_point_sweep(df: pd.DataFrame, thresholds: np.ndarray,
                          min_consecutive: int = 1,
                          hours_per_window: float = 1.0) -> pd.DataFrame:
    """operating_point() across a threshold grid, one row per threshold."""
    return pd.DataFrame([
        operating_point(df, tau, min_consecutive, hours_per_window)
        for tau in thresholds
    ])


# ---------------------------------------------------------------------------
# Task 2 - per-turbine / per-farm breakdown
# ---------------------------------------------------------------------------

def classification_metrics(labels: np.ndarray, probs: np.ndarray,
                           threshold: float) -> dict:
    """Threshold-free and thresholded metrics. NaN where a class is absent."""
    preds = (probs > threshold).astype(int)
    both_classes = len(np.unique(labels)) == 2

    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probs)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(labels, probs)) if both_classes else np.nan,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def breakdown(df: pd.DataFrame, threshold: float,
              by: str = "asset_id") -> pd.DataFrame:
    """
    Per-group metrics plus a pooled row.

    by='asset_id' answers the per-turbine question. Add a 'farm' column
    upstream and pass by='farm' for the onshore/offshore comparison.

    A group holding only one class gets NaN for roc_auc and pr_auc rather than
    an error. On the Farm A test split every row is asset 13, so this returns
    one group row and reduces to the pooled numbers by construction.
    """
    if by not in df.columns:
        raise ValueError(f"grouping column '{by}' not in the table")

    rows = []
    for key, group in df.groupby(by):
        metrics = classification_metrics(
            group["label"].to_numpy(),
            group["probability"].to_numpy(),
            threshold,
        )
        rows.append({by: key, **metrics})

    pooled = classification_metrics(
        df["label"].to_numpy(), df["probability"].to_numpy(), threshold
    )
    rows.append({by: "ALL", **pooled})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 3 - cost-sensitive analysis
# ---------------------------------------------------------------------------

def cost_curve(df: pd.DataFrame, cost_fn: float = 10.0, cost_fp: float = 1.0,
               thresholds: np.ndarray | None = None,
               fp_cost_by_asset: dict | None = None) -> pd.DataFrame:
    """
    Total weighted cost across a threshold sweep.

    cost_fn / cost_fp are a documented ratio, not currency. The default 10:1
    says a missed fault costs ten unnecessary dispatches; state the reasoning
    in the paper and keep it consistent with Kamal's AccessCost term.

    fp_cost_by_asset maps asset_id -> false-positive cost, overriding cost_fp
    per turbine. This is where the offshore access premium goes: offshore
    turbines get a higher false-positive cost than onshore ones, because
    sending a crew out costs more.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    labels = df["label"].to_numpy()
    probs = df["probability"].to_numpy()

    if fp_cost_by_asset is None:
        fp_weights = np.full(len(df), float(cost_fp))
    else:
        assets = df["asset_id"].to_numpy()
        missing = set(np.unique(assets)) - set(fp_cost_by_asset)
        if missing:
            raise ValueError(f"fp_cost_by_asset has no entry for assets: {sorted(missing)}")
        fp_weights = np.array([fp_cost_by_asset[a] for a in assets], dtype=float)

    rows = []
    for tau in thresholds:
        preds = (probs > tau).astype(int)
        fn = (labels == 1) & (preds == 0)
        fp = (labels == 0) & (preds == 1)

        rows.append({
            "threshold": float(tau),
            "n_fn": int(fn.sum()),
            "n_fp": int(fp.sum()),
            "n_tp": int(((labels == 1) & (preds == 1)).sum()),
            "n_tn": int(((labels == 0) & (preds == 0)).sum()),
            "cost_fn_total": float(fn.sum() * cost_fn),
            "cost_fp_total": float(fp_weights[fp].sum()),
            "total_cost": float(fn.sum() * cost_fn + fp_weights[fp].sum()),
        })

    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> dict:
    """Lowest-cost row of a cost_curve(), as a plain dict."""
    return curve.loc[curve["total_cost"].idxmin()].to_dict()


# ---------------------------------------------------------------------------
# Task 4 - calibration
# ---------------------------------------------------------------------------

def brier_score(df: pd.DataFrame) -> float:
    """Mean squared error between predicted probability and label."""
    return float(np.mean((df["probability"].to_numpy() - df["label"].to_numpy()) ** 2))


def reliability_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Equal-width probability bins with predicted vs observed fault rate.

    Written out rather than using sklearn's calibration_curve because we also
    need the per-bin counts to weight the ECE, and empty bins should survive
    as rows so the reliability plot shows where the model never predicts.
    """
    probs = df["probability"].to_numpy()
    labels = df["label"].to_numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": count,
            "mean_predicted": float(probs[sel].mean()) if count else np.nan,
            "observed_rate": float(labels[sel].mean()) if count else np.nan,
        })

    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean gap between predicted and observed rate."""
    filled = table[table["count"] > 0]
    if filled.empty:
        return float("nan")
    gaps = (filled["mean_predicted"] - filled["observed_rate"]).abs()
    return float(np.average(gaps, weights=filled["count"]))


def calibration_report(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Brier, ECE and the reliability table in one call."""
    table = reliability_table(df, n_bins=n_bins)
    return {
        "brier_score": brier_score(df),
        "ece": expected_calibration_error(table),
        "n_bins_populated": int((table["count"] > 0).sum()),
        "reliability_table": table,
    }