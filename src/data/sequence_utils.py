"""Shared sequence, metadata, and scaler utilities for CARE exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INTERVAL_MINUTES = 10
LOOKBACK_HOURS = 24
HORIZON_HOURS = 48
STRIDE_HOURS = 1
SEQ_LEN = 144
MIN_COVERAGE = 0.95
MAX_GAP_MINUTES = 30


def detect_care_delimiter(path: str | Path) -> str:
    """Detect the delimiter in a CARE CSV header."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"Cannot detect a delimiter in empty CARE CSV: {csv_path}")
    try:
        return csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
    except csv.Error as exc:
        raise ValueError(
            f"Could not detect a comma, semicolon, or tab delimiter in CARE CSV: {csv_path}"
        ) from exc


def read_care_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CARE CSV using the delimiter detected from its header."""
    separator = detect_care_delimiter(path)
    return pd.read_csv(path, sep=separator, **kwargs)


def timestep_labels(index: pd.DatetimeIndex, event_label: str, event_end: pd.Timestamp) -> np.ndarray:
    if event_label != "anomaly":
        return np.zeros(len(index), dtype=np.uint8)
    hours = (event_end - index).total_seconds() / 3600.0
    return np.asarray((hours > 0) & (hours <= HORIZON_HOURS), dtype=np.uint8)


def build_timestep_metadata(
    *,
    farm: str,
    split: str,
    sequence_idx: int,
    index: pd.DatetimeIndex,
    asset_id: Any,
    event_id: int,
    event_label: str,
    event_end: pd.Timestamp,
    labels: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    if not (len(index) == len(labels) == len(mask) == SEQ_LEN):
        raise ValueError("Metadata inputs must all have sequence length 144")
    fault_time = event_end if event_label == "anomaly" else pd.NaT
    if event_label == "anomaly":
        hours_to_fault = (event_end - index).total_seconds() / 3600.0
    else:
        hours_to_fault = np.full(SEQ_LEN, np.nan)
    return pd.DataFrame(
        {
            "farm": farm.upper(),
            "split": split,
            "sequence_idx": np.full(SEQ_LEN, sequence_idx, dtype=np.int64),
            "timestep_idx": np.arange(SEQ_LEN, dtype=np.int16),
            "asset_id": asset_id,
            "event_id": np.full(SEQ_LEN, event_id, dtype=np.int64),
            "window_end": index,
            "event_end": event_end,
            "fault_time": fault_time,
            "hours_to_fault": hours_to_fault,
            "label": labels.astype(np.uint8),
            "mask": mask.astype(np.uint8),
        }
    )


def fit_train_scaler(train_x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_x.ndim != 3 or train_mask.shape != train_x.shape[:2]:
        raise ValueError("Scaler expects X=(N,T,F) and mask=(N,T)")
    real = train_x.reshape(-1, train_x.shape[-1])[train_mask.reshape(-1).astype(bool)]
    if not len(real):
        raise ValueError("Cannot fit scaler without valid Farm C train timesteps")
    mean = real.mean(axis=0).astype(np.float32)
    std = real.std(axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Non-finite Farm C train scaler statistics")
    std[std < 1.0e-6] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.shape[-1] != len(mean) or mean.shape != std.shape:
        raise ValueError("Scaler dimension does not match X")
    result = ((x - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Scaling produced NaN or infinite values")
    return result


def save_scaler(
    path: str | Path, mean: np.ndarray, std: np.ndarray, feature_names: list[str]
) -> None:
    np.savez(
        path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=np.asarray(feature_names),
        source_farm=np.asarray("C"),
        source_split=np.asarray("train"),
        fit_mask=np.asarray("real_and_physical_valid_timesteps_only"),
    )


def load_scaler(path: str | Path, expected_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as scaler:
        names = scaler["feature_names"].astype(str).tolist()
        source_farm = str(scaler["source_farm"].item())
        source_split = str(scaler["source_split"].item())
        mean = scaler["mean"].astype(np.float32)
        std = scaler["std"].astype(np.float32)
    if names != expected_features:
        raise ValueError(f"Scaler feature order {names} != expected {expected_features}")
    if (source_farm, source_split) != ("C", "train"):
        raise ValueError("Scaler provenance must be Farm C train")
    return mean, std


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_passing_frequency_audit(path: str | Path, farm: str) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("farm") != farm.upper():
        raise ValueError(f"Audit report is for Farm {report.get('farm')}, not {farm}")
    validation = report.get("frequency_validation", {})
    if not validation.get("passed", False):
        raise ValueError("Frequency audit did not confirm a 50 Hz-centered signal")
    return report
