"""Train reproducible per-timestep classical baselines on CARE Farm C.

The population in this script intentionally matches the GRU evaluation
population: exported physical-v1 sequences are clipped to [-10, 10], flattened
in sequence/timestep order, and only rows whose mask equals one are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "CARE_Farm_C" / "sequences"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baseline" / "farm_c"

FARM = "C"
SEQUENCE_LENGTH = 144
FEATURE_COUNT = 10
CLIP_BOUNDS = (-10.0, 10.0)
RANDOM_SEED = 42
SPLITS = ("train", "val", "test")
PREDICTION_SPLITS = ("val", "test")

PREDICTION_METADATA_COLUMNS = [
    "farm",
    "split",
    "sequence_idx",
    "timestep_idx",
    "asset_id",
    "event_id",
    "window_end",
    "fault_time",
    "hours_to_fault",
    "label",
]
REQUIRED_METADATA_COLUMNS = set(PREDICTION_METADATA_COLUMNS) | {"mask"}


@dataclass(frozen=True)
class PreparedSplit:
    """A validated split plus its mask==1 per-timestep population."""

    name: str
    input_shape: tuple[int, ...]
    X: np.ndarray
    y: np.ndarray
    full_y: np.ndarray
    full_mask: np.ndarray
    valid_selector: np.ndarray


def _require_binary(values: np.ndarray, description: str) -> None:
    """Reject missing, infinite, or non-binary labels and masks."""

    array = np.asarray(values)
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(f"{description} must be numeric and binary") from exc
    if not finite.all() or not set(np.unique(array)).issubset({0, 1}):
        raise ValueError(f"{description} must contain only binary values 0 and 1")


def flatten_valid_timesteps(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    clip_bounds: tuple[float, float] = CLIP_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip, flatten, and retain exactly the mask==1 timestep rows."""

    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.asarray(mask)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (N,T,F), got {X.shape}")
    if y.shape != X.shape[:2] or mask.shape != X.shape[:2]:
        raise ValueError(
            f"X/y/mask shapes do not align: X={X.shape}, y={y.shape}, mask={mask.shape}"
        )
    _require_binary(y, "labels")
    _require_binary(mask, "mask")

    clipped = np.clip(X, clip_bounds[0], clip_bounds[1])
    flat_selector = mask.reshape(-1) == 1
    flat_X = clipped.reshape(-1, X.shape[-1])[flat_selector]
    flat_y = y.reshape(-1)[flat_selector].astype(np.uint8, copy=False)
    if not len(flat_y):
        raise ValueError("No mask==1 timesteps remain after filtering")
    if not np.isfinite(flat_X).all():
        raise ValueError("NaN or infinite feature values would reach sklearn")
    return flat_X, flat_y, flat_selector


def load_split(data_dir: Path, split: str) -> PreparedSplit:
    """Load one physical-v1 split and enforce the Farm C array contract."""

    X = np.load(data_dir / f"{split}_X.npy", allow_pickle=False)
    y = np.load(data_dir / f"{split}_y.npy", allow_pickle=False)
    mask = np.load(data_dir / f"{split}_mask.npy", allow_pickle=False)
    expected_tail = (SEQUENCE_LENGTH, FEATURE_COUNT)
    if X.ndim != 3 or X.shape[1:] != expected_tail:
        raise ValueError(
            f"{split} X shape {X.shape}; expected (N,{SEQUENCE_LENGTH},{FEATURE_COUNT})"
        )
    flat_X, flat_y, selector = flatten_valid_timesteps(X, y, mask)
    return PreparedSplit(
        name=split,
        input_shape=tuple(int(value) for value in X.shape),
        X=flat_X,
        y=flat_y,
        full_y=y,
        full_mask=mask,
        valid_selector=selector,
    )


def validate_and_filter_metadata(
    metadata: pd.DataFrame,
    y: np.ndarray,
    mask: np.ndarray,
    split: str,
) -> pd.DataFrame:
    """Prove metadata/array row alignment, then apply the exact flattened mask."""

    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"{split} metadata missing columns: {sorted(missing)}")
    expected_rows = y.shape[0] * SEQUENCE_LENGTH
    if y.ndim != 2 or y.shape[1] != SEQUENCE_LENGTH or mask.shape != y.shape:
        raise ValueError(f"{split} y/mask do not have aligned (N,{SEQUENCE_LENGTH}) shapes")
    if len(metadata) != expected_rows:
        raise ValueError(
            f"{split} metadata rows {len(metadata)} != {y.shape[0]} * {SEQUENCE_LENGTH}"
        )

    expected_sequence = np.repeat(np.arange(y.shape[0]), SEQUENCE_LENGTH)
    expected_timestep = np.tile(np.arange(SEQUENCE_LENGTH), y.shape[0])
    if not np.array_equal(metadata["sequence_idx"].to_numpy(), expected_sequence):
        raise ValueError(f"{split} metadata sequence_idx is not aligned with flattened arrays")
    if not np.array_equal(metadata["timestep_idx"].to_numpy(), expected_timestep):
        raise ValueError(f"{split} metadata timestep_idx is not aligned with flattened arrays")
    if not metadata["farm"].astype(str).str.upper().eq(FARM).all():
        raise ValueError(f"{split} metadata is not Farm C only")
    if not metadata["split"].astype(str).eq(split).all():
        raise ValueError(f"{split} metadata contains a different split")

    metadata_y = pd.to_numeric(metadata["label"], errors="raise").to_numpy()
    metadata_mask = pd.to_numeric(metadata["mask"], errors="raise").to_numpy()
    _require_binary(metadata_y, f"{split} metadata labels")
    _require_binary(metadata_mask, f"{split} metadata masks")
    if not np.array_equal(metadata_y.astype(np.uint8), y.reshape(-1).astype(np.uint8)):
        raise ValueError(f"{split} metadata labels do not align with y")
    if not np.array_equal(metadata_mask.astype(np.uint8), mask.reshape(-1).astype(np.uint8)):
        raise ValueError(f"{split} metadata masks do not align with mask array")

    filtered = metadata.loc[mask.reshape(-1) == 1, PREDICTION_METADATA_COLUMNS].copy()
    return filtered.reset_index(drop=True)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation probability threshold that maximizes F1."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def _fixed_threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.uint8)
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Calculate selected-threshold and threshold-independent metrics."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != y_true.shape:
        raise ValueError("Probability and label shapes do not match")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model probabilities contain NaN or infinite values")
    predictions = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n_evaluated_timesteps": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "fixed_threshold_0_5": {
            "threshold": 0.5,
            **_fixed_threshold_metrics(y_true, probabilities, 0.5),
        },
    }


def select_threshold_and_evaluate(
    validation_y: np.ndarray,
    validation_probabilities: np.ndarray,
    test_y: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Freeze a validation-only F1 threshold and apply it to val and test."""

    threshold = choose_threshold(validation_y, validation_probabilities)
    validation_metrics = evaluate_probabilities(
        validation_y, validation_probabilities, threshold
    )
    test_metrics = evaluate_probabilities(test_y, test_probabilities, threshold)
    return threshold, validation_metrics, test_metrics


def build_models(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Construct models without adding an imputer or a second scaler."""

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        ),
    }


def prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Attach model output to already mask-filtered, aligned metadata."""

    if len(metadata) != len(probabilities):
        raise ValueError("Prediction count does not match filtered metadata")
    result = metadata.copy()
    result["model"] = model_name
    result["probability"] = probabilities
    result["validation_selected_threshold"] = float(threshold)
    result["prediction"] = (probabilities >= threshold).astype(np.uint8)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_model_params(model: Any) -> dict[str, Any]:
    params = model.get_params(deep=False)
    return {
        key: value
        for key, value in params.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def run(data_dir: Path, output_dir: Path) -> None:
    """Load Farm C, fit both baselines, and write all requested artifacts."""

    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    splits = {split: load_split(data_dir, split) for split in SPLITS}
    for split, prepared in splits.items():
        if set(np.unique(prepared.y)) != {0, 1}:
            raise ValueError(f"{split} mask==1 population must contain both classes")

    metadata: dict[str, pd.DataFrame] = {}
    for split in PREDICTION_SPLITS:
        prepared = splits[split]
        source = pd.read_parquet(data_dir / f"{split}_metadata.parquet")
        metadata[split] = validate_and_filter_metadata(
            source, prepared.full_y, prepared.full_mask, split
        )
        if len(metadata[split]) != len(prepared.y):
            raise RuntimeError(f"{split} metadata mask did not produce the model population")

    output_dir.mkdir(parents=True, exist_ok=True)
    models = build_models()
    train_weights = compute_sample_weight(class_weight="balanced", y=splits["train"].y)
    all_metrics: dict[str, Any] = {}
    predictions: dict[str, list[pd.DataFrame]] = {split: [] for split in PREDICTION_SPLITS}
    thresholds: dict[str, float] = {}

    for model_name, model in models.items():
        if model_name == "hist_gradient_boosting":
            model.fit(splits["train"].X, splits["train"].y, sample_weight=train_weights)
        else:
            model.fit(splits["train"].X, splits["train"].y)

        val_probabilities = model.predict_proba(splits["val"].X)[:, 1]
        test_probabilities = model.predict_proba(splits["test"].X)[:, 1]
        threshold, val_metrics, test_metrics = select_threshold_and_evaluate(
            splits["val"].y,
            val_probabilities,
            splits["test"].y,
            test_probabilities,
        )
        thresholds[model_name] = threshold
        all_metrics[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }
        predictions["val"].append(
            prediction_frame(metadata["val"], val_probabilities, threshold, model_name)
        )
        predictions["test"].append(
            prediction_frame(metadata["test"], test_probabilities, threshold, model_name)
        )
        joblib.dump(model, output_dir / f"{model_name}.joblib")

    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    for split in PREDICTION_SPLITS:
        pd.concat(predictions[split], ignore_index=True).to_parquet(
            output_dir / f"{split}_predictions.parquet", index=False, compression="zstd"
        )

    input_files = [
        data_dir / f"{split}_{suffix}"
        for split in SPLITS
        for suffix in ("X.npy", "y.npy", "mask.npy")
    ] + [data_dir / f"{split}_metadata.parquet" for split in PREDICTION_SPLITS]
    manifest = {
        "schema_version": "farm-c-classical-baseline-v1",
        "farm": FARM,
        "farm_scope": "Farm C only",
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "timestep_population_policy": "mask==1",
        "clipping": [CLIP_BOUNDS[0], CLIP_BOUNDS[1]],
        "additional_feature_scaler_fitted": False,
        "threshold_selection": "validation F1",
        "threshold_comparison_operator": ">=",
        "random_seed": RANDOM_SEED,
        "input_directory": str(data_dir),
        "output_directory": str(output_dir),
        "input_shapes": {
            split: {
                "X": list(prepared.input_shape),
                "y": list(prepared.full_y.shape),
                "mask": list(prepared.full_mask.shape),
            }
            for split, prepared in splits.items()
        },
        "valid_row_counts": {
            split: int(len(prepared.y)) for split, prepared in splits.items()
        },
        "valid_class_counts": {
            split: {
                "negative": int(np.sum(prepared.y == 0)),
                "positive": int(np.sum(prepared.y == 1)),
            }
            for split, prepared in splits.items()
        },
        "models": {
            name: _json_model_params(model) for name, model in models.items()
        },
        "validation_selected_thresholds": thresholds,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "input_sha256": {path.name: _sha256(path) for path in input_files},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Saved Farm C baseline artifacts to {output_dir}")
    for model_name, model_metrics in all_metrics.items():
        print(f"\n{model_name} (threshold={thresholds[model_name]:.6f})")
        for split in PREDICTION_SPLITS:
            values = model_metrics["validation" if split == "val" else "test"]
            print(
                f"  {split}: ROC-AUC={values['roc_auc']:.4f} "
                f"PR-AUC={values['pr_auc']:.4f} F1={values['f1']:.4f} "
                f"n={values['n_evaluated_timesteps']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
