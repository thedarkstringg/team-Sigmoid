from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.environ.get(
        "CARE_FARM_A_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "processed" / "CARE_Farm_A"),
    )
)
WINDOW_FILE = DATA_DIR / "farm_a_windows.csv"
FEATURE_FILE = DATA_DIR / "feature_manifest.csv"

RANDOM_STATE = 42


def choose_threshold(y_true, probabilities):
    """Picks the validation threshold that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, probabilities, threshold):
    """Calculates the classification metrics we need for model comparison."""
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prediction_frame(
    source,
    probabilities,
    threshold,
    model_name,
    split_name,
):
    """Packages probabilities and decisions so later evaluation is reproducible."""
    result = source[
        [
            "event_id",
            "asset_id",
            "window_end",
            "hours_to_fault",
            "label",
        ]
    ].copy()

    result["model"] = model_name
    result["split"] = split_name
    result["probability"] = probabilities
    result["threshold"] = threshold
    result["prediction"] = (
        probabilities >= threshold
    ).astype(int)

    return result


def main():
    df = pd.read_csv(
        WINDOW_FILE,
        parse_dates=["window_end"],
    )

    feature_cols = (
        pd.read_csv(FEATURE_FILE)["feature"]
        .tolist()
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    X_train = train[feature_cols]
    y_train = train["label"].astype(int)

    X_val = val[feature_cols]
    y_val = val["label"].astype(int)

    X_test = test[feature_cols]
    y_test = test["label"].astype(int)

    print("=== BASELINE TRAINING ===")
    print(f"Features: {len(feature_cols)}")
    print(
        f"Train: {len(train)} "
        f"(positive={y_train.sum()}, "
        f"negative={(y_train == 0).sum()})"
    )
    print(
        f"Validation: {len(val)} "
        f"(positive={y_val.sum()}, "
        f"negative={(y_val == 0).sum()})"
    )
    print(
        f"Test: {len(test)} "
        f"(positive={y_test.sum()}, "
        f"negative={(y_test == 0).sum()})"
    )

    imbalance_ratio = (
        (y_train == 0).sum()
        / y_train.sum()
    )

    print(
        f"Train negative:positive ratio = "
        f"{imbalance_ratio:.2f}:1"
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    all_metrics = {}
    all_predictions = []

    for model_name, model in models.items():
        print(f"\n=== {model_name.upper()} ===")

        if model_name == "hist_gradient_boosting":
            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=balanced_weights,
            )
        else:
            model.fit(
                X_train,
                y_train,
            )

        val_prob = model.predict_proba(X_val)[:, 1]

        threshold = choose_threshold(
            y_val,
            val_prob,
        )

        test_prob = model.predict_proba(X_test)[:, 1]

        val_metrics = evaluate(
            y_val,
            val_prob,
            threshold,
        )

        test_metrics = evaluate(
            y_test,
            test_prob,
            threshold,
        )

        all_metrics[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        print(
            f"Validation-selected threshold: "
            f"{threshold:.4f}"
        )

        print("\nValidation:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value}")

        print("\nTest:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value}")

        all_predictions.append(
            prediction_frame(
                val,
                val_prob,
                threshold,
                model_name,
                "val",
            )
        )

        all_predictions.append(
            prediction_frame(
                test,
                test_prob,
                threshold,
                model_name,
                "test",
            )
        )

        joblib.dump(
            model,
            DATA_DIR / f"{model_name}.joblib",
        )

    with open(
        DATA_DIR / "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_metrics,
            f,
            indent=2,
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions.to_csv(
        DATA_DIR / "baseline_predictions.csv",
        index=False,
    )

    print("\n=== SAVED ===")
    print(DATA_DIR / "baseline_metrics.json")
    print(DATA_DIR / "baseline_predictions.csv")
    print(DATA_DIR / "logistic_regression.joblib")
    print(DATA_DIR / "hist_gradient_boosting.joblib")


if __name__ == "__main__":
    main()
