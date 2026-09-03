from pathlib import Path
import argparse
import json

import numpy as np


EXPECTED = {
    "train": {"x_shape": (4357, 144, 54), "assets": [0, 10, 11]},
    "val": {"x_shape": (1092, 144, 54), "assets": [21]},
    "test": {"x_shape": (778, 144, 54), "assets": [13]},
}


def validate_split(data_dir, split):
    """Checks one exported split is safe to hand to the training pipeline."""
    x = np.load(data_dir / f"{split}_X.npy")
    y = np.load(data_dir / f"{split}_y.npy")
    mask = np.load(data_dir / f"{split}_mask.npy")

    expected_x = EXPECTED[split]["x_shape"]

    if x.shape != expected_x:
        raise ValueError(
            f"{split}_X shape {x.shape}, expected {expected_x}"
        )

    expected_seq_shape = expected_x[:2]

    if y.shape != expected_seq_shape:
        raise ValueError(
            f"{split}_y shape {y.shape}, expected {expected_seq_shape}"
        )

    if mask.shape != expected_seq_shape:
        raise ValueError(
            f"{split}_mask shape {mask.shape}, expected {expected_seq_shape}"
        )

    if x.dtype != np.float32:
        raise TypeError(f"{split}_X dtype is {x.dtype}, expected float32")

    if y.dtype != np.uint8:
        raise TypeError(f"{split}_y dtype is {y.dtype}, expected uint8")

    if mask.dtype != np.uint8:
        raise TypeError(f"{split}_mask dtype is {mask.dtype}, expected uint8")

    if not np.isfinite(x).all():
        raise ValueError(f"{split}_X contains NaN or inf")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{split}_y contains values outside {{0,1}}")

    if not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(f"{split}_mask contains values outside {{0,1}}")

    real = mask.astype(bool)

    real_labels = y[real]

    positives = int(real_labels.sum())
    negatives = int(real_labels.size - positives)

    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{split} must contain both positive and negative real timesteps"
        )

    masked_timesteps = int((mask == 0).sum())

    print(
        f"{split:5s} | "
        f"X={x.shape} {x.dtype} | "
        f"y={y.shape} {y.dtype} | "
        f"mask={mask.shape} {mask.dtype} | "
        f"positive={positives} | "
        f"negative={negatives} | "
        f"masked={masked_timesteps}"
    )


def validate_scaler(data_dir):
    """Checks train-only scaler metadata matches the 54-feature contract."""
    scaler = np.load(
        data_dir / "scaler_stats.npz",
        allow_pickle=False,
    )

    mean = scaler["mean"]
    std = scaler["std"]
    sensors = scaler["sensor_columns"]

    if mean.shape != (54,):
        raise ValueError(f"Scaler mean shape is {mean.shape}")

    if std.shape != (54,):
        raise ValueError(f"Scaler std shape is {std.shape}")

    if sensors.shape != (54,):
        raise ValueError(f"Sensor list shape is {sensors.shape}")

    if not np.isfinite(mean).all():
        raise ValueError("Scaler mean contains NaN or inf")

    if not np.isfinite(std).all():
        raise ValueError("Scaler std contains NaN or inf")

    if (std <= 0).any():
        raise ValueError("Scaler std contains non-positive values")

    print("scaler | 54 means/stds/sensor names OK")


def validate_summary(data_dir):
    """Checks exported metadata still matches the fixed asset split."""
    path = data_dir / "export_summary.json"

    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if summary["sequence_length"] != 144:
        raise ValueError("Expected sequence length 144")

    if summary["num_features"] != 54:
        raise ValueError("Expected 54 features")

    for split, expected in EXPECTED.items():
        actual_assets = summary["splits"][split]["assets"]

        if actual_assets != expected["assets"]:
            raise ValueError(
                f"{split} assets {actual_assets}, "
                f"expected {expected['assets']}"
            )

    train_assets = set(summary["splits"]["train"]["assets"])
    val_assets = set(summary["splits"]["val"]["assets"])
    test_assets = set(summary["splits"]["test"]["assets"])

    if train_assets & val_assets:
        raise ValueError("Train/val asset leakage")

    if train_assets & test_assets:
        raise ValueError("Train/test asset leakage")

    if val_assets & test_assets:
        raise ValueError("Val/test asset leakage")

    print("summary | asset-disjoint split OK")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    required = []

    for split in ("train", "val", "test"):
        required.extend(
            [
                args.data_dir / f"{split}_X.npy",
                args.data_dir / f"{split}_y.npy",
                args.data_dir / f"{split}_mask.npy",
            ]
        )

    required.extend(
        [
            args.data_dir / "scaler_stats.npz",
            args.data_dir / "export_summary.json",
        ]
    )

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing export files:\n" + "\n".join(missing)
        )

    print("=== VALIDATING FARM A SEQUENCE EXPORT ===")

    for split in ("train", "val", "test"):
        validate_split(args.data_dir, split)

    validate_scaler(args.data_dir)
    validate_summary(args.data_dir)

    print("\nALL SEQUENCE EXPORT CHECKS PASSED")


if __name__ == "__main__":
    main()
