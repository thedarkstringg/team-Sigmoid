"""
Cross-farm checkpoint/data compatibility check and evaluation entry point.

Verifies that a Farm C-trained checkpoint's input dimension matches the target
farm's exported sequence feature dimension before delegating to the existing
evaluator in ``src/model/evaluate.py``. Fails loudly on mismatch instead of
relying on a later ``load_state_dict`` shape error.

Supports both 10-feature (strict zero-shot) and 11-feature (with
``power_residual``) checkpoints through the same path: ``input_size`` is
always inferred from the checkpoint weights and the target data, never
hard-coded per farm.

Usage from repo root:

    python -m src.eval.cross_farm_evaluate \
        --checkpoint checkpoints/farm_c_power_residual/best.pt \
        --data_dir data/processed/CARE_Farm_A/physical_sequences
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from src.eval.export_gru_predictions import checkpoint_state, infer_model_shape
except ModuleNotFoundError:  # Support direct execution from src/eval.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_gru_predictions import checkpoint_state, infer_model_shape


def load_checkpoint_input_size(checkpoint_path: Path) -> tuple[int, int, int]:
    """Return (input_size, hidden_size, num_layers) inferred from checkpoint weights."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint_state(checkpoint)
    return infer_model_shape(state)


def load_target_input_size(data_dir: Path) -> int:
    """
    Return the feature dimension F from the target sequence directory.

    Prefers ``val_X.npy``; falls back to ``test_X.npy`` for external-validation
    exports (e.g. Farm A/B zero-shot targets) that intentionally contain only
    a test split and no val files at all.
    """
    for filename in ("val_X.npy", "test_X.npy"):
        x_path = data_dir / filename
        if x_path.is_file():
            x = np.load(x_path, mmap_mode="r")
            if x.ndim != 3:
                raise ValueError(f"Expected 3D {filename} (N, T, F), got shape {x.shape}")
            return int(x.shape[-1])
    raise FileNotFoundError(
        f"Neither val_X.npy nor test_X.npy found in {data_dir}"
    )


def check_compatibility(checkpoint_path: Path, data_dir: Path) -> tuple[int, int, bool]:
    """
    Check checkpoint/data feature-dimension compatibility.

    Returns (checkpoint_input_size, target_input_size, compatible).
    Prints a one-line compatibility report and raises on mismatch.
    """
    ckpt_input, ckpt_hidden, ckpt_layers = load_checkpoint_input_size(checkpoint_path)
    target_input = load_target_input_size(data_dir)
    compatible = ckpt_input == target_input

    print(f"checkpoint input_size = {ckpt_input}")
    print(f"target data input_size = {target_input}")
    print(f"compatible = {compatible}")

    if not compatible:
        raise ValueError(
            f"Checkpoint input_size={ckpt_input} does not match target data "
            f"input_size={target_input} in {data_dir}. "
            f"Check that the target farm was exported with the same feature schema "
            f"(10-feature strict vs 11-feature power_residual) as the checkpoint's training data."
        )
    return ckpt_input, target_input, compatible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    check_compatibility(args.checkpoint, args.data_dir)

    # Delegate to the existing evaluator instead of duplicating metric logic.
    try:
        from src.model import evaluate as model_evaluate
    except ModuleNotFoundError:  # Support direct execution from src/eval.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
        import evaluate as model_evaluate

    eval_args = argparse.Namespace(
        checkpoint=str(args.checkpoint),
        data_dir=str(args.data_dir),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        threshold=args.threshold,
    )
    model_evaluate.main_with_args(eval_args)


if __name__ == "__main__":
    main()
