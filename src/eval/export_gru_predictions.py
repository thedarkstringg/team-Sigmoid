"""Export aligned per-timestep GRU probabilities without choosing a threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from src.model.model import TemporalRiskModel
except ModuleNotFoundError:  # Support direct execution from src/eval.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from model import TemporalRiskModel


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def infer_model_shape(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    try:
        input_weight = state["gru.weight_ih_l0"]
        hidden_weight = state["gru.weight_hh_l0"]
    except KeyError as exc:
        raise ValueError("Checkpoint is not a TemporalRiskModel GRU checkpoint") from exc
    input_size = int(input_weight.shape[1])
    hidden_size = int(hidden_weight.shape[1])
    layers = 0
    while f"gru.weight_ih_l{layers}" in state:
        layers += 1
    return input_size, hidden_size, layers


def validate_metadata(metadata: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> None:
    required = {
        "sequence_idx", "timestep_idx", "asset_id", "event_id", "window_end", "label", "mask"
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata columns missing: {sorted(missing)}")
    n, timesteps = y.shape
    if timesteps != 144 or mask.shape != y.shape or len(metadata) != n * timesteps:
        raise ValueError("Metadata/y/mask shapes do not satisfy N * 144")
    expected_sequence = np.repeat(np.arange(n), timesteps)
    expected_timestep = np.tile(np.arange(timesteps), n)
    if not np.array_equal(metadata["sequence_idx"].to_numpy(), expected_sequence):
        raise ValueError("Metadata is not in exact sequence_idx-major array order")
    if not np.array_equal(metadata["timestep_idx"].to_numpy(), expected_timestep):
        raise ValueError("Metadata timestep_idx is not 0..143 for every sequence")
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.uint8), y.reshape(-1).astype(np.uint8)):
        raise ValueError("Metadata labels do not align with the y array")
    if not np.array_equal(metadata["mask"].to_numpy(dtype=np.uint8), mask.reshape(-1).astype(np.uint8)):
        raise ValueError("Metadata masks do not align with the mask array")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--metadata-parquet", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    x = np.load(args.data_dir / f"{args.split}_X.npy", mmap_mode="r")
    y = np.load(args.data_dir / f"{args.split}_y.npy")
    mask = np.load(args.data_dir / f"{args.split}_mask.npy")
    if x.ndim != 3 or x.shape[:2] != y.shape or x.shape[1] != 144:
        raise ValueError(f"Unexpected array shapes: X={x.shape}, y={y.shape}, mask={mask.shape}")
    metadata = pd.read_parquet(args.metadata_parquet)
    validate_metadata(metadata, y, mask)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint_state(checkpoint)
    inferred_input, inferred_hidden, inferred_layers = infer_model_shape(state)
    input_size = args.input_size or inferred_input
    hidden_size = args.hidden_size or inferred_hidden
    num_layers = args.num_layers or inferred_layers
    if input_size != x.shape[-1]:
        raise ValueError(f"Model input_size={input_size}, but X has F={x.shape[-1]}")
    if (input_size, hidden_size, num_layers) != (inferred_input, inferred_hidden, inferred_layers):
        raise ValueError(
            "Supplied model dimensions disagree with checkpoint: "
            f"supplied={(input_size, hidden_size, num_layers)}, "
            f"checkpoint={(inferred_input, inferred_hidden, inferred_layers)}"
        )
    model = TemporalRiskModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    probability_chunks = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(np.asarray(x[start:start + args.batch_size])).float().to(device)
            probability_chunks.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32))
    probabilities = np.concatenate(probability_chunks, axis=0)
    if probabilities.shape != y.shape:
        raise RuntimeError(f"Prediction shape {probabilities.shape} != labels {y.shape}")
    columns = ["sequence_idx", "timestep_idx", "asset_id", "event_id", "window_end"]
    for optional in ("farm", "fault_time", "hours_to_fault"):
        if optional in metadata.columns:
            columns.append(optional)
    output = metadata[columns].copy()
    output["prob"] = probabilities.reshape(-1)
    output["label"] = y.reshape(-1).astype(np.uint8)
    output["mask"] = mask.reshape(-1).astype(np.uint8)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_parquet, index=False, compression="zstd")
    print(
        f"Wrote {len(output)} aligned rows ({len(x)} sequences) to "
        f"{args.output_parquet}; no threshold was fitted or applied."
    )


if __name__ == "__main__":
    main()

