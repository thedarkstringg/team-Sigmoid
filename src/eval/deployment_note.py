"""
Deployment characteristics of a trained checkpoint: inference latency on CPU
and GPU, model size, and peak inference memory.

Track 2 asks for evaluation under realistic deployment conditions, so these
numbers belong in the Results section next to the predictive metrics. A SCADA
system produces one 144-timestep window per turbine per hour, so any latency
in the millisecond range is operationally irrelevant — the point of reporting
it is to show the model is deployable on ordinary hardware, not to optimise it.

Usage:
    python src/eval/deployment_note.py \
        --checkpoint checkpoints/h32_dropout/best.pt \
        --hidden_size 32 --dropout 0.3

Peak TRAINING memory is not measurable here — it needs nvidia-smi during an
actual training run. Capture it separately and add it to the writeup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# src/model is a sibling package; make TemporalRiskModel importable whether this
# runs from the repo root or from inside src/eval.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "model"))

from model import TemporalRiskModel  # noqa: E402

SEQ_LEN = 144
INPUT_SIZE = 54


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def benchmark(model: torch.nn.Module, device: torch.device, batch_size: int,
              repeats: int, warmup: int) -> dict:
    """Median wall-clock latency for one forward pass at the given batch size."""
    model = model.to(device).eval()
    x = torch.randn(batch_size, SEQ_LEN, INPUT_SIZE, device=device)

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize(device)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    median_ms = timings[len(timings) // 2]

    result = {
        "device": str(device),
        "batch_size": batch_size,
        "median_batch_latency_ms": round(median_ms, 4),
        "median_per_sequence_latency_ms": round(median_ms / batch_size, 4),
        "min_batch_latency_ms": round(timings[0], 4),
        "max_batch_latency_ms": round(timings[-1], 4),
    }

    if is_cuda:
        result["peak_inference_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3
        )

    return result


def measure_training_memory(model: torch.nn.Module, device: torch.device,
                            batch_size: int, steps: int = 5) -> dict:
    """
    Peak GPU memory during training, measured with torch's own allocator.

    Synthetic tensors of the correct shape are used rather than real sequences.
    Peak memory is determined by architecture, batch size and sequence length,
    not by the values in the data, so this gives the same answer as a real run
    while needing no data files. Say so in the writeup.

    Preferred over nvidia-smi on a shared GPU: nvidia-smi reports whole-device
    usage including the CUDA context and any other user's job, whereas these
    counters are scoped to this process.
    """
    if device.type != "cuda":
        return {"available": False, "reason": "training memory requires a CUDA device"}

    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    x = torch.randn(batch_size, SEQ_LEN, INPUT_SIZE, device=device)
    y = torch.randint(0, 2, (batch_size, SEQ_LEN), device=device).float()
    mask = torch.ones(batch_size, SEQ_LEN, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = (criterion(logits, y) * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device)

    result = {
        "available": True,
        "batch_size": batch_size,
        "steps": steps,
        "peak_training_memory_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3
        ),
        "peak_training_memory_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2), 3
        ),
    }

    model.eval()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/h32_dropout/best.pt")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="must match the value used when this checkpoint was trained")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 64],
                        help="1 = single-turbine inference, 64 = fleet-scale batch")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="should match train.py's --batch_size for a comparable number")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=str, default="artifacts/deployment_note.json")
    args = parser.parse_args()

    model = TemporalRiskModel(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_size_bytes": os.path.getsize(args.checkpoint),
        "checkpoint_size_kb": round(os.path.getsize(args.checkpoint) / 1024, 1),
        "config": {
            "input_size": INPUT_SIZE,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_length": SEQ_LEN,
        },
        "trained_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "torch_version": torch.__version__,
        **count_parameters(model),
        "benchmarks": [],
    }

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        devices.append(cuda)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda)
        report["gpu_total_memory_mb"] = round(total_bytes / (1024 ** 2), 1)
        report["gpu_free_memory_mb_at_start"] = round(free_bytes / (1024 ** 2), 1)
    else:
        report["gpu_name"] = None
        print("WARNING: CUDA not available - GPU latency will be missing from the report.")

    for device in devices:
        for batch_size in args.batch_sizes:
            report["benchmarks"].append(
                benchmark(model, device, batch_size, args.repeats, args.warmup)
            )

    if torch.cuda.is_available():
        report["training_memory"] = measure_training_memory(
            model, torch.device("cuda"), args.train_batch_size
        )
    else:
        report["training_memory"] = {"available": False, "reason": "no CUDA device"}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()