"""
Training loop for TemporalRiskModel.

Currently wired to DUMMY data (see `build_dummy_dataloaders`) so the full
pipeline (train -> checkpoint -> resume -> log) can be proven correct before
Ismayil's real windowed SCADA data lands. Swap `build_dummy_dataloaders()`
for the real data loader once `src/data/` produces windowed tensors -
everything else in this file should not need to change.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import TemporalRiskModel


# ---------------------------------------------------------------------------
# Reproducibility: fixed seeds (brief requirement - cross-track, §1)
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# TEMPORARY: dummy data, until Ismayil's src/data/ pipeline exists.
# Replace this function's body with the real loader. Keep the same
# return signature (train_loader, val_loader) so nothing downstream breaks.
# ---------------------------------------------------------------------------
def build_dummy_dataloaders(batch_size=32, seq_len=144, num_features=86,
                             n_train=256, n_val=64):
    x_train = torch.randn(n_train, seq_len, num_features)
    y_train = torch.randint(0, 2, (n_train, seq_len)).float()
    x_val = torch.randn(n_val, seq_len, num_features)
    y_val = torch.randint(0, 2, (n_val, seq_len)).float()

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: any shape, binary {0,1}. Computes pos_weight for
    BCEWithLogitsLoss as (num_negative / num_positive).
    IMPORTANT: call this on the TRAINING split only - never val/test,
    to avoid leaking split statistics into the loss function.
    """
    num_pos = labels.sum()
    num_neg = labels.numel() - num_pos
    if num_pos == 0:
        raise ValueError("No positive examples found - cannot compute pos_weight.")
    return (num_neg / num_pos).unsqueeze(0)


def train_epoch(model, loader, optimizer, criterion, device, scaler,
                 grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x_batch, y_batch) in enumerate(loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Mixed precision forward pass (brief §2.1: "cut memory")
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x_batch)
            loss = criterion(logits, y_batch) / grad_accum_steps

        scaler.scale(loss).backward()

        # Gradient accumulation (brief §2.1: keep effective batch large
        # under a smaller per-step batch size)
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    # Checkpoint frequently, be able to resume (brief §2.1: "protect your work")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5,
                         help="early stopping patience, in epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_path", type=str, default="logs/train_log.jsonl")
    parser.add_argument("--resume", action="store_true",
                         help="resume from checkpoint_dir/last.pt if present")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    # TODO: replace with real data loader once src/data/ exists.
    train_loader, val_loader = build_dummy_dataloaders(batch_size=args.batch_size)

    # --- pos_weight from TRAINING labels only ---
    all_train_labels = torch.cat([y for _, y in train_loader])
    pos_weight = compute_pos_weight(all_train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Model / optimizer ---
    model = TemporalRiskModel(input_size=86, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    last_ckpt_path = os.path.join(args.checkpoint_dir, "last.pt")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")

    if args.resume and os.path.exists(last_ckpt_path):
        completed_epoch, best_val_loss = load_checkpoint(last_ckpt_path, model, optimizer, device)
        start_epoch = completed_epoch + 1  # resume AFTER the last completed epoch
        print(f"Resumed after epoch {completed_epoch}, starting at epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")

    log_file = open(args.log_path, "a")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, scaler, args.grad_accum_steps)
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Log metrics to disk, not just stdout (brief §2.1)
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_sec": elapsed,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        print(log_entry)

        # Checkpoint every epoch (cheap insurance against window ending mid-run)
        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_loss)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    log_file.close()


if __name__ == "__main__":
    main()