"""
Training loop for TemporalRiskModel.
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
def build_dummy_dataloaders(batch_size=32, seq_len=144, num_features=54,
                             n_train=256, n_val=64):
    x_train = torch.randn(n_train, seq_len, num_features)
    y_train = torch.randint(0, 2, (n_train, seq_len)).float()
    mask_train = torch.ones(n_train, seq_len)  # dummy data has no gaps
    x_val = torch.randn(n_val, seq_len, num_features)
    y_val = torch.randint(0, 2, (n_val, seq_len)).float()
    mask_val = torch.ones(n_val, seq_len)

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, mask_val),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_real_dataloaders(data_dir="data/processed/CARE_Farm_A/sequences",
                            batch_size=32):
    """
    Loads exported .npy files per the agreed contract:
      train_X.npy (N,144,54) float32, train_y.npy (N,144) uint8,
      train_mask.npy (N,144) - 1=real observation, 0=gap-filled timestep.
      Same for val_/test_.
    """
    def _load_split(split):
        x = np.load(os.path.join(data_dir, f"{split}_X.npy"))
        y = np.load(os.path.join(data_dir, f"{split}_y.npy"))
        mask = np.load(os.path.join(data_dir, f"{split}_mask.npy"))
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(mask).float())

    x_train, y_train, mask_train = _load_split("train")
    x_val, y_val, mask_val = _load_split("val")

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, mask_val),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(labels: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    labels: any shape, binary {0,1}. Computes pos_weight for
    BCEWithLogitsLoss as (num_negative / num_positive).
    IMPORTANT: call this on the TRAINING split only - never val/test,
    to avoid leaking split statistics into the loss function.
    If mask is given, gap-filled timesteps (mask==0) are excluded so the
    ratio reflects only real observations.
    """
    if mask is not None:
        labels = labels[mask.bool()]
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

    for step, (x_batch, y_batch, mask_batch) in enumerate(loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        # Mixed precision forward pass (brief §2.1: "cut memory")
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x_batch)
            # per-timestep loss, then mask out gap-filled timesteps before
            # averaging - filled timesteps carry no real signal and
            # shouldn't contribute to the gradient (per Ismayil's fill scheme)
            loss_per_timestep = criterion(logits, y_batch)
            loss = (loss_per_timestep * mask_batch).sum() / mask_batch.sum()
            loss = loss / grad_accum_steps

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
    for x_batch, y_batch, mask_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        logits = model(x_batch)
        loss_per_timestep = criterion(logits, y_batch)
        loss = (loss_per_timestep * mask_batch).sum() / mask_batch.sum()
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
    parser.add_argument("--use_real_data", action="store_true",
                         help="load real Farm A sequences instead of dummy data")
    parser.add_argument("--data_dir", type=str,
                         default="data/processed/CARE_Farm_A/sequences")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    if args.use_real_data:
        train_loader, val_loader = build_real_dataloaders(args.data_dir, args.batch_size)
    else:
        train_loader, val_loader = build_dummy_dataloaders(batch_size=args.batch_size)

    # --- pos_weight from TRAINING labels only, excluding gap-filled timesteps ---
    all_train_labels = torch.cat([y for _, y, _ in train_loader])
    all_train_masks = torch.cat([m for _, _, m in train_loader])
    pos_weight = compute_pos_weight(all_train_labels, all_train_masks).to(device)
    # reduction="none": we need per-timestep loss to apply the mask ourselves
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # --- Model / optimizer ---
    model = TemporalRiskModel(input_size=54, hidden_size=args.hidden_size).to(device)
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