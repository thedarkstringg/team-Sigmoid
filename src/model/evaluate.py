"""
Loads a trained checkpoint and computes classification metrics on val/test,
using only real (non-masked) timesteps - directly comparable to Ismayil's
baseline numbers (ROC-AUC 0.662, PR-AUC 0.319, F1 0.384, recall 0.396).
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

from model import TemporalRiskModel


def load_split(data_dir, split):
    x = np.load(f"{data_dir}/{split}_X.npy")
    y = np.load(f"{data_dir}/{split}_y.npy")
    mask = np.load(f"{data_dir}/{split}_mask.npy")
    return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(mask).float()


def evaluate(model, x, y, mask, device, threshold=0.5, batch_size=64):
    model.eval()
    all_probs, all_labels, all_mask = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i+batch_size].to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(y[i:i+batch_size])
            all_mask.append(mask[i:i+batch_size])

    probs = torch.cat(all_probs).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    m = torch.cat(all_mask).numpy().flatten().astype(bool)

    # only evaluate on real (non-gap-filled) timesteps
    probs, labels = probs[m], labels[m]
    preds = (probs > threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds),
        "recall": recall_score(labels, preds),
        "n_timesteps": len(labels),
        "positive_rate": labels.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/CARE_Farm_A/sequences")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0,
                         help="must match the dropout used when this checkpoint was trained")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalRiskModel(input_size=54, hidden_size=args.hidden_size,
                               dropout=args.dropout).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"recorded best_val_loss={checkpoint['best_val_loss']:.4f}")

    for split in ["val", "test"]:
        x, y, mask = load_split(args.data_dir, split)
        metrics = evaluate(model, x, y, mask, device, args.threshold)
        print(f"\n--- {split} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()