"""
Task 15 - build the required cross-farm results artifacts.

Consumes the per-timestep prediction tables already exported by
export_gru_predictions.py for the designated checkpoint
(checkpoints/farm_c_lr5e4/best.pt, 10-feature strict schema) and produces:

    artifacts/cross_farm_results.json
    artifacts/cross_farm_results.md

The frozen decision threshold is selected ONCE on Farm C validation using
select_threshold_max_f1() (same 'validation F1' method already used for the
Farm C classical baselines, see artifacts/baseline/farm_c/run_manifest.json)
and then applied UNCHANGED to Farm C test and both zero-shot targets - no
target-domain threshold tuning, no retraining, no new scaler fit, per the
roadmap's constraints.

Does not invent numbers: every value in the output traces back to one of
the four input parquet files or the scaler_stats.npz file. If any input is
missing, this script fails loudly rather than filling in a placeholder.

Usage (from repo root):
    python src/eval/build_cross_farm_results.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from export_gru_predictions import checkpoint_state, infer_model_shape  # noqa: E402

CHECKPOINT_NAME = "farm_c_lr5e4"

# A prior cross-farm attempt was committed to this repository (Kamal, PRs
# #23/#24, commits ddf0264/1b49625) before the corrected checkpoint was
# designated. That run used checkpoints/farm_c_final/best.pt, which its own
# original writeup incorrectly described as the 11-feature power_residual
# checkpoint - this was verified false by inspecting the checkpoint's actual
# weight shape (infer_model_shape -> input_size=10, matching this run's
# checkpoint). farm_c_final and farm_c_lr5e4 are two distinct 10-feature
# checkpoints from the original six-config training sweep; they differ in
# hyperparameters (at minimum dropout: 0.3 vs 0.4 here), not feature count.
# The numbers below are the real, previously-published results for that run,
# kept for provenance rather than re-derived (its raw prediction file is not
# available to this script) - not invented, and not presented as the
# designated result.
SUPERSEDED_PRIOR_RESULT = {
    "checkpoint_name": "farm_c_final",
    "checkpoint_path": "checkpoints/farm_c_final/best.pt",
    "verified_input_size": 10,
    "config_as_originally_reported": {
        "hidden_size": 32, "num_layers": 2, "dropout": 0.3, "bidirectional": False,
    },
    "note_on_original_writeup_error": (
        "The original writeup for this run stated farm_c_final was 'also "
        "previously referred to as' the 11-feature farm_c_power_residual "
        "checkpoint. This is incorrect: farm_c_final's checkpoint weights "
        "were directly verified (infer_model_shape) to have input_size=10, "
        "not 11. Corrected here rather than repeated."
    ),
    "farm_c_test_result": {
        "role": "Source-domain test", "threshold": 0.475,
        "roc_auc": 0.475130, "pr_auc": 0.037999,
        "precision": 0.028658, "recall": 0.624541, "f1": 0.054801,
        "n_valid_timesteps": 290793, "n_positive_labels": 10611, "positive_rate": 0.036490,
    },
    "farm_a_b_zero_shot": "Not evaluated for this checkpoint; superseded before Farm A/B were run.",
    "source": "artifacts/cross_farm_results.md as committed in PR #23/#24 (commits ddf0264, 1b49625), prior to this run.",
}


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "prob" in df.columns and "probability" not in df.columns:
        df = df.rename(columns={"prob": "probability"})
    return M.prepare(df)


def load_scaler_info(scaler_path: str) -> dict:
    data = np.load(scaler_path, allow_pickle=False)
    return {
        "scaler_file": scaler_path,
        "feature_names": data["feature_names"].tolist(),
        "mean": data["mean"].tolist(),
        "std": data["std"].tolist(),
        "source_farm": str(data["source_farm"]),
        "source_split": str(data["source_split"]),
        "fit_mask_policy": str(data["fit_mask"]),
        "note": "Scaler fit on Farm C train only; reused unchanged on Farm A and Farm B (no target-domain refit).",
    }


def load_checkpoint_config(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint_state(ckpt)
    input_size, hidden_size, num_layers = infer_model_shape(state)
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": CHECKPOINT_NAME,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "trained_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_val_loss": ckpt.get("best_val_loss") if isinstance(ckpt, dict) else None,
    }


def build_row(name: str, role: str, df: pd.DataFrame, threshold: float) -> dict:
    m = M.classification_metrics(df["label"].to_numpy(), df["probability"].to_numpy(), threshold)
    return {
        "split": name,
        "role": role,
        "roc_auc": round(m["roc_auc"], 4),
        "pr_auc": round(m["pr_auc"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "n_valid_timesteps": m["n"],
        "n_positive_labels": m["n_positive"],
        "positive_rate": round(m["positive_rate"], 4),
        "frozen_threshold": threshold,
    }


def build_asset_breakdown(name: str, df: pd.DataFrame, threshold: float) -> list[dict]:
    brk = M.breakdown(df, threshold, by="asset_id")
    brk.insert(0, "split", name)
    return brk.round(4).to_dict(orient="records")


def write_markdown(results: dict, out_path: str) -> None:
    lines = []
    lines.append("# Cross-Farm Generalization Results (Task 15)\n")
    lines.append(f"**Checkpoint:** `{results['model_config']['checkpoint_path']}` "
                 f"({results['model_config']['checkpoint_name']})\n")
    lines.append(f"**Config:** GRU, input_size={results['model_config']['input_size']}, "
                 f"hidden_size={results['model_config']['hidden_size']}, "
                 f"num_layers={results['model_config']['num_layers']}, "
                 f"trained_epoch={results['model_config']['trained_epoch']}, "
                 f"best_val_loss={results['model_config']['best_val_loss']}\n")
    lines.append(f"**Frozen threshold:** {results['frozen_threshold']} "
                 f"(selected via max-F1 on Farm C validation only; "
                 f"F1 at threshold = {results['frozen_threshold_val_f1']}. "
                 f"NOT re-tuned on Farm A or Farm B.)\n")
    lines.append(f"**Preprocessing:** scaler fit on {results['scaler']['source_farm']} "
                 f"({results['scaler']['source_split']} split) only, features: "
                 f"{', '.join(results['scaler']['feature_names'])}. Reused unchanged on all splits.\n")
    lines.append("\n## Superseded prior result (kept for provenance)\n")
    prior = results["superseded_prior_result"]
    lines.append(f"An earlier cross-farm attempt in this repository used "
                 f"`{prior['checkpoint_path']}` (**{prior['checkpoint_name']}**), "
                 f"verified input_size={prior['verified_input_size']}, config "
                 f"{prior['config_as_originally_reported']}.\n")
    lines.append(f"> **Correction to the original writeup:** {prior['note_on_original_writeup_error']}\n")
    r = prior["farm_c_test_result"]
    lines.append("| Split | Role | ROC-AUC | PR-AUC | Precision | Recall | F1 | threshold |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| Farm C test | {r['role']} | {r['roc_auc']} | {r['pr_auc']} | "
                 f"{r['precision']} | {r['recall']} | {r['f1']} | {r['threshold']} |")
    lines.append(f"\n*{prior['farm_a_b_zero_shot']}*\n")
    lines.append(f"\nSource: {prior['source']}\n")

    lines.append("\n## Results table\n")
    lines.append("| Split | Role | ROC-AUC | PR-AUC | Precision | Recall | F1 | n valid | n positive | pos. rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in results["results_table"]:
        lines.append(
            f"| {row['split']} | {row['role']} | {row['roc_auc']} | {row['pr_auc']} | "
            f"{row['precision']} | {row['recall']} | {row['f1']} | {row['n_valid_timesteps']} | "
            f"{row['n_positive_labels']} | {row['positive_rate']} |"
        )

    lines.append("\n## Per-asset breakdown (appendix)\n")
    lines.append("| Split | Asset | n | ROC-AUC | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for split_name in ["Farm C test (source)", "Farm A test (zero-shot)", "Farm B test (zero-shot)"]:
        for row in results["asset_breakdown"][split_name]:
            lines.append(
                f"| {split_name} | {row['asset_id']} | {row['n']} | {row.get('roc_auc','')} | "
                f"{row.get('precision','')} | {row.get('recall','')} | {row.get('f1','')} |"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        Path("/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/checkpoints/farm_c_lr5e4/best.pt")))
    parser.add_argument("--scaler", default=str(
        Path("/sdb-disk/notebooks/team12/team-Sigmoid-crossfarm/data/processed/CARE_Farm_C/sequences/scaler_stats.npz")))
    parser.add_argument("--farm-c-val", default="artifacts/cross_farm/farm_c_lr5e4_val.parquet")
    parser.add_argument("--farm-c-test", default="artifacts/cross_farm/farm_c_lr5e4_test.parquet")
    parser.add_argument("--farm-a-test", default="artifacts/cross_farm/farm_a_zeroshot_test.parquet")
    parser.add_argument("--farm-b-test", default="artifacts/cross_farm/farm_b_zeroshot_test.parquet")
    parser.add_argument("--out-json", default="artifacts/cross_farm_results.json")
    parser.add_argument("--out-md", default="artifacts/cross_farm_results.md")
    args = parser.parse_args()

    df_c_val = load_predictions(args.farm_c_val)
    df_c_test = load_predictions(args.farm_c_test)
    df_a_test = load_predictions(args.farm_a_test)
    df_b_test = load_predictions(args.farm_b_test)

    selection = M.select_threshold_max_f1(
        df_c_val["label"].to_numpy(), df_c_val["probability"].to_numpy()
    )
    threshold = selection["threshold"]

    results = {
        "model_config": load_checkpoint_config(args.checkpoint),
        "scaler": load_scaler_info(args.scaler),
        "frozen_threshold": threshold,
        "frozen_threshold_selection_method": selection["selection_method"],
        "frozen_threshold_comparison_operator": selection["comparison_operator"],
        "frozen_threshold_val_f1": round(selection["f1_at_threshold"], 4),
        "results_table": [
            build_row("Farm C test", "Source-domain test", df_c_test, threshold),
            build_row("Farm A test", "Zero-shot target", df_a_test, threshold),
            build_row("Farm B test", "Zero-shot target", df_b_test, threshold),
        ],
        "asset_breakdown": {
            "Farm C test (source)": build_asset_breakdown("Farm C test", df_c_test, threshold),
            "Farm A test (zero-shot)": build_asset_breakdown("Farm A test", df_a_test, threshold),
            "Farm B test (zero-shot)": build_asset_breakdown("Farm B test", df_b_test, threshold),
        },
        "superseded_prior_result": SUPERSEDED_PRIOR_RESULT,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, args.out_md)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"\nFrozen threshold: {threshold} (val F1={selection['f1_at_threshold']:.4f})")
    for row in results["results_table"]:
        print(f"  {row['split']:15s} {row['role']:20s} ROC-AUC={row['roc_auc']}  F1={row['f1']}")


if __name__ == "__main__":
    main()
