from pathlib import Path
import os
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(
    os.environ.get(
        "CARE_FARM_A_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "processed" / "CARE_Farm_A"),
    )
)

df = pd.read_csv(
    ROOT / "baseline_predictions.csv",
    parse_dates=["window_end"]
)

test = df[df["split"] == "test"].copy()

print("=== TEST PERFORMANCE BY EVENT ===")

for model in test["model"].unique():

    print(f"\n--- {model.upper()} ---")

    model_df = test[test["model"] == model]

    for event_id, group in model_df.groupby("event_id"):

        label_type = (
            "ANOMALY"
            if group["label"].sum() > 0
            else "NORMAL"
        )

        predicted_positive = int(
            group["prediction"].sum()
        )

        true_positive = int(
            (
                (group["prediction"] == 1)
                & (group["label"] == 1)
            ).sum()
        )

        false_positive = int(
            (
                (group["prediction"] == 1)
                & (group["label"] == 0)
            ).sum()
        )

        max_probability = group["probability"].max()

        positive_rows = group[
            (group["prediction"] == 1)
            & (group["label"] == 1)
        ]

        if len(positive_rows):
            earliest_warning = (
                positive_rows["hours_to_fault"].max()
            )
        else:
            earliest_warning = None

        print(
            f"Event {event_id:>2} | "
            f"{label_type:7s} | "
            f"windows={len(group):>3} | "
            f"predicted+={predicted_positive:>3} | "
            f"TP={true_positive:>3} | "
            f"FP={false_positive:>3} | "
            f"max_prob={max_probability:.4f} | "
            f"earliest_warning_h={earliest_warning}"
        )
