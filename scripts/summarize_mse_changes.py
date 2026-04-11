import os
import re
from pathlib import Path

import pandas as pd


CHECKPOINTS_ROOT = Path("checkpoints")


def find_metric_files(root: Path):
    """Yield (experiment_name, dataset, split, csv_path) for each metrics CSV found."""
    if not root.exists():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        for fname in filenames:
            if not fname.endswith("_metrics_filtered.csv"):
                continue
            csv_path = dirpath / fname

            # Try to infer dataset name from a parent folder like local_pics_ETTh1
            dataset = None
            for parent in csv_path.parents:
                if parent.name.startswith("local_pics_"):
                    dataset = parent.name.replace("local_pics_", "", 1)
                    break

            # Experiment name: top-level checkpoint dir under checkpoints/
            try:
                rel = csv_path.relative_to(root)
                experiment_name = rel.parts[0]
            except ValueError:
                experiment_name = "unknown"

            # Split should also be in the CSV, but we keep filename info as a fallback
            if "test" in fname:
                split_hint = "test"
            elif "val" in fname:
                split_hint = "val"
            else:
                split_hint = "unknown"

            yield experiment_name, dataset, split_hint, csv_path


def summarize_one_csv(experiment_name: str, dataset: str, split_hint: str, csv_path: Path):
    df = pd.read_csv(csv_path)
    # Expect columns: target_coverage, split_mse_selected, split_mse_orig, split
    required_cols = {"target_coverage", "split_mse_selected", "split_mse_orig"}
    if not required_cols.issubset(df.columns):
        # Not the metrics format we expect; skip
        return None

    # Use the split column if present, otherwise fall back to hint
    split_col = df["split"].iloc[0] if "split" in df.columns else split_hint

    # Sanity: original MSE should be constant across rows; take the first
    orig_mse = df["split_mse_orig"].iloc[0]

    # Avoid division by zero; if orig MSE is zero, percentage change is undefined
    if orig_mse == 0:
        df["mse_delta"] = df["split_mse_selected"] - df["split_mse_orig"]
        df["mse_delta_percent"] = float("nan")
    else:
        df["mse_delta"] = df["split_mse_selected"] - df["split_mse_orig"]
        df["mse_delta_percent"] = df["mse_delta"] / orig_mse * 100.0

    df["experiment"] = experiment_name
    df["dataset"] = dataset
    df["split_effective"] = split_col
    df["metrics_file"] = str(csv_path)

    # Reorder columns for readability
    cols = [
        "experiment",
        "dataset",
        "split_effective",
        "target_coverage",
        "split_mse_selected",
        "split_mse_orig",
        "mse_delta",
        "mse_delta_percent",
        "metrics_file",
    ]
    return df[cols]


def main():
    all_rows = []
    for experiment_name, dataset, split_hint, csv_path in find_metric_files(CHECKPOINTS_ROOT):
        summary_df = summarize_one_csv(experiment_name, dataset, split_hint, csv_path)
        if summary_df is not None:
            all_rows.append(summary_df)

    if not all_rows:
        print("No *_metrics_filtered.csv files found under 'checkpoints/'. Nothing to summarize.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # Save a global summary CSV at repo root
    out_path = Path("mse_change_summary.csv")
    full_df.to_csv(out_path, index=False)

    # Also print a short human-readable summary grouped by experiment/dataset/split
    print(f"Saved detailed summary to {out_path}")
    print()

    grouped = full_df.groupby(["experiment", "dataset", "split_effective"])
    for (exp, ds, sp), g in grouped:
        # Use the max target_coverage row as a compact summary point
        row = g.loc[g["target_coverage"].idxmax()]
        delta_pct = row["mse_delta_percent"]
        direction = "decrease" if delta_pct < 0 else "increase"
        print(
            f"{exp} | dataset={ds} | split={sp} | coverage={row['target_coverage']:.3f}: "
            f"selected MSE {direction} of {abs(delta_pct):.3f}% vs original"
        )


if __name__ == "__main__":
    main()
