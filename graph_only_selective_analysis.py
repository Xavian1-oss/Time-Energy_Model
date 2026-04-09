import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _run_matches_output_parent_filter(
    run_dir: Path, require_substr: Optional[str]
) -> bool:
    """Keep only runs whose saved args.output_parent_path contains require_substr."""
    if not require_substr:
        return True
    args_path = run_dir / "args.csv"
    if not args_path.exists():
        return False
    try:
        df = pd.read_csv(args_path)
        if df.empty:
            return False
        op = df.iloc[0].get("output_parent_path", "")
        return require_substr in str(op)
    except Exception:
        return False


def _load_args(args_path: Path) -> SimpleNamespace:
    """Load args.csv as a namespace (compatible with compare_ebm_graph_fusion).

    Only needs a small subset of fields: model, data, data_path, features.
    """
    df = pd.read_csv(args_path)
    if df.empty:
        raise RuntimeError(f"Empty args file: {args_path}")

    args_dict = df.iloc[0].to_dict()
    clean_dict = {}
    for k, v in args_dict.items():
        if isinstance(v, np.bool_):
            clean_dict[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            clean_dict[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_dict[k] = float(v)
        else:
            clean_dict[k] = v
    return SimpleNamespace(**clean_dict)


def _iter_graph_metrics_runs(checkpoints_root: Path) -> List[Tuple[Path, Path, Path]]:
    """Find all runs that have graph_val/test_metrics_filtered.csv.

    Returns a list of (run_dir, val_csv, test_csv).
    """
    runs: List[Tuple[Path, Path, Path]] = []

    for test_csv in checkpoints_root.rglob("graph_test_metrics_filtered.csv"):
        pics_dir = test_csv.parent
        val_csv = pics_dir / "graph_val_metrics_filtered.csv"
        if not val_csv.exists():
            continue
        # run_dir is the parent of local_pics_* directory
        run_dir = pics_dir.parent
        args_path = run_dir / "args.csv"
        if not args_path.exists():
            # Skip runs without args.csv; we cannot annotate model/dataset.
            continue
        runs.append((run_dir, val_csv, test_csv))

    return runs


def aggregate_graph_only_metrics(
    checkpoints_root: Path,
    require_output_parent_substr: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-run graph gate metrics into global val/test DataFrames.

    Compatible with graph-only 模式：只依赖 graph_*_metrics_filtered.csv
    和 args.csv，不要求存在任何 EBM 结果文件。

    Args:
        checkpoints_root: Root to scan (e.g. ./checkpoints).
        require_output_parent_substr: If set, only runs whose args.csv
            output_parent_path contains this string are included (same idea
            as compare_ebm_graph_fusion.py batch filtering).
    """
    all_val: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []

    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root not found: {checkpoints_root}")

    runs = _iter_graph_metrics_runs(checkpoints_root)
    if require_output_parent_substr:
        runs = [
            r
            for r in runs
            if _run_matches_output_parent_filter(r[0], require_output_parent_substr)
        ]
    print(f"Found {len(runs)} runs with graph_*_metrics_filtered.csv", end="")
    if require_output_parent_substr:
        print(f" (filtered by output_parent_path containing {require_output_parent_substr!r})")
    else:
        print()

    for run_dir, val_csv, test_csv in runs:
        try:
            args_path = run_dir / "args.csv"
            args = _load_args(args_path)

            model_name = getattr(args, "model", "unknown")
            raw_dataset = getattr(args, "data", "unknown")
            data_path_val = getattr(args, "data_path", None)

            if raw_dataset == "custom" and data_path_val is not None:
                dataset_name = Path(str(data_path_val)).stem
            else:
                dataset_name = raw_dataset

            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)

            # Annotate with experiment metadata and method name.
            experiment = run_dir.name
            for df in (val_df, test_df):
                df["experiment"] = experiment
                df["dataset"] = dataset_name
                df["model"] = model_name
                df["method"] = "graph_only"

            all_val.append(val_df)
            all_test.append(test_df)

            print(f"Aggregated graph metrics for run: {run_dir}")
        except Exception as e:
            print(f"[WARN] Skipping run {run_dir} due to error: {e}")

    if not all_val:
        print("No valid graph-only runs processed; nothing to aggregate.")
        return pd.DataFrame(), pd.DataFrame()

    val_all = pd.concat(all_val, ignore_index=True)
    test_all = pd.concat(all_test, ignore_index=True)
    return val_all, test_all


def _plot_graph_only_curves(test_df: pd.DataFrame, output_dir: str = "graph_only_plots") -> None:
    """Plot coverage→MSE curves (test) for graph-only metrics.

    直观展示 graph gate 的 selective 效果。
    """
    if test_df.empty:
        print("Empty test DataFrame; skip plotting.")
        return

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    methods = ["graph_only"]

    # If model column exists, group by backbone first.
    if "model" in test_df.columns:
        model_values = sorted(test_df["model"].dropna().unique())
        for model_name in model_values:
            df_model = test_df[test_df["model"] == model_name].copy()
            if df_model.empty:
                continue

            model_dir = out_root / str(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            datasets = sorted(df_model["dataset"].unique())
            for dataset in datasets:
                sub = df_model[df_model["dataset"] == dataset].copy()
                if sub.empty:
                    continue

                baseline = float(sub["split_mse_orig"].iloc[0])

                plt.figure(figsize=(6, 4))

                for method in methods:
                    msub = sub[sub["method"] == method].copy()
                    if msub.empty:
                        continue
                    msub = msub.sort_values("target_coverage")

                    x = msub["target_coverage"].values
                    y = msub["split_mse_selected"].values

                    plt.plot(x, y, marker="o", linestyle="-", label="graph_only")

                plt.axhline(y=baseline, color="gray", linestyle="--", label="orig_MSE")

                plt.xlabel("Target coverage")
                plt.ylabel("Selected MSE (test)")
                plt.title(f"{model_name} | {dataset} - Graph-only selective (test)")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()

                out_file = model_dir / f"{dataset}_graph_only_test_mse_curves.png"
                plt.savefig(out_file, dpi=150)
                plt.close()
                print(f"Saved {out_file}")
    else:
        # Fallback: group only by dataset.
        datasets = sorted(test_df["dataset"].unique())
        for dataset in datasets:
            sub = test_df[test_df["dataset"] == dataset].copy()
            if sub.empty:
                continue

            baseline = float(sub["split_mse_orig"].iloc[0])

            plt.figure(figsize=(6, 4))

            for method in methods:
                msub = sub[sub["method"] == method].copy()
                if msub.empty:
                    continue
                msub = msub.sort_values("target_coverage")

                x = msub["target_coverage"].values
                y = msub["split_mse_selected"].values

                plt.plot(x, y, marker="o", linestyle="-", label="graph_only")

            plt.axhline(y=baseline, color="gray", linestyle="--", label="orig_MSE")

            plt.xlabel("Target coverage")
            plt.ylabel("Selected MSE (test)")
            plt.title(f"{dataset} - Graph-only selective (test)")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            out_file = out_root / f"{dataset}_graph_only_test_mse_curves.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"Saved {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate graph-only selective metrics from checkpoints and plot "
            "coverage→MSE curves (no EBM files required)."
        )
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="./checkpoints",
        help="Root directory to scan for runs with graph_*_metrics_filtered.csv.",
    )
    parser.add_argument(
        "--require-output-parent-substr",
        type=str,
        default=None,
        help=(
            "If set, only include runs whose args.output_parent_path contains "
            "this substring (e.g. batch folder all_runs_tem_graph_joint_M)."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="graph_only_plots",
        help="Directory for PNG curves (default: graph_only_plots).",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="graph_only_val_metrics.csv",
        help="Output path for aggregated validation metrics CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="graph_only_test_metrics.csv",
        help="Output path for aggregated test metrics CSV.",
    )
    cli = parser.parse_args()

    checkpoints_root = Path(cli.checkpoints_root)
    val_all, test_all = aggregate_graph_only_metrics(
        checkpoints_root,
        require_output_parent_substr=cli.require_output_parent_substr,
    )
    if val_all.empty or test_all.empty:
        print("No aggregated graph-only metrics; exiting.")
        return

    # Save global CSVs.
    out_val = Path(cli.val_csv)
    out_test = Path(cli.test_csv)
    val_all.to_csv(out_val, index=False)
    test_all.to_csv(out_test, index=False)

    print(f"Saved aggregated validation metrics to {out_val}")
    print(f"Saved aggregated test metrics to {out_test}")

    # Quick text summary on test split (per backbone when model column exists).
    print("\n=== Graph-only test split summary (by dataset, coverage) ===")
    group_keys = (
        ["model", "dataset", "target_coverage"]
        if "model" in test_all.columns
        else ["dataset", "target_coverage"]
    )
    summary = (
        test_all.groupby(group_keys)[
            ["train_coverage", "split_mse_selected", "split_mse_orig"]
        ]
        .mean()
        .reset_index()
    )
    summary["error_reduction_percent"] = (
        1.0
        - summary["split_mse_selected"]
        / summary["split_mse_orig"].replace(0.0, np.nan)
    ) * 100.0
    print(summary.to_string(index=False))

    # Plot coverage→MSE curves for graph-only.
    _plot_graph_only_curves(test_all, output_dir=cli.plot_dir)


if __name__ == "__main__":
    main()
