import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mse_curves(csv_path: str, output_dir: str = "plots_ebm_graph_fusion") -> None:
    df = pd.read_csv(csv_path)

    # 只看 test split
    df = df[df["split"] == "test"].copy()
    if df.empty:
        print("No test split rows found in the CSV.")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique())
    methods = ["ebm_only", "graph_only", "fusion"]

    for dataset in datasets:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue

        # 取基线 MSE（split_mse_orig 在同一数据集/方法下是常数）
        baseline = float(sub["split_mse_orig"].iloc[0])

        plt.figure(figsize=(6, 4))

        for method in methods:
            msub = sub[sub["method"] == method].copy()
            if msub.empty:
                continue
            msub = msub.sort_values("target_coverage")

            x = msub["target_coverage"].values
            y = msub["split_mse_selected"].values
            plt.plot(x, y, marker="o", label=method)

        # 画一条基线（不做选择的平均 MSE）
        plt.axhline(y=baseline, color="gray", linestyle="--", label="orig_MSE")

        plt.xlabel("Target coverage")
        plt.ylabel("Selected MSE (test)")
        plt.title(f"{dataset} - MSE vs coverage (test)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_file = out_dir / f"{dataset}_test_mse_curves.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved {out_file}")


if __name__ == "__main__":
    csv_path = "ebm_graph_fusion_test_metrics.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    plot_mse_curves(csv_path)
