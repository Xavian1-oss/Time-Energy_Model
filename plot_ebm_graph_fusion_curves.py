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

    methods = ["ebm_only", "graph_only", "fusion"]

    # 使用不同的线型和 marker，让三种方法在图上更容易区分；
    # 即便数值完全重合，至少可以从图例和样式上看出它们都存在。
    line_styles = {
        "ebm_only": "-",
        "graph_only": "--",
        "fusion": "-.",
    }
    markers = {
        "ebm_only": "o",
        "graph_only": "s",
        "fusion": "^",
    }

    # 如果 CSV 中带有 model 列，则按 backbone 进一步分类，
    # 每个 backbone 生成一个子目录保存对应的数据集曲线。
    if "model" in df.columns:
        model_values = sorted(df["model"].dropna().unique())
        for model_name in model_values:
            df_model = df[df["model"] == model_name].copy()
            if df_model.empty:
                continue

            model_dir = out_dir / str(model_name)
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

                    label = method
                    if method == "fusion" and "fusion_alpha" in msub.columns:
                        try:
                            alpha_val = float(msub["fusion_alpha"].iloc[0])
                            label = f"fusion (alpha={alpha_val:.2f})"
                        except Exception:
                            label = "fusion"

                    plt.plot(
                        x,
                        y,
                        marker=markers.get(method, "o"),
                        linestyle=line_styles.get(method, "-"),
                        label=label,
                    )

                plt.axhline(y=baseline, color="gray", linestyle="--", label="orig_MSE")

                plt.xlabel("Target coverage")
                plt.ylabel("Selected MSE (test)")
                plt.title(f"{model_name} | {dataset} - MSE vs coverage (test)")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()

                out_file = model_dir / f"{dataset}_test_mse_curves.png"
                plt.savefig(out_file, dpi=150)
                plt.close()
                print(f"Saved {out_file}")
    else:
        # 兼容旧格式：没有 model 列时，仍然只按数据集区分。
        datasets = sorted(df["dataset"].unique())

        for dataset in datasets:
            sub = df[df["dataset"] == dataset].copy()
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

                label = method
                if method == "fusion" and "fusion_alpha" in msub.columns:
                    try:
                        alpha_val = float(msub["fusion_alpha"].iloc[0])
                        label = f"fusion (alpha={alpha_val:.2f})"
                    except Exception:
                        label = "fusion"

                plt.plot(
                    x,
                    y,
                    marker=markers.get(method, "o"),
                    linestyle=line_styles.get(method, "-"),
                    label=label,
                )

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
