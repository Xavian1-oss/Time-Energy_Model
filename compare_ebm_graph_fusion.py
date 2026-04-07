import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from utils.graph_energy_gate import compute_structure_energy


COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9]


def _select_best_fusion_alpha(
    e_ebm_val_z: np.ndarray,
    e_graph_val_z: np.ndarray,
    mse_sel_val: np.ndarray,
    mse_orig_val: np.ndarray,
    candidate_alphas = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> float:
    """Select fusion weight alpha on the validation split.

    For each candidate alpha, build fused validation energies
    e_fused_val = alpha * e_ebm_val_z + (1 - alpha) * e_graph_val_z,
    run the standard selective-inference procedure (only on val), and
    use the average selected MSE across COVERAGES as the objective.

    Returns the alpha that minimizes this objective.
    """

    best_alpha = candidate_alphas[0]
    best_score = float("inf")

    # We only care about validation performance here. To reuse the
    # existing helper, we pass the same fused energies and MSEs as
    # both "val" and "test" inputs; downstream, thresholds are still
    # determined purely from the val energies.
    for alpha in candidate_alphas:
        e_fused_val = alpha * e_ebm_val_z + (1.0 - alpha) * e_graph_val_z
        val_df, _ = _compute_method_metrics(
            method="fusion_tmp",
            energies_val=e_fused_val,
            energies_test=e_fused_val,
            mse_sel_val=mse_sel_val,
            mse_orig_val=mse_orig_val,
            mse_sel_test=mse_sel_val,
            mse_orig_test=mse_orig_val,
        )

        if val_df.empty:
            continue

        score = float(val_df["split_mse_selected"].mean())
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return float(best_alpha)


def _load_args(args_path: Path) -> SimpleNamespace:
    df = pd.read_csv(args_path)
    if df.empty:
        raise RuntimeError(f"Empty args file: {args_path}")
    args_dict = df.iloc[0].to_dict()
    # pandas may cast bools/ints as numpy types; convert to Python types where needed
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


def _find_result_objects_dir(run_dir: Path) -> Path:
    ro_dir = run_dir / "result_objects"
    if not ro_dir.exists() or not ro_dir.is_dir():
        raise FileNotFoundError(f"result_objects directory not found under {run_dir}")
    return ro_dir


def _find_result_npz(ro_dir: Path, split: str) -> Path:
    # Expect filenames like result_obj_*_<split>_*.npz
    pattern = f"result_obj_*_{split}_*.npz"
    matches = list(ro_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result_obj npz found for split='{split}' in {ro_dir}")
    # If multiple, pick the first (they should be equivalent version-wise)
    return matches[0]


def _compute_method_metrics(
    method: str,
    energies_val: np.ndarray,
    energies_test: np.ndarray,
    mse_sel_val: np.ndarray,
    mse_orig_val: np.ndarray,
    mse_sel_test: np.ndarray,
    mse_orig_test: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple energy-sorting selective inference for a given method.

    Returns (val_df, test_df) with columns:
        target_coverage, train_coverage, split_mse_selected,
        split_mse_orig, split, method
    """

    def _to_1d(arr: np.ndarray) -> np.ndarray:
        """Flatten per-sample quantities: average over non-batch dims."""
        arr = np.asarray(arr)
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(1, arr.ndim)))
        return arr.reshape(-1)

    val_energy = _to_1d(energies_val)
    test_energy = _to_1d(energies_test)

    val_mse_sel = _to_1d(mse_sel_val)
    test_mse_sel = _to_1d(mse_sel_test)

    val_mse_orig = _to_1d(mse_orig_val)
    test_mse_orig = _to_1d(mse_orig_test)

    order = np.argsort(val_energy)
    sorted_val_energy = val_energy[order]
    n_val = len(sorted_val_energy)

    def build_df_for_split(
        split_name: str,
        split_energy: np.ndarray,
        split_mse_sel: np.ndarray,
        split_mse_orig: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        if n_val == 0:
            return pd.DataFrame(rows)

        for target_cov in COVERAGES:
            k = max(1, int(round(target_cov * n_val)))
            k = min(k, n_val)
            thresh = sorted_val_energy[k - 1]

            val_mask = val_energy <= thresh
            val_cov = float(val_mask.sum()) / float(n_val)

            split_mask = split_energy <= thresh
            if split_mask.sum() == 0:
                continue

            mse_selected = float(split_mse_sel[split_mask].mean())
            mse_orig_all = float(split_mse_orig.mean())

            rows.append(
                {
                    "target_coverage": float(target_cov),
                    "train_coverage": val_cov,
                    "split_mse_selected": mse_selected,
                    "split_mse_orig": mse_orig_all,
                    "split": split_name,
                    "method": method,
                }
            )
        return pd.DataFrame(rows)

    val_df = build_df_for_split("val", val_energy, val_mse_sel, val_mse_orig)
    test_df = build_df_for_split("test", test_energy, test_mse_sel, test_mse_orig)
    return val_df, test_df


def _build_correlation_adjacency_from_array(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Build a correlation-based adjacency matrix from a 3D array.

    Input shape [N, L, D]; output is row-normalized non-negative [D, D].
    """
    if arr.ndim != 3:
        raise ValueError(f"Input must have shape [N, L, D], got {arr.shape}")

    N, L, D = arr.shape
    flat = arr.reshape(-1, D)  # [N*L, D]
    num_rows = flat.shape[0]

    # Subsample rows to limit memory usage on very large datasets.
    max_rows = 50000
    if num_rows > max_rows:
        idx = np.random.choice(num_rows, size=max_rows, replace=False)
        flat = flat[idx]

    features = torch.from_numpy(flat).float()

    count = features.size(0)
    if count == 0:
        raise RuntimeError("Empty array when building correlation adjacency.")

    sum_x = features.sum(dim=0).double()  # [D]
    sum_x2 = (features ** 2).sum(dim=0).double()  # [D]
    sum_cross = features.t().double().mm(features.double())  # [D, D]

    mean = sum_x / count
    ex2 = sum_x2 / count
    var = ex2 - mean ** 2
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)

    exy = sum_cross / count
    cov = exy - mean.view(-1, 1) * mean.view(1, -1)

    std_outer = std.view(-1, 1) * std.view(1, -1)
    std_outer = torch.clamp(std_outer, min=1e-8)
    corr = cov / std_outer

    A_raw = torch.abs(corr)
    A_raw = torch.nan_to_num(A_raw, nan=0.0)
    row_sums = A_raw.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    A = A_raw / row_sums
    return A.to(device=device, dtype=torch.float32)


def _compute_graph_energies_chunked(
    y_hat_np: np.ndarray,
    A: torch.Tensor,
    device: torch.device,
    max_batch_size: int,
) -> np.ndarray:
    """Compute graph-structural energies in small batches to limit memory usage.

    y_hat_np has shape [N, H, D], A is [D, D]. We iterate over the
    first dimension in chunks so that compute_structure_energy never
    sees a huge batch at once (which would create a massive
    [B, H, D, D] tensor internally and risk OOM).
    """

    N = y_hat_np.shape[0]
    energies: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, N, max_batch_size):
            end = min(start + max_batch_size, N)
            if start >= end:
                break
            chunk = torch.from_numpy(y_hat_np[start:end]).to(device)
            e_chunk = compute_structure_energy(chunk, A)
            energies.append(e_chunk.cpu().numpy())

    if not energies:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(energies, axis=0)


def process_single_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process one experiment run directory under checkpoints.

    Expects:
        - args.csv in run_dir
        - result_objects/ with val/test npz caches

    Returns concatenated (val_df_all, test_df_all) over methods.
    """
    args_path = run_dir / "args.csv"
    if not args_path.exists():
        raise FileNotFoundError(f"args.csv not found in {run_dir}")

    args = _load_args(args_path)

    # Backbone model name (Autoformer, PatchTST, etc.) so that downstream
    # metrics and plots can be grouped per backbone.
    model_name = getattr(args, "model", "unknown")

    # Only consider multivariate (M) tasks as requested.
    if getattr(args, "features", None) != "M":
        raise RuntimeError(f"Run {run_dir} is not multivariate (features == 'M'); skipping.")

    ro_dir = _find_result_objects_dir(run_dir)
    val_npz_path = _find_result_npz(ro_dir, "val")
    test_npz_path = _find_result_npz(ro_dir, "test")

    val_obj = dict(np.load(str(val_npz_path), allow_pickle=True))
    test_obj = dict(np.load(str(test_npz_path), allow_pickle=True))

    device = torch.device("cpu")

    # EBM-based energies and MSEs.
    e_ebm_val = np.asarray(val_obj["energy_hats_init_orig_model"])
    e_ebm_test = np.asarray(test_obj["energy_hats_init_orig_model"])

    mse_sel_val = np.asarray(val_obj["mse_init_orig_model"])
    mse_orig_val = np.asarray(val_obj["mse_orig"])
    mse_sel_test = np.asarray(test_obj["mse_init_orig_model"])
    mse_orig_test = np.asarray(test_obj["mse_orig"])

    # Decide whether to include graph-based analysis for this run.
    # Only construct graph_only / fusion curves when the run was
    # actually configured to use an adaptive graph during training.
    graph_mode = getattr(args, "graph_mode", "auto")
    use_adaptive_graph = bool(getattr(args, "use_adaptive_graph", False))
    use_graph_for_analysis = (graph_mode != "none") and use_adaptive_graph

    # Determine a human-readable dataset name once so it can be reused
    # for both ETT-style and custom datasets.
    raw_dataset = getattr(args, "data", "unknown")
    data_path_val = getattr(args, "data_path", None)
    if raw_dataset == "custom" and data_path_val is not None:
        dataset_name = Path(str(data_path_val)).stem
    else:
        dataset_name = raw_dataset

    # EBM-only metrics (always available).
    val_ebm, test_ebm = _compute_method_metrics(
        method="ebm_only",
        energies_val=e_ebm_val,
        energies_test=e_ebm_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
    )

    # If this run did not train an adaptive graph (e.g., TEM-only with
    # graph_mode == "none"), do not construct any graph_only / fusion
    # curves. This keeps graph-based analysis restricted to runs that
    # actually used a learned graph during training.
    if not use_graph_for_analysis:
        # Reuse the dataset_name logic above for consistency.
        dataset = dataset_name

        experiment = run_dir.name
        for df in (val_ebm, test_ebm):
            df["experiment"] = experiment
            df["dataset"] = dataset
            df["model"] = model_name

        return val_ebm, test_ebm

    # Graph-structural energies from forecasts and adjacency.
    y_hat_val = np.asarray(val_obj["y_hats_init_orig_model"]).astype(np.float32)
    y_hat_test = np.asarray(test_obj["y_hats_init_orig_model"]).astype(np.float32)

    D = y_hat_val.shape[-1]

    # Prefer using the learned adjacency matrix saved during training
    # (online graph gate) so that offline graph/EBM fusion uses the
    # exact same structure. If it is missing or incompatible, fall
    # back to a correlation-based adjacency built from forecasts.
    A = None
    graph_A_path = run_dir / "learned_graph_A.npy"
    if graph_A_path.exists():
        try:
            A_np = np.load(graph_A_path)
            if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
                raise ValueError(
                    f"learned_graph_A.npy has invalid shape {A_np.shape}, expected square [D, D]"
                )
            if A_np.shape[0] != D:
                raise ValueError(
                    f"learned_graph_A.npy dimension {A_np.shape[0]} does not match output D={D}"
                )
            A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
            print(f"[GraphEnergy] Using learned adjacency from {graph_A_path}")
        except Exception as e:
            print(
                f"[GraphEnergy][WARN] Failed to load learned adjacency from {graph_A_path}, "
                f"falling back to correlation-based adjacency: {e}"
            )

    # If no valid learned adjacency is available, build a
    # correlation-based adjacency directly from validation forecasts.
    if A is None:
        A = _build_correlation_adjacency_from_array(y_hat_val, device)

    # Compute graph energies in small batches to avoid building a huge
    # [B, H, D, D] tensor at once (which is especially problematic for
    # high-dimensional datasets like electricity/traffic).
    if D > 128:
        # Very high-dimensional outputs: use tiny batches for safety.
        max_batch_size = 1
    else:
        max_batch_size = 64

    e_graph_val = _compute_graph_energies_chunked(
        y_hat_np=y_hat_val,
        A=A,
        device=device,
        max_batch_size=max_batch_size,
    )
    e_graph_test = _compute_graph_energies_chunked(
        y_hat_np=y_hat_test,
        A=A,
        device=device,
        max_batch_size=max_batch_size,
    )

    # Fusion energies: z-score on validation set, shared for test.
    e_ebm_val_z = (e_ebm_val - e_ebm_val.mean()) / (e_ebm_val.std() + 1e-8)
    e_graph_val_z = (e_graph_val - e_graph_val.mean()) / (e_graph_val.std() + 1e-8)

    e_ebm_test_z = (e_ebm_test - e_ebm_val.mean()) / (e_ebm_val.std() + 1e-8)
    e_graph_test_z = (e_graph_test - e_graph_val.mean()) / (e_graph_val.std() + 1e-8)

    # Select per-run fusion weight on validation split.
    best_alpha = _select_best_fusion_alpha(
        e_ebm_val_z=e_ebm_val_z,
        e_graph_val_z=e_graph_val_z,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
    )

    e_fused_val = best_alpha * e_ebm_val_z + (1.0 - best_alpha) * e_graph_val_z
    e_fused_test = best_alpha * e_ebm_test_z + (1.0 - best_alpha) * e_graph_test_z

    # Compute metrics for each method.
    val_graph, test_graph = _compute_method_metrics(
        method="graph_only",
        energies_val=e_graph_val,
        energies_test=e_graph_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
    )

    val_fused, test_fused = _compute_method_metrics(
        method="fusion",
        energies_val=e_fused_val,
        energies_test=e_fused_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
    )

    # Record the chosen fusion alpha for analysis.
    val_fused["fusion_alpha"] = best_alpha
    test_fused["fusion_alpha"] = best_alpha

    # Annotate with experiment metadata.
    # 对于 ETT 系列，args.data 本身就是 ETTh1/ETTm1 等；
    # 对于自定义数据（args.data == "custom"），用 data_path 的文件名
    # （去掉扩展名）来区分 electricity / exchange_rate / weather 等。
    # Reuse the dataset_name logic above for consistency.
    dataset = dataset_name

    experiment = run_dir.name
    for df in (val_ebm, test_ebm, val_graph, test_graph, val_fused, test_fused):
        df["experiment"] = experiment
        df["dataset"] = dataset
        df["model"] = model_name

    val_all = pd.concat([val_ebm, val_graph, val_fused], ignore_index=True)
    test_all = pd.concat([test_ebm, test_graph, test_fused], ignore_index=True)
    return val_all, test_all


def main():
    checkpoints_root = Path("./checkpoints")
    all_val: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []

    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root not found: {checkpoints_root}")

    run_dirs = sorted(
        [
            p
            for p in checkpoints_root.rglob("*")
            if (p / "result_objects").exists() and (p / "args.csv").exists()
        ]
    )

    print(f"Found {len(run_dirs)} candidate runs with result_objects and args.csv")

    for run_dir in run_dirs:
        try:
            print(f"Processing run: {run_dir}")
            val_df, test_df = process_single_run(run_dir)
            all_val.append(val_df)
            all_test.append(test_df)
        except Exception as e:
            print(f"[WARN] Skipping run {run_dir} due to error: {e}")

    if not all_val:
        print("No valid multivariate runs processed; nothing to save.")
        return

    val_all = pd.concat(all_val, ignore_index=True)
    test_all = pd.concat(all_test, ignore_index=True)

    # Global aggregated CSVs (backward-compatible).
    out_val = Path("ebm_graph_fusion_val_metrics.csv")
    out_test = Path("ebm_graph_fusion_test_metrics.csv")
    val_all.to_csv(out_val, index=False)
    test_all.to_csv(out_test, index=False)

    print(f"Saved aggregated validation metrics to {out_val}")
    print(f"Saved aggregated test metrics to {out_test}")

    # Additionally, store metrics grouped by backbone under
    # ebm_graph_fusion_metrics/<model>/... so that different
    # backbones' curves/results are cleanly separated.
    metrics_root = Path("ebm_graph_fusion_metrics")
    metrics_root.mkdir(parents=True, exist_ok=True)

    for split_name, df_split in ("val", val_all), ("test", test_all):
        if "model" not in df_split.columns:
            continue
        for model_name in sorted(df_split["model"].dropna().unique()):
            sub = df_split[df_split["model"] == model_name]
            if sub.empty:
                continue
            model_dir = metrics_root / str(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            out_path = model_dir / f"ebm_graph_fusion_{split_name}_metrics.csv"
            sub.to_csv(out_path, index=False)
            print(f"Saved {split_name} metrics for model={model_name} to {out_path}")

    # Quick text summary on test split.
    print("\n=== Test split summary (by dataset, method, coverage) ===")
    summary = (
        test_all.groupby(["dataset", "method", "target_coverage"])[
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


if __name__ == "__main__":
    main()
