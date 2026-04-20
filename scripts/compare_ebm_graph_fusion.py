import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

# Running as `python scripts/compare_ebm_graph_fusion.py` puts `scripts/` on
# sys.path first; project modules (utils, ...) live at the repository root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from utils.graph_energy_gate import compute_structure_energy
from utils.selective_baselines import (
    error_predictor_energies,
    mc_dropout_energies_from_run_dir,
)


COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9]

_LEGACY_FUSION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _empirical_cdf_from_val(val_reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map scalar energies x to [0, 1] using the empirical CDF of val_reference."""
    v = np.sort(np.asarray(val_reference, dtype=np.float64).ravel())
    x = np.asarray(x, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return np.zeros_like(x, dtype=np.float64)
    return np.searchsorted(v, x, side="right") / float(n)


def _prepare_fusion_channels(
    e_ebm_val: np.ndarray,
    e_graph_val: np.ndarray,
    e_ebm_test: np.ndarray,
    e_graph_test: np.ndarray,
    fusion_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ebm_val, graph_val, ebm_test, graph_test) in the space used for fusion.

    * ``linear_z``: z-score each channel using val statistics (current default behaviour).
    * ``rank_cdf``: map to empirical CDF under val (test points mapped through val's
      distribution) so both signals live on [0, 1] before blending — no retraining.
    """
    e_ebm_val = np.asarray(e_ebm_val, dtype=np.float64)
    e_graph_val = np.asarray(e_graph_val, dtype=np.float64)
    e_ebm_test = np.asarray(e_ebm_test, dtype=np.float64)
    e_graph_test = np.asarray(e_graph_test, dtype=np.float64)

    if fusion_mode == "linear_z":
        ez = (e_ebm_val - e_ebm_val.mean()) / (e_ebm_val.std() + 1e-8)
        gz = (e_graph_val - e_graph_val.mean()) / (e_graph_val.std() + 1e-8)
        ez_t = (e_ebm_test - e_ebm_val.mean()) / (e_ebm_val.std() + 1e-8)
        gz_t = (e_graph_test - e_graph_val.mean()) / (e_graph_val.std() + 1e-8)
        return ez, gz, ez_t, gz_t

    if fusion_mode == "rank_cdf":
        u_ebm_val = _empirical_cdf_from_val(e_ebm_val, e_ebm_val)
        u_graph_val = _empirical_cdf_from_val(e_graph_val, e_graph_val)
        u_ebm_test = _empirical_cdf_from_val(e_ebm_val, e_ebm_test)
        u_graph_test = _empirical_cdf_from_val(e_graph_val, e_graph_test)
        return u_ebm_val, u_graph_val, u_ebm_test, u_graph_test

    raise ValueError(f"Unknown fusion_mode={fusion_mode!r}; use 'linear_z' or 'rank_cdf'.")


def _fusion_alpha_candidates(fusion_alpha_step: Optional[float]) -> np.ndarray:
    if fusion_alpha_step is None:
        return np.array(_LEGACY_FUSION_ALPHAS, dtype=np.float64)
    step = float(fusion_alpha_step)
    if step <= 0 or step > 1:
        raise ValueError("fusion_alpha_step must be in (0, 1].")
    return np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)


def _select_best_fusion_alpha(
    e_ebm_val_p: np.ndarray,
    e_graph_val_p: np.ndarray,
    mse_sel_val: np.ndarray,
    mse_orig_val: np.ndarray,
    candidate_alphas: np.ndarray,
    interior_bias: float = 0.0,
) -> float:
    """Select fusion weight alpha on the validation split (mean selective MSE objective)."""

    best_alpha = float(candidate_alphas[0])
    best_score = float("inf")

    for alpha in candidate_alphas:
        e_fused_val = alpha * e_ebm_val_p + (1.0 - alpha) * e_graph_val_p
        val_df, _ = _compute_method_metrics(
            method="fusion_tmp",
            energies_val=e_fused_val,
            energies_test=e_fused_val,
            mse_sel_val=mse_sel_val,
            mse_orig_val=mse_orig_val,
            mse_sel_test=mse_sel_val,
            mse_orig_test=mse_orig_val,
            calibrate_threshold_on_split=False,
        )

        if val_df.empty:
            continue

        score = float(val_df["split_mse_selected"].mean())
        score += float(interior_bias) * (float(alpha) - 0.5) ** 2
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha


def _run_matches_output_parent_filter(
    run_dir: Path, require_substr: Optional[str]
) -> bool:
    """Optionally keep only runs whose args.output_parent_path contains a tag."""
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


def _dataset_name_from_args(args: SimpleNamespace) -> str:
    """Same naming rule as ``process_single_run`` (ETT name vs custom CSV stem)."""
    raw_dataset = getattr(args, "data", "unknown")
    data_path_val = getattr(args, "data_path", None)
    if raw_dataset == "custom" and data_path_val is not None:
        return Path(str(data_path_val)).stem
    return str(raw_dataset)


def _run_matches_model_dataset_filters(
    run_dir: Path,
    only_model: Optional[str],
    only_dataset: Optional[str],
) -> bool:
    """Lightweight filter before ``process_single_run`` (reads args.csv only)."""
    if only_model is None and only_dataset is None:
        return True
    args_path = run_dir / "args.csv"
    if not args_path.exists():
        return False
    try:
        args = _load_args(args_path)
    except Exception:
        return False
    if only_model is not None and getattr(args, "model", "") != only_model:
        return False
    if only_dataset is not None and _dataset_name_from_args(args) != only_dataset:
        return False
    return True


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


def _find_result_npz_optional(ro_dir: Path, split: str) -> Optional[Path]:
    """Like ``_find_result_npz`` but return None if no cache (e.g. train not saved)."""
    pattern = f"result_obj_*_{split}_*.npz"
    matches = list(ro_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _len_after_selective_to_1d(mse_or_energy: np.ndarray) -> int:
    """Length of per-window scalars after the same reduction as ``_compute_method_metrics._to_1d``."""
    arr = np.asarray(mse_or_energy)
    if arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(1, arr.ndim)))
    return int(arr.reshape(-1).shape[0])


def _compute_method_metrics(
    method: str,
    energies_val: np.ndarray,
    energies_test: np.ndarray,
    mse_sel_val: np.ndarray,
    mse_orig_val: np.ndarray,
    mse_sel_test: np.ndarray,
    mse_orig_test: np.ndarray,
    calibrate_threshold_on_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple energy-sorting selective inference for a given method.

    Returns (val_df, test_df) with columns:
        target_coverage, train_coverage, split_mse_selected,
        split_mse_orig, split, method

    If ``calibrate_threshold_on_split`` is True (default for compare script), test
    rows use thresholds from **test** energies so empirical coverage on test
    matches ``target_coverage``. If False, test uses the **val**-derived threshold
    (legacy).
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

    # 复用 analysis.perform_selective_inference_experiments 中
    # 提取出的核心逻辑，保证 TEM 与 offline fusion 指标一致。
    from analysis import _compute_energy_sorted_selective_metrics

    val_df = _compute_energy_sorted_selective_metrics(
        val_energy=val_energy,
        val_mse_sel=val_mse_sel,
        val_mse_orig=val_mse_orig,
        split_name="val",
        split_energy=val_energy,
        split_mse_sel=val_mse_sel,
        split_mse_orig=val_mse_orig,
        target_coverages=COVERAGES,
    )

    test_df = _compute_energy_sorted_selective_metrics(
        val_energy=val_energy,
        val_mse_sel=val_mse_sel,
        val_mse_orig=val_mse_orig,
        split_name="test",
        split_energy=test_energy,
        split_mse_sel=test_mse_sel,
        split_mse_orig=test_mse_orig,
        target_coverages=COVERAGES,
        calibrate_threshold_on_split=calibrate_threshold_on_split,
    )

    # 标记 method 字段，保持 compare_ebm_graph_fusion 的接口不变。
    for df in (val_df, test_df):
        df["method"] = method

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


def _selective_baseline_blocks(
    run_dir: Path,
    calibrate_threshold_on_split: bool,
    include_error_predictor: bool,
    include_mc_dropout: bool,
    mc_dropout_passes: int,
    error_predictor_mlp: bool,
    error_predictor_fit_on: str,
    y_hat_val: np.ndarray,
    y_hat_test: np.ndarray,
    y_hat_train: Optional[np.ndarray],
    mse_sel_train: Optional[np.ndarray],
    mse_sel_val: np.ndarray,
    mse_orig_val: np.ndarray,
    mse_sel_test: np.ndarray,
    mse_orig_test: np.ndarray,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Error-predictor and MC-dropout selective metrics.

    Same ``calibrate_threshold_on_split`` as EBM/Graph/Fusion: when True (default),
    **test** rows use test-set quantile thresholds; when False, test uses val-derived
    thresholds (legacy).
    """
    blocks: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    if include_error_predictor:
        ep_method = "error_predictor"
        ep_kw = {}
        if error_predictor_fit_on == "train":
            if y_hat_train is not None and mse_sel_train is not None:
                ep_kw["y_hat_fit"] = y_hat_train
                ep_kw["mse_fit"] = np.asarray(mse_sel_train, dtype=np.float64)
                ep_method = "error_predictor_trainfit"
            else:
                print(
                    "[Error-predictor] --error-predictor-fit-on train but no "
                    "result_obj_*_train_*.npz under result_objects/; fitting on val."
                )
        ev, et = error_predictor_energies(
            y_hat_val,
            np.asarray(mse_sel_val, dtype=np.float64),
            y_hat_test,
            use_mlp=error_predictor_mlp,
            **ep_kw,
        )
        val_df, test_df = _compute_method_metrics(
            method=ep_method,
            energies_val=ev,
            energies_test=et,
            mse_sel_val=mse_sel_val,
            mse_orig_val=mse_orig_val,
            mse_sel_test=mse_sel_test,
            mse_orig_test=mse_orig_test,
            calibrate_threshold_on_split=calibrate_threshold_on_split,
        )
        blocks.append((val_df, test_df))
    if include_mc_dropout:
        mc = mc_dropout_energies_from_run_dir(run_dir, n_passes=mc_dropout_passes)
        if mc is not None:
            ev, et = mc
            # Cached MSE may be [N, D] while MC energies are one per loader batch row (N).
            # Do not use .ravel().size — that counts N*D and wrongly skips MC-dropout.
            n_v = _len_after_selective_to_1d(mse_sel_val)
            n_t = _len_after_selective_to_1d(mse_sel_test)
            if ev.shape[0] == n_v and et.shape[0] == n_t:
                val_df, test_df = _compute_method_metrics(
                    method="mc_dropout",
                    energies_val=ev,
                    energies_test=et,
                    mse_sel_val=mse_sel_val,
                    mse_orig_val=mse_orig_val,
                    mse_sel_test=mse_sel_test,
                    mse_orig_test=mse_orig_test,
                    calibrate_threshold_on_split=calibrate_threshold_on_split,
                )
                blocks.append((val_df, test_df))
            else:
                print(
                    f"[MC-Dropout] Length mismatch vs cached npz "
                    f"(val {ev.shape[0]}!={n_v} or test {et.shape[0]}!={n_t}); skip."
                )
    return blocks


def process_single_run(
    run_dir: Path,
    calibrate_threshold_on_split: bool = True,
    fusion_mode: str = "linear_z",
    fusion_alpha_step: Optional[float] = None,
    fusion_interior_bias: float = 0.0,
    include_error_predictor: bool = False,
    include_mc_dropout: bool = False,
    mc_dropout_passes: int = 20,
    error_predictor_mlp: bool = False,
    error_predictor_fit_on: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    train_npz_path = _find_result_npz_optional(ro_dir, "train")

    val_obj = dict(np.load(str(val_npz_path), allow_pickle=True))
    test_obj = dict(np.load(str(test_npz_path), allow_pickle=True))
    y_hat_train: Optional[np.ndarray] = None
    mse_sel_train: Optional[np.ndarray] = None
    if train_npz_path is not None:
        train_obj = dict(np.load(str(train_npz_path), allow_pickle=True))
        y_hat_train = np.asarray(train_obj["y_hats_init_orig_model"]).astype(np.float32)
        mse_sel_train = np.asarray(train_obj["mse_init_orig_model"])

    device = torch.device("cpu")

    # EBM-based energies and MSEs.
    e_ebm_val = np.asarray(val_obj["energy_hats_init_orig_model"])
    e_ebm_test = np.asarray(test_obj["energy_hats_init_orig_model"])

    mse_sel_val = np.asarray(val_obj["mse_init_orig_model"])
    mse_orig_val = np.asarray(val_obj["mse_orig"])
    mse_sel_test = np.asarray(test_obj["mse_init_orig_model"])
    mse_orig_test = np.asarray(test_obj["mse_orig"])

    y_hat_val = np.asarray(val_obj["y_hats_init_orig_model"]).astype(np.float32)
    y_hat_test = np.asarray(test_obj["y_hats_init_orig_model"]).astype(np.float32)

    # Decide whether to include graph-based analysis for this run.
    # Only construct graph_only / fusion curves when the run was
    # actually configured to use an adaptive graph during training.
    graph_mode = getattr(args, "graph_mode", "auto")
    use_adaptive_graph = bool(getattr(args, "use_adaptive_graph", False))
    use_graph_for_analysis = (graph_mode != "none") and use_adaptive_graph

    # Human-readable dataset name (same rule as ``--only-dataset`` filter).
    dataset_name = _dataset_name_from_args(args)

    # EBM-only metrics (always available).
    val_ebm, test_ebm = _compute_method_metrics(
        method="ebm_only",
        energies_val=e_ebm_val,
        energies_test=e_ebm_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
        calibrate_threshold_on_split=calibrate_threshold_on_split,
    )

    # If this run did not train an adaptive graph (e.g., TEM-only with
    # graph_mode == "none"), do not construct any graph_only / fusion
    # curves. This keeps graph-based analysis restricted to runs that
    # actually used a learned graph during training.
    if not use_graph_for_analysis:
        dataset = dataset_name
        experiment = run_dir.name
        val_parts: List[pd.DataFrame] = [val_ebm]
        test_parts: List[pd.DataFrame] = [test_ebm]
        for val_b, test_b in _selective_baseline_blocks(
            run_dir,
            calibrate_threshold_on_split,
            include_error_predictor,
            include_mc_dropout,
            mc_dropout_passes,
            error_predictor_mlp,
            error_predictor_fit_on,
            y_hat_val,
            y_hat_test,
            y_hat_train,
            mse_sel_train,
            mse_sel_val,
            mse_orig_val,
            mse_sel_test,
            mse_orig_test,
        ):
            val_parts.append(val_b)
            test_parts.append(test_b)
        val_all = pd.concat(val_parts, ignore_index=True)
        test_all = pd.concat(test_parts, ignore_index=True)
        for df in (val_all, test_all):
            df["experiment"] = experiment
            df["dataset"] = dataset
            df["model"] = model_name
        return val_all, test_all

    # Graph-structural energies from forecasts and adjacency.
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

    # Fusion: blend EBM and graph in ``fusion_mode`` space, then pick alpha on val.
    e_ebm_val_p, e_graph_val_p, e_ebm_test_p, e_graph_test_p = _prepare_fusion_channels(
        e_ebm_val,
        e_graph_val,
        e_ebm_test,
        e_graph_test,
        fusion_mode,
    )
    alphas = _fusion_alpha_candidates(fusion_alpha_step)
    best_alpha = _select_best_fusion_alpha(
        e_ebm_val_p,
        e_graph_val_p,
        mse_sel_val,
        mse_orig_val,
        candidate_alphas=alphas,
        interior_bias=fusion_interior_bias,
    )
    e_fused_val = best_alpha * e_ebm_val_p + (1.0 - best_alpha) * e_graph_val_p
    e_fused_test = best_alpha * e_ebm_test_p + (1.0 - best_alpha) * e_graph_test_p

    # Compute metrics for each method.
    val_graph, test_graph = _compute_method_metrics(
        method="graph_only",
        energies_val=e_graph_val,
        energies_test=e_graph_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
        calibrate_threshold_on_split=calibrate_threshold_on_split,
    )

    val_fused, test_fused = _compute_method_metrics(
        method="fusion",
        energies_val=e_fused_val,
        energies_test=e_fused_test,
        mse_sel_val=mse_sel_val,
        mse_orig_val=mse_orig_val,
        mse_sel_test=mse_sel_test,
        mse_orig_test=mse_orig_test,
        calibrate_threshold_on_split=calibrate_threshold_on_split,
    )

    # Record fusion settings for analysis.
    val_fused["fusion_alpha"] = best_alpha
    test_fused["fusion_alpha"] = best_alpha
    val_fused["fusion_mode"] = fusion_mode
    test_fused["fusion_mode"] = fusion_mode

    # Annotate with experiment metadata.
    # 对于 ETT 系列，args.data 本身就是 ETTh1/ETTm1 等；
    # 对于自定义数据（args.data == "custom"），用 data_path 的文件名
    # （去掉扩展名）来区分 electricity / exchange_rate / weather 等。
    # Reuse the dataset_name logic above for consistency.
    dataset = dataset_name

    experiment = run_dir.name
    val_parts = [val_ebm, val_graph, val_fused]
    test_parts = [test_ebm, test_graph, test_fused]
    for val_b, test_b in _selective_baseline_blocks(
        run_dir,
        calibrate_threshold_on_split,
        include_error_predictor,
        include_mc_dropout,
        mc_dropout_passes,
        error_predictor_mlp,
        error_predictor_fit_on,
        y_hat_val,
        y_hat_test,
        y_hat_train,
        mse_sel_train,
        mse_sel_val,
        mse_orig_val,
        mse_sel_test,
        mse_orig_test,
    ):
        val_parts.append(val_b)
        test_parts.append(test_b)

    val_all = pd.concat(val_parts, ignore_index=True)
    test_all = pd.concat(test_parts, ignore_index=True)
    for df in (val_all, test_all):
        df["experiment"] = experiment
        df["dataset"] = dataset
        df["model"] = model_name
    return val_all, test_all


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate selective-inference metrics (EBM, graph, fusion, optional baselines) "
            "from completed runs under a checkpoints tree. By default, **test** rows use "
            "**test**-set quantile thresholds (empirical test coverage matches target). "
            "The legacy flag --per-split-coverage is a no-op (kept for old scripts)."
        )
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="./checkpoints",
        help="Root directory to scan for run folders containing args.csv and result_objects/.",
    )
    parser.add_argument(
        "--require-output-parent-substr",
        type=str,
        default=None,
        help=(
            "If set (e.g. a batch output folder name), only include runs whose "
            "saved args.output_parent_path contains this substring."
        ),
    )
    parser.add_argument(
        "--per-split-coverage",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-per-split-coverage",
        action="store_true",
        help=(
            "Legacy: **test** rows use a single threshold from **val** energies (coverage on test may "
            "differ from target). Default: **test** quantile thresholds on **test** for all methods "
            "including baselines."
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="linear_z",
        choices=["linear_z", "rank_cdf"],
        help=(
            "Fusion channel alignment: linear_z (z-score each energy on val, default) or "
            "rank_cdf (map to [0,1] via val empirical CDF before blending; no retraining)."
        ),
    )
    parser.add_argument(
        "--fusion-alpha-step",
        type=float,
        default=None,
        metavar="STEP",
        help=(
            "Alpha grid step in [0,1], e.g. 0.05 for 21 values. "
            "Omit for legacy grid {0, 0.25, 0.5, 0.75, 1}."
        ),
    )
    parser.add_argument(
        "--fusion-interior-bias",
        type=float,
        default=0.0,
        help="Add lambda*(alpha-0.5)^2 to val fusion objective to favor mixed alphas when close.",
    )
    parser.add_argument(
        "--only-model",
        type=str,
        default=None,
        help=(
            "If set (e.g. FEDformer), only process runs whose args.model matches exactly. "
            "Output CSVs will contain only those runs (full overwrite, not merge)."
        ),
    )
    parser.add_argument(
        "--only-dataset",
        type=str,
        default=None,
        help=(
            "If set (e.g. traffic), only process runs whose dataset label matches: "
            "for custom data, the CSV stem (traffic for traffic.csv); else args.data. "
            "Output CSVs will contain only those runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "If set, write all outputs under this directory (global CSVs and "
            "ebm_graph_fusion_metrics/<model>/). Default is current directory, "
            "which overwrites ebm_graph_fusion_*.csv — use --output-dir to "
            "keep the original tables untouched."
        ),
    )
    parser.add_argument(
        "--include-error-predictor",
        action="store_true",
        help=(
            "Add selective baseline: Ridge/MLP to predict per-sample MSE from flattened y_hat. "
            "Fit split is chosen by --error-predictor-fit-on (default train). "
            "Requires result_obj_*_<split>_*.npz under result_objects/ (train cache from full TEM pipeline)."
        ),
    )
    parser.add_argument(
        "--error-predictor-mlp",
        action="store_true",
        help="Use a small MLP instead of Ridge for the error predictor (slower).",
    )
    parser.add_argument(
        "--error-predictor-fit-on",
        type=str,
        default="train",
        choices=["val", "train"],
        help=(
            "Where to fit the error predictor: train (default; no val labels for the head; "
            "needs result_obj_*_train_*.npz, else falls back to val) or val (legacy / replication)."
        ),
    )
    parser.add_argument(
        "--include-mc-dropout",
        action="store_true",
        help=(
            "Add MC-dropout baseline (checkpoint under args.checkpoints/<setting>/; slow, GPU recommended). "
            "Same test selective protocol as EBM/Graph by default."
        ),
    )
    parser.add_argument(
        "--mc-dropout-passes",
        type=int,
        default=20,
        help="Stochastic forward passes per batch for MC dropout (default: 20).",
    )
    cli = parser.parse_args()

    checkpoints_root = Path(cli.checkpoints_root)
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
    filt = cli.require_output_parent_substr
    if filt:
        run_dirs = [p for p in run_dirs if _run_matches_output_parent_filter(p, filt)]

    run_dirs = [
        p
        for p in run_dirs
        if _run_matches_model_dataset_filters(p, cli.only_model, cli.only_dataset)
    ]

    print(f"Found {len(run_dirs)} candidate runs with result_objects and args.csv")
    if filt:
        print(f"(filtered by output_parent_path containing {filt!r})")
    if cli.only_model or cli.only_dataset:
        print(
            f"(filtered by only_model={cli.only_model!r}, only_dataset={cli.only_dataset!r}; "
            "global CSVs will list only matching runs)"
        )
    out_base = Path(cli.output_dir) if cli.output_dir else Path(".")
    if cli.output_dir:
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"Writing outputs under {out_base.resolve()} (default files in CWD are not touched)")

    for run_dir in run_dirs:
        try:
            print(f"Processing run: {run_dir}")
            val_df, test_df = process_single_run(
                run_dir,
                calibrate_threshold_on_split=not cli.no_per_split_coverage,
                fusion_mode=cli.fusion_mode,
                fusion_alpha_step=cli.fusion_alpha_step,
                fusion_interior_bias=cli.fusion_interior_bias,
                include_error_predictor=cli.include_error_predictor,
                include_mc_dropout=cli.include_mc_dropout,
                mc_dropout_passes=cli.mc_dropout_passes,
                error_predictor_mlp=cli.error_predictor_mlp,
                error_predictor_fit_on=cli.error_predictor_fit_on,
            )
            all_val.append(val_df)
            all_test.append(test_df)
        except Exception as e:
            print(f"[WARN] Skipping run {run_dir} due to error: {e}")

    if not all_val:
        print("No valid multivariate runs processed; nothing to save.")
        return

    val_all = pd.concat(all_val, ignore_index=True)
    test_all = pd.concat(all_test, ignore_index=True)

    # Global aggregated CSVs (backward-compatible paths when --output-dir is unset).
    out_val = out_base / "ebm_graph_fusion_val_metrics.csv"
    out_test = out_base / "ebm_graph_fusion_test_metrics.csv"
    val_all.to_csv(out_val, index=False)
    test_all.to_csv(out_test, index=False)

    print(f"Saved aggregated validation metrics to {out_val.resolve()}")
    print(f"Saved aggregated test metrics to {out_test.resolve()}")

    # Additionally, store metrics grouped by backbone under
    # ebm_graph_fusion_metrics/<model>/... so that different
    # backbones' curves/results are cleanly separated.
    metrics_root = out_base / "ebm_graph_fusion_metrics"
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
