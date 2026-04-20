"""Post-hoc selective inference baselines (error predictor, MC dropout).

Used by scripts/compare_ebm_graph_fusion.py. Energies are scalars per sample where
**lower** is kept first (same convention as EBM / graph energies).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_provider.experiment_data import ExperimentData
from models import Autoformer, FEDformer, Informer, PatchTST, TimesNet
from run_commons import ExperimentConstants
from utils.graph_energy_gate import AdaptiveEmbeddingGraphBuilder


def _is_blank_or_nan(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and not np.isfinite(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "none", "nan"):
        return True
    return False


def _flatten_y_hat(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y.reshape(y.shape[0], -1)


def _per_sample_mse_for_error_head(mse: np.ndarray, n_samples: int) -> np.ndarray:
    """Match ``compare_ebm_graph_fusion._compute_method_metrics._to_1d`` semantics.

    Cached ``mse_init_orig_model`` may be ``[N]``, ``[N, D]``, or raveled ``[N * D]``
    (per-channel / extra dims); we reduce to one scalar MSE per forecast window.
    """
    m = np.asarray(mse, dtype=np.float64)
    if m.shape[0] == n_samples:
        if m.ndim == 1:
            return m.reshape(-1)
        return m.mean(axis=tuple(range(1, m.ndim))).reshape(-1)
    if m.size == n_samples:
        return m.reshape(-1)
    if m.size % n_samples != 0:
        raise ValueError(
            f"Cannot align mse shape {m.shape} (size {m.size}) with n_samples={n_samples}"
        )
    return m.reshape(n_samples, -1).mean(axis=1)


def error_predictor_energies(
    y_hat_val: np.ndarray,
    mse_val: np.ndarray,
    y_hat_test: np.ndarray,
    *,
    y_hat_fit: Optional[np.ndarray] = None,
    mse_fit: Optional[np.ndarray] = None,
    use_mlp: bool = False,
    target_log1p: bool = True,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict per-sample MSE from forecast tensor; use prediction as abstention energy.

    By default fits on **validation** (``y_hat_val``, ``mse_val``). If ``y_hat_fit`` and
    ``mse_fit`` are set, fits on that split instead (e.g. train cache) and still scores
    val and test. Lower predicted error => more confident (low-energy kept first).
    """
    X_val = _flatten_y_hat(y_hat_val)
    X_test = _flatten_y_hat(y_hat_test)

    if y_hat_fit is not None and mse_fit is not None:
        X_fit = _flatten_y_hat(y_hat_fit)
        y_tr = _per_sample_mse_for_error_head(
            np.asarray(mse_fit, dtype=np.float64), X_fit.shape[0]
        )
    else:
        X_fit = X_val
        y_tr = _per_sample_mse_for_error_head(
            np.asarray(mse_val, dtype=np.float64), X_val.shape[0]
        )

    target = np.log1p(y_tr) if target_log1p else y_tr.copy()

    if use_mlp:
        reg: Any = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        random_state=random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        )
    else:
        reg = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )

    reg.fit(X_fit, target)
    pred_val = reg.predict(X_val).astype(np.float64)
    pred_test = reg.predict(X_test).astype(np.float64)
    return pred_val.ravel(), pred_test.ravel()


def _sanitize_exp_args(ns: SimpleNamespace) -> SimpleNamespace:
    """Coerce common numeric args loaded from CSV so Exp / data loaders work."""
    int_keys = (
        "seq_len",
        "label_len",
        "pred_len",
        "enc_in",
        "dec_in",
        "c_out",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "factor",
        "embed",
        "distil",
        "batch_size",
        "num_workers",
        "itr",
        "train_epochs",
        "patience",
        "ebm_epochs",
        "gpu",
        "is_training",
        "only_rerun_inference",
        "should_log",
        "is_test_mode",
        "use_amp",
        "output_attention",
    )
    d = vars(ns).copy()
    for k in int_keys:
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if isinstance(v, float) and np.isfinite(v) and float(int(v)) == v:
            d[k] = int(v)
        elif isinstance(v, str) and v.isdigit():
            d[k] = int(v)
    return SimpleNamespace(**d)


def setting_from_args(args: SimpleNamespace, itr_index: int = 0) -> str:
    """Rebuild training `setting` string (see scripts/run_ebmExp.py)."""
    data_path = str(getattr(args, "data_path", "")).replace(".", "_")
    if _is_blank_or_nan(data_path) or data_path.lower() == "nan":
        data_path = ""
    site_suffix = ""
    sid = getattr(args, "site_id", "None")
    if not _is_blank_or_nan(sid) and str(sid).strip() != "None":
        site_suffix = f"_{str(sid).replace(',', '_')}"
    return (
        "{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            ExperimentConstants.SETTINGS_PREFIX,
            args.model_id,
            args.model,
            args.data,
            data_path,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            itr_index,
        )
        + site_suffix
    )


def _checkpoint_seed_postfix(args: SimpleNamespace) -> str:
    return (
        f"_{args.ebm_seed}" if getattr(args, "ebm_seed", 42) not in (42, 2021) else ""
    )


def _resolve_model_checkpoint(args: SimpleNamespace, setting: str) -> Optional[str]:
    """Prefer checkpoint{{seed}}.pth, then checkpoint.pth, then any checkpoint*.pth."""
    base = os.path.join(str(args.checkpoints), setting)
    sp = _checkpoint_seed_postfix(args)
    candidates = [
        os.path.join(base, f"checkpoint{sp}.pth"),
        os.path.join(base, "checkpoint.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    if os.path.isdir(base):
        matches = sorted(Path(base).glob("checkpoint*.pth"))
        if matches:
            return str(matches[0])
    return None


def _setting_from_run_dir_layout(run_dir: Path) -> Optional[str]:
    """Infer setting folder from ``.../<setting>/neoebm_*/<run_id>`` (see run_ebmExp layout)."""
    try:
        p = run_dir.resolve()
        parent = p.parent
        if parent.name.startswith("neoebm"):
            grand = parent.parent
            if grand.name:
                return str(grand.name)
    except (OSError, ValueError):
        pass
    return None


def _setting_candidates_for_mc(run_dir: Path, args: SimpleNamespace) -> List[str]:
    out: List[str] = []
    s0 = setting_from_args(args, itr_index=0)
    if s0:
        out.append(s0)
    s1 = _setting_from_run_dir_layout(run_dir)
    if s1 and s1 not in out:
        out.append(s1)
    return out


def _build_backbone(args: SimpleNamespace) -> nn.Module:
    model_dict = {
        "Autoformer": Autoformer,
        "Informer": Informer,
        "FEDformer": FEDformer,
        "TimesNet": TimesNet,
        "PatchTST": PatchTST,
    }
    if args.model not in model_dict:
        raise ValueError(f"Unknown model {args.model!r} for MC dropout")
    model = model_dict[args.model].Model(args).float()
    if getattr(args, "use_adaptive_graph", False):
        num_nodes = getattr(args, "enc_in", getattr(args, "c_out", None))
        if num_nodes is not None and num_nodes > 0:
            dep_builder = AdaptiveEmbeddingGraphBuilder(
                num_nodes=num_nodes, embed_dim=16
            )
            setattr(model, "dep_graph_builder", dep_builder)
    return model


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _forward_prediction(
    model: nn.Module,
    args: SimpleNamespace,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
) -> torch.Tensor:
    device = batch_x.device
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = (
        torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
    )
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    if "DLinear" in args.model:
        outputs = model(batch_x)
    else:
        if getattr(args, "output_attention", False):
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if args.features == "MS" else 0
    return outputs[:, -args.pred_len :, f_dim:]


def _mc_dropout_scores_for_loader(
    model: nn.Module,
    args: SimpleNamespace,
    loader,
    n_passes: int,
    device: torch.device,
) -> List[float]:
    """Mean predictive variance over pred window and channels; higher = less certain."""
    scores: List[float] = []
    model.train()
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            preds = []
            for _ in range(n_passes):
                y_hat = _forward_prediction(
                    model, args, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                preds.append(y_hat)
            stacked = torch.stack(preds, dim=0)  # [T, B, H, D]
            var = stacked.var(dim=0, unbiased=False).mean(dim=(1, 2))
            scores.extend(var.detach().cpu().numpy().ravel().tolist())
    return scores


def mc_dropout_energies_from_run_dir(
    run_dir: Path,
    n_passes: int = 20,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load backbone from checkpoint next to args.csv; return (val_energy, test_energy).

    Energies = mean predictive variance (higher => reject first under ascending sort).
    Returns None if checkpoint missing or any failure.
    """
    args_path = run_dir / "args.csv"
    if not args_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(args_path)
        if df.empty:
            return None
        raw = df.iloc[0].to_dict()
        clean: dict = {}
        for k, v in raw.items():
            if pd.isna(v):
                if k in ("site_id", "target_site_id"):
                    clean[k] = "None"
                else:
                    clean[k] = v
                continue
            if isinstance(v, np.bool_):
                clean[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            else:
                clean[k] = v
        for key in ("site_id", "target_site_id"):
            if key in clean and _is_blank_or_nan(clean[key]):
                clean[key] = "None"
        args = _sanitize_exp_args(SimpleNamespace(**clean))
    except Exception:
        return None

    if getattr(args, "features", None) != "M":
        return None

    if not hasattr(args, "should_log"):
        args.should_log = False
    args.is_test_mode = 1

    try:
        ckpt_path: Optional[str] = None
        tried_settings: List[str] = []
        for setting in _setting_candidates_for_mc(run_dir, args):
            tried_settings.append(setting)
            ckpt_path = _resolve_model_checkpoint(args, setting)
            if ckpt_path:
                break
        if not ckpt_path:
            print(
                f"[MC-Dropout] No checkpoint under {args.checkpoints!r} for settings "
                f"{tried_settings!r} (tried checkpoint*.pth)."
            )
            return None

        model = _build_backbone(args)
        device = torch.device(
            "cuda:0"
            if getattr(args, "use_gpu", False) and torch.cuda.is_available()
            else "cpu"
        )
        if device.type == "cuda":
            model = model.to(device)
        else:
            model = model.to(device)

        state = torch.load(ckpt_path, map_location=device)
        state = _strip_module_prefix(state)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[MC-Dropout] Non-strict load (missing={len(missing)}, unexpected={len(unexpected)})"
                )

        experiment_data = ExperimentData.from_args(args)
        val_loader = experiment_data.val_loader
        test_loader = experiment_data.test_loader

        e_val = np.asarray(
            _mc_dropout_scores_for_loader(model, args, val_loader, n_passes, device),
            dtype=np.float64,
        )
        e_test = np.asarray(
            _mc_dropout_scores_for_loader(model, args, test_loader, n_passes, device),
            dtype=np.float64,
        )
        return e_val, e_test
    except Exception as e:
        print(f"[MC-Dropout] Skipped for {run_dir}: {e}")
        return None
