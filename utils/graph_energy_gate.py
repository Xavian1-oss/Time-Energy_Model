import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import unwrap_dataparallel


class Normalizer:
    """Simple z-score normalizer."""

    def __init__(self, eps: float = 1e-6):
        self.mean: float = 0.0
        self.std: float = 1.0
        self.eps: float = eps

    def fit(self, values: Tensor) -> None:
        values = values.detach().float()
        self.mean = values.mean().item()
        self.std = values.std(unbiased=False).item()

    def transform(self, values: Tensor) -> Tensor:
        return (values - self.mean) / (self.std + self.eps)


class DependencyGraphBuilder:
    """Interface for building a dependency matrix A(X) of shape (D, D)."""

    def build(
        self,
        batch_x: Tensor,
        exp: Optional[Any] = None,
        target_dims: Optional[Sequence[int]] = None,
    ) -> Tensor:
        raise NotImplementedError

class AdaptiveEmbeddingGraphBuilder(nn.Module):
    """Learnable adjacency via node embeddings.

    A = softmax(ReLU(E E^T)) over rows, where E are node embeddings.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # For high-dimensional graphs, a fully dense learned adjacency
        # can be noisy且难以泛化。这里内置一个简单的 top-k 稀疏策略：
        # - 当通道数较大时，仅保留每行前 top_k 个最大的关联；
        # - 对于低维数据（D 很小），则保持全连接以避免过度剪枝。
        # 这个值目前是经验默认值，如有需要可以在构建时改为可配置。 
        self.top_k = 10 if num_nodes > 10 else None

    def forward(self) -> Tensor:
        # [D, D] unnormalized affinities
        A = torch.matmul(self.node_emb, self.node_emb.t())
        A = F.relu(A)

        D = A.size(0)
        top_k = self.top_k
        if top_k is not None and 0 < top_k < D:
            # 对每一行仅保留 top_k 个最大的值，其余置零，形成稀疏图。
            values, indices = torch.topk(A, k=top_k, dim=1)
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            A = torch.where(mask, A, torch.zeros_like(A))

        # Row-normalize so each row sums to 1
        A = F.softmax(A, dim=1)
        return A


def build_correlation_adjacency_from_loader(
    data_loader: Iterable,
    device: torch.device,
    top_k: Optional[int] = None,
    add_self_loops: bool = True,
) -> Tensor:
    """Build a correlation-based adjacency matrix from a data loader.

    We treat each time step across all batches as an i.i.d. sample and
    compute a Spearman-like correlation across channels. The resulting
    matrix is converted to a non-negative, row-normalized adjacency.

    To make the graph more robust (especially in high dimensions), we
    optionally sparsify each row to keep only the top_k strongest
    correlations and/or add self-loops before row normalization.

    Args:
        data_loader: Iterable yielding (batch_x, batch_y, batch_x_mark, batch_y_mark).
        device: Torch device for the returned tensor.
        top_k: If provided and smaller than the number of channels D,
            each row keeps only its top_k largest affinities.
        add_self_loops: Whether to add identity connections before
            normalizing rows.

    Returns:
        A: [D, D] row-normalized adjacency tensor.
    """
    sum_x = None
    sum_x2 = None
    sum_cross = None
    count: int = 0

    for batch in data_loader:
        batch_x = batch[0]
        # batch_x: [B, L, D]
        B, L, D = batch_x.shape
        features = batch_x.reshape(-1, D).float()  # [B*L, D]

        if sum_x is None:
            sum_x = torch.zeros(D, dtype=torch.float64)
            sum_x2 = torch.zeros(D, dtype=torch.float64)
            sum_cross = torch.zeros(D, D, dtype=torch.float64)

        sum_x += features.sum(dim=0).double()
        sum_x2 += (features ** 2).sum(dim=0).double()
        sum_cross += features.t().double().mm(features.double())
        count += features.size(0)

    if count == 0:
        raise RuntimeError("Empty data loader when building correlation adjacency.")

    mean = sum_x / count  # [D]
    ex2 = sum_x2 / count  # [D]
    var = ex2 - mean ** 2
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)  # [D]

    exy = sum_cross / count  # [D, D]
    cov = exy - mean.view(-1, 1) * mean.view(1, -1)  # [D, D]

    std_outer = std.view(-1, 1) * std.view(1, -1)
    std_outer = torch.clamp(std_outer, min=1e-8)
    corr = cov / std_outer

    # Use absolute correlation as affinity.
    A_raw = torch.abs(corr)
    A_raw = torch.nan_to_num(A_raw, nan=0.0)

    D = A_raw.size(0)

    # Optional sparsification: keep only top_k neighbors per row.
    if top_k is not None and 0 < top_k < D:
        # For each row, zero out all but the top_k largest entries.
        values, indices = torch.topk(A_raw, k=top_k, dim=1)
        mask = torch.zeros_like(A_raw, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        A_raw = torch.where(mask, A_raw, torch.zeros_like(A_raw))

    # Optional self-loops before normalization.
    if add_self_loops:
        A_raw = A_raw + torch.eye(D, device=A_raw.device, dtype=A_raw.dtype)

    row_sums = A_raw.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    A = A_raw / row_sums
    return A.to(device=device, dtype=torch.float32)


def compute_structure_energy(y_hat: Tensor, A: Tensor) -> Tensor:
    """Dirichlet-style graph smoothness energy.

    Args:
        y_hat: [B, H, D] forecast.
        A:     [D, D] adjacency (row-normalized).

    Returns:
        E_struct: [B]
    """
    if y_hat.ndim != 3:
        raise ValueError("y_hat must have shape [B, H, D]")
    if A.ndim != 2 or A.size(0) != A.size(1):
        raise ValueError("A must have shape [D, D]")

    B, H, D = y_hat.shape
    if A.size(0) != D:
        raise ValueError(f"Adjacency size {A.size()} incompatible with D={D}")

    # Pairwise squared differences across channels, summed over horizon
    # y_hat: [B, H, D]
    diff = y_hat.unsqueeze(-1) - y_hat.unsqueeze(-2)  # [B, H, D, D]
    dist = (diff ** 2).sum(dim=1)  # [B, D, D]

    # Weight by adjacency
    E_struct = (A.unsqueeze(0) * dist).sum(dim=(1, 2))  # [B]
    # Normalize by number of pairs for scale stability
    E_struct = E_struct / float(D * D)
    return E_struct


def alignment_loss_struct_vs_mse(
    E_struct: Tensor,
    per_sample_mse: Tensor,
    eps: float = 1e-6,
    use_log_mse: bool = True,
) -> Tensor:
    """Differentiable alignment: encourage higher graph energy when MSE is higher.

    GraphEnergyGate accepts samples with *low* normalized structural energy;
    calibration thresholds the energy distribution. Thus we want a *positive*
    monotonic relationship: large per-sample MSE should co-occur with large
    E_struct on the same forecasts.

    Uses negative batch Pearson correlation between E_struct and (log) MSE.
    Minimizing this loss increases correlation. With batch size < 2, returns 0.

    Args:
        E_struct: [B] from ``compute_structure_energy(y_hat, A)``.
        per_sample_mse: [B] non-negative errors (typically detached).
        use_log_mse: Map MSE through log1p for heavy-tailed batch statistics.
    """
    if E_struct.ndim != 1 or per_sample_mse.ndim != 1:
        raise ValueError("E_struct and per_sample_mse must be 1D [B]")
    if E_struct.shape[0] != per_sample_mse.shape[0]:
        raise ValueError("Batch size mismatch between energy and MSE")

    B = E_struct.shape[0]
    if B < 2:
        return E_struct.new_zeros(())

    E = E_struct.float().reshape(-1)
    m = per_sample_mse.float().reshape(-1)
    if use_log_mse:
        m = torch.log1p(m.clamp_min(0.0))

    Ez = (E - E.mean()) / (E.std(unbiased=False) + eps)
    Mz = (m - m.mean()) / (m.std(unbiased=False) + eps)
    # Minimize negative correlation <=> maximize Pearson(E, m)
    return -(Ez * Mz).mean()


@dataclass
class GateConfig:
    """Configuration returned by calibration.

    The gate currently operates in a purely graph-structural mode, so
    mu_feat/std_feat/lambda_ are retained only for backward
    compatibility and are fixed to dummy values.
    """

    mu_feat: float
    std_feat: float
    mu_struct: float
    std_struct: float
    tau: float
    lambda_: float
    coverage: float
    achieved_coverage: float


def calibrate_gate(
    exp: Any,
    gate: "GraphEnergyGate",
    calib_loader,
    coverage: float,
    lambda_: float,
) -> GateConfig:
    """Run calibration over a loader and compute z-score stats and tau.

    Returns:
        GateConfig with statistics and threshold.
    """
    device = exp.device
    all_E_struct = []
    all_errors = []  # per-sample forecasting errors for selective risk

    with torch.no_grad():
        for batch in calib_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            E_struct, diagnostics = gate._compute_energies(
                exp,
                (batch_x, batch_y, batch_x_mark, batch_y_mark),
            )
            all_E_struct.append(E_struct.detach().cpu())
            # Optional: per-sample MSE for each validation sample
            per_sample_mse = diagnostics.get("per_sample_mse", None)
            if per_sample_mse is not None:
                all_errors.append(per_sample_mse.detach().cpu())

    if not all_E_struct:
        raise RuntimeError("Calibration loader is empty; cannot calibrate gate.")

    E_struct_all = torch.cat(all_E_struct, dim=0).float()
    errors_all = torch.cat(all_errors, dim=0).float() if all_errors else None

    norm_struct = Normalizer()
    norm_struct.fit(E_struct_all)
    E_struct_z_all = norm_struct.transform(E_struct_all)

    # 在当前实现中，门控完全基于结构能量 E_struct_z，
    # 不再混入任何 feature 头能量；因此 joint 能量就是
    # E_struct_z 本身。
    E_joint = E_struct_z_all
    E_joint_np = E_joint.numpy()
    tau = float(np.quantile(E_joint_np, coverage))
    accept_mask = E_joint <= tau
    achieved_coverage = float(accept_mask.float().mean().item())

    selected_lambda = 0.0

    return GateConfig(
        mu_feat=0.0,
        std_feat=1.0,
        mu_struct=float(norm_struct.mean),
        std_struct=float(norm_struct.std),
        tau=tau,
        lambda_=selected_lambda,
        coverage=coverage,
        achieved_coverage=achieved_coverage,
    )


class GraphEnergyGate:
    """Graph-structural selective gate for long-term forecasting models.

    特征能量完全交给外部的 TEM/EBM 流水线处理，本 gate 只基于
    图结构能量 E_struct 进行接收/拒绝决策（graph-only 模式）。
    """

    def __init__(
        self,
        dep_graph_builder: Optional[Callable[[], Tensor]] = None,
        lambda_: float = 0.6,
        coverage: float = 0.9,
    ):
        # 当前实现为纯 graph-only 门控，lambda_ 仅作为
        # 兼容字段保留，不参与计算。
        self.lambda_ = lambda_
        self.coverage = coverage

        # 依赖图构建器：可以是自适应的（例如 AdaptiveEmbeddingGraphBuilder），
        # 也可以是一个返回固定邻接矩阵的 lambda。
        self.dep_graph_builder = dep_graph_builder

        self.config: Optional[GateConfig] = None
        self.norm_struct = Normalizer()

    @property
    def is_calibrated(self) -> bool:
        return self.config is not None

    def _get_target_channel_indices(self, exp: Any, batch_x: Tensor, D: int) -> Sequence[int]:
        """Map local target indices [0..D-1] to global channel indices in x."""
        if getattr(exp.args, "features", "M") == "MS":
            # Single target is typically the last channel
            return [batch_x.size(-1) - 1]
        # Default: assume first D channels correspond to targets
        return list(range(D))

    def _base_forecast(self, exp: Any, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Sequence[int]]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        # decoder input as in TSL
        dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if getattr(exp.args, "features", "M") == "MS" else 0
        y_hat = outputs[:, -exp.args.pred_len :, f_dim:]
        D = y_hat.size(-1)
        target_indices = self._get_target_channel_indices(exp, batch_x, D)
        return y_hat, target_indices

    def _compute_energies(
        self,
        exp: Any,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        # Base forecast
        y_hat, target_indices = self._base_forecast(exp, batch)

        # Per-sample forecasting error (MSE over horizon and channels)
        f_dim = -1 if getattr(exp.args, "features", "M") == "MS" else 0
        y_true = batch_y[:, -exp.args.pred_len :, f_dim :].detach()
        per_sample_mse = ((y_hat.detach() - y_true) ** 2).mean(dim=(1, 2))

        # Dependency / adjacency
        D = y_hat.size(-1)
        if self.dep_graph_builder is not None:
            A = self.dep_graph_builder()
        else:
            A = torch.eye(D, device=y_hat.device, dtype=y_hat.dtype)

        E_struct = compute_structure_energy(y_hat, A)

        diagnostics = {
            "E_struct": E_struct,
            "per_sample_mse": per_sample_mse,
        }
        return E_struct, diagnostics

    def fit_calibration(
        self,
        exp: Any,
        calib_loader,
        coverage: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> GateConfig:
        if coverage is None:
            coverage = self.coverage
        if lambda_ is None:
            lambda_ = self.lambda_

        config = calibrate_gate(
            exp=exp,
            gate=self,
            calib_loader=calib_loader,
            coverage=coverage,
            lambda_=lambda_,
        )

        self.config = config
        self.lambda_ = config.lambda_
        self.coverage = config.coverage

        # Update internal normalizers
        self.norm_struct.mean = config.mu_struct
        self.norm_struct.std = config.std_struct

        print(
            f"[GraphEnergyGate] Calibration done: target_cov={coverage:.3f}, "
            f"achieved_cov={config.achieved_coverage:.3f}, tau={config.tau:.4f}"
        )
        return config

    def forward_gate(
        self,
        exp: Any,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if not self.is_calibrated:
            raise RuntimeError("GraphEnergyGate must be calibrated before calling forward_gate().")

        E_struct, diagnostics = self._compute_energies(exp, batch)

        # Normalize using calibration stats
        E_struct_z = self.norm_struct.transform(E_struct)

        # Graph-only joint energy.
        E_joint = E_struct_z

        tau = self.config.tau if self.config is not None else 0.0
        reject_mask = E_joint > tau

        diagnostics = {
            **diagnostics,
            "E_struct_z": E_struct_z,
            "E_joint": E_joint,
            "reject_mask": reject_mask,
        }
        return reject_mask, E_joint, diagnostics


def _evaluate_gate_on_loader(
    gate: GraphEnergyGate,
    exp: Any,
    data_loader: Iterable,
) -> Tuple[float, float, float]:
    """Evaluate a calibrated gate on a data loader.

    Returns:
        coverage: fraction of accepted samples (1 - reject_rate).
        mse_selected: MSE on accepted samples only.
        mse_all: MSE over all samples (original model risk).
    """
    all_mse = []
    all_reject = []

    device = exp.device

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.float().to(device) for t in batch)
            reject_mask, _E_joint, diagnostics = gate.forward_gate(exp, batch)
            per_sample_mse = diagnostics["per_sample_mse"].detach().cpu()
            all_mse.append(per_sample_mse)
            all_reject.append(reject_mask.detach().cpu())

    if not all_mse:
        raise RuntimeError("Empty data loader when evaluating gate.")

    mse_all_tensor = torch.cat(all_mse, dim=0).float()
    reject_tensor = torch.cat(all_reject, dim=0).bool()
    accept_mask = ~reject_tensor

    if accept_mask.sum() == 0:
        # No accepted samples; fall back to overall MSE.
        mse_selected = float(mse_all_tensor.mean().item())
        coverage = 0.0
    else:
        mse_selected = float(mse_all_tensor[accept_mask].mean().item())
        coverage = float(accept_mask.float().mean().item())

    mse_all = float(mse_all_tensor.mean().item())
    return coverage, mse_selected, mse_all


def run_graph_gate_evaluation(
    exp: Any,
    experiment_data: Any,
    parent_path_pics: str,
    coverages: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run graph-structural selective inference for multivariate tasks.

    This uses GraphEnergyGate in graph-only mode (feature head disabled)
    with a correlation-based adjacency built from the training loader.

    It produces coverage→MSE curves on both validation and test sets,
    with the same column names as the TEM noise-based analysis so that
    downstream tooling (e.g., summarization scripts) can consume them.
    """
    if coverages is None:
        coverages = [0.5, 0.6, 0.7, 0.8, 0.9]

    device = exp.device

    # Prefer a learned adaptive graph if available on the backbone;
    # otherwise fall back to a correlation-based adjacency built from
    # the training data.
    backbone = unwrap_dataparallel(exp.model)
    if hasattr(backbone, "dep_graph_builder") and backbone.dep_graph_builder is not None:
        print("[GraphEnergyGate] Using learned adaptive graph from model.dep_graph_builder()...")
        def_dep_builder: Callable[[], Tensor] = lambda: backbone.dep_graph_builder()
        dep_graph_builder = def_dep_builder
    else:
        print("[GraphEnergyGate] Building correlation-based adjacency from train loader...")
        # 使用稀疏相关图：每个节点仅保留若干最强相关的邻居，并加入自环，
        # 在高维数据集上通常比完全密集的相关矩阵更稳健。
        A = build_correlation_adjacency_from_loader(
            experiment_data.train_loader,
            device,
            top_k=10,
            add_self_loops=True,
        )
        dep_graph_builder = lambda: A

    gate = GraphEnergyGate(
        dep_graph_builder=dep_graph_builder,
        lambda_=0.0,
        coverage=coverages[-1],  # will be overridden per-coverage anyway
    )

    # Mark that we are using an adaptive graph so that calibration
    # automatically switches to graph-only mode.
    if not hasattr(exp.args, "use_adaptive_graph"):
        setattr(exp.args, "use_adaptive_graph", True)
    else:
        exp.args.use_adaptive_graph = True

    def build_metrics_for_split(split_name: str, data_loader: Iterable) -> pd.DataFrame:
        rows = []

        for target_cov in coverages:
            # Calibrate gate for this target coverage on the validation set.
            print(
                f"[GraphEnergyGate] Calibrating for coverage={target_cov:.2f} on validation set..."
            )
            config = gate.fit_calibration(
                exp=exp,
                calib_loader=experiment_data.val_loader,
                coverage=target_cov,
                lambda_=0.0,
            )

            # Evaluate on the requested split (val or test).
            cov_achieved, mse_selected, mse_all = _evaluate_gate_on_loader(
                gate, exp, data_loader
            )

            row = {
                "target_coverage": float(target_cov),
                # For compatibility with existing tooling, retain the
                # column name train_coverage even though this is the
                # empirical coverage of the current split.
                "train_coverage": float(cov_achieved),
                "split_mse_selected": float(mse_selected),
                "split_mse_orig": float(mse_all),
                "split": split_name,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    print("[GraphEnergyGate] Evaluating gate on validation and test loaders...")
    val_df = build_metrics_for_split("val", experiment_data.val_loader)
    test_df = build_metrics_for_split("test", experiment_data.test_loader)

    # Persist to CSV next to the TEM metrics for easy comparison.
    val_metrics_path = os.path.join(parent_path_pics, "graph_val_metrics_filtered.csv")
    test_metrics_path = os.path.join(parent_path_pics, "graph_test_metrics_filtered.csv")
    try:
        val_df.to_csv(val_metrics_path, index=False)
        test_df.to_csv(test_metrics_path, index=False)
        print(
            f"[GraphEnergyGate] Saved graph-based metrics to {val_metrics_path} and {test_metrics_path}"
        )
    except Exception as e:
        print(f"[GraphEnergyGate][WARN] Failed to save metrics CSVs: {e}")

    return val_df, test_df