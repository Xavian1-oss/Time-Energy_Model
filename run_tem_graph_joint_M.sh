#!/usr/bin/env bash

# Compare TEM-only (ebm_only), Graph-only (graph_only), and
# Joint (fusion = EBM + Graph) selective inference on multivariate (M)
# forecasting tasks.
#
# This script will:
#   1) Train backbone + EBM + adaptive graph head for each
#      (model, data_path) pair in M mode (graph_mode=auto).
#   2) Run online TEM selective inference and graph-based gate.
#   3) (Optional) Aggregate offline EBM-only / Graph-only / Fusion curves
#      using compare_ebm_graph_fusion.py.
#   4) (Optional) Plot test coverage→MSE curves per dataset via
#      plot_ebm_graph_fusion_curves.py.
#
# Usage:
#   chmod +x run_tem_graph_joint_M.sh
#   # 默认：跑 TEM + Graph + Fusion 全流程
#   ./run_tem_graph_joint_M.sh
#   # 只跑训练 + 在线 Graph gate，不做离线 EBM/Graph/Fusion 汇总和画图：
#   RUN_MODE=graph_only ./run_tem_graph_joint_M.sh
#
# You can edit the MODELS / DATA_PATHS arrays below to match the
# experiments you care about.

set -e

# RUN_MODE 控制是否执行离线 EBM/Graph/Fusion 汇总与画图：
#   all        : 默认，全流程（TEM + Graph + Fusion + plot）
#   graph_only : 只跑训练和在线 Graph gate，跳过 compare_ebm_graph_fusion.py
#                和 plot_ebm_graph_fusion_curves.py，方便快速看 Graph 头效果。
RUN_MODE=${RUN_MODE:-all}

# You can override this before calling the script if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Models to evaluate
MODELS=("PatchTST")

# Data paths (CSV filenames under ./dataset)
DATA_PATHS=(
  # "ETTh1.csv"
  # "ETTh2.csv"
  # "ETTm1.csv"
  "ETTm2.csv"
  "exchange_rate.csv"
  "national_illness.csv"
  "weather.csv"
  # "electricity.csv"
  # "traffic.csv"
)

OUT_ROOT="./all_runs_tem_graph_joint_M"
mkdir -p "${OUT_ROOT}"

for model in "${MODELS[@]}"; do
  for data_path in "${DATA_PATHS[@]}"; do
    exp_name="${model}_M_${data_path//\//_}"
    out_dir="${OUT_ROOT}/${exp_name}"
    mkdir -p "${out_dir}"

    echo "==============================================="
    echo "[TRAIN] model=${model}, data_path=${data_path}, features=M"
    echo "         graph_mode=auto (train adaptive graph head + run graph gate)"
    echo "Outputs (checkpoints, result_objects, analysis CSVs) go under the\ncheckpoints tree; out_dir only satisfies --output_parent_path."
    echo "==============================================="

    python run_ebmExp.py \
      --model "${model}" \
      --data_path "${data_path}" \
      --features M \
      --inference_strategy noise \
      --output_parent_path "${out_dir}" \
      --is_test_mode 0 \
      --graph_mode auto
  done
done

  if [ "${RUN_MODE}" = "all" ]; then
    echo "==============================================="
    echo "[OFFLINE] Aggregating EBM-only / Graph-only / Fusion curves..."
    echo "==============================================="
    python compare_ebm_graph_fusion.py

    echo "==============================================="
    echo "[PLOT] Drawing test coverage→MSE curves per dataset..."
    echo "==============================================="
    python plot_ebm_graph_fusion_curves.py

    echo "Done. Check ebm_graph_fusion_test_metrics.csv and plots_ebm_graph_fusion/*.png for TEM-only (ebm_only), Graph-only (graph_only), and Joint (fusion) results."
  else
    echo "==============================================="
    echo "[SKIP OFFLINE] RUN_MODE=${RUN_MODE}, skipping compare_ebm_graph_fusion.py and plot_ebm_graph_fusion_curves.py."
    echo "You can inspect per-run Graph gate metrics under each checkpoint's local_pics_*/graph_*_metrics_filtered.csv."
    echo "==============================================="
  fi
