#!/usr/bin/env bash

# Run PatchTST and Autoformer on multiple datasets in M mode
# Usage:
#   chmod +x run_all_patchtst_autoformer_M.sh
#   ./run_all_patchtst_autoformer_M.sh

set -e

# You can override this before calling the script if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODELS=("PatchTST")

# Data paths are relative to the default root_path (./dataset)
# Datasets in this repo live directly under ./dataset, not in
# per-dataset subfolders, so we just pass the CSV filenames.
DATA_PATHS=(
  # "ETTm1.csv"
  # "ETTm2.csv"
  # "exchange_rate.csv"
  "electricity.csv"
  "traffic.csv"
  "weather.csv"
  "national_illness.csv"
)

OUT_ROOT="./all_runs_PatchTST_Autoformer_M"
mkdir -p "${OUT_ROOT}"

for model in "${MODELS[@]}"; do
  for data_path in "${DATA_PATHS[@]}"; do
    exp_name="${model}_M_${data_path//\//_}"
    out_dir="${OUT_ROOT}/${exp_name}"
    mkdir -p "${out_dir}"

    echo "==============================================="
    echo "Running model=${model}, data_path=${data_path}, features=M"
    echo "Outputs (args.csv, checkpoints, analysis CSVs) will go under the default checkpoints tree;"
    echo "this script only uses out_dir to satisfy --output_parent_path."
    echo "==============================================="

    python run_ebmExp.py \
      --model "${model}" \
      --data_path "${data_path}" \
      --features M \
      --inference_strategy noise \
      --output_parent_path "${out_dir}" \
      --is_test_mode 0
  done
done
