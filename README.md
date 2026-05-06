# Time–Energy Model (NeoEBM / TEM)

Time-series forecasting experiments with **long-horizon backbones** (Autoformer, FEDformer, Informer, PatchTST, TimesNet), an optional **energy-based model (EBM)** on top of the backbone, **test-time energy / optimisation (TEM)**, learned **graph-structural energies** on multivariate tasks, and offline **fusion / baseline** comparisons (error predictor, MC dropout).

---

## Features

- **Backbone training** with optional **adaptive graph** regularisation (`use_adaptive_graph`, multivariate `M`).
- **EBM training** (contrastive-style energy on `xy_decoder`, optional staged `y_encoder`).
- **`run_ebmExp.py`**: end-to-end training, checkpointing, TEM inference, graph-gate evaluation, `learned_graph_A.npy`, and NPZ **`result_objects/`** for later analysis.
- **`scripts/compare_ebm_graph_fusion.py`**: scan `checkpoints/…` for completed runs; compute selective metrics (EBM-only, graph-only, fusion, optional baselines).
- **Warm-start**: if backbone checkpoint (and EBM artefact when `ebm_mode=auto`) already exist, training can be skipped and weights reloaded (see below).

---

## Requirements

- **Python 3** (3.9+ recommended).
- **PyTorch** with CUDA if you use `--use_gpu` (install from [pytorch.org](https://pytorch.org) to match your platform).
- Other Python deps:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists libraries such as `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `clearml`, etc. It does **not** pin a specific `torch` build; install PyTorch separately.

---

## Data

Place datasets under `dataset/` (default `root_path` is `./dataset`).  
Examples: `ETTh1.csv`, `traffic.csv`, `electricity.csv`, `weather.csv`, …

`scripts/run_ebmExp.py` maps `--data_path` prefixes to `args.data` and channel counts (e.g. ETT names, traffic/electricity/weather dimensions).

---

## Quick start (training + analysis)

Run from the **repository root** so imports resolve.

### Minimal experiment

```bash
python scripts/run_ebmExp.py \
  --model PatchTST \
  --data_path ETTh2.csv \
  --features M \
  --output_parent_path ./exports/my_run
```

Important CLI flags:

| Flag | Role |
|------|------|
| `--model` | `TimesNet`, `Autoformer`, `Informer`, `FEDformer`, `PatchTST` |
| `--data_path` | CSV name under `dataset/` |
| `--features` | `S` / `MS` / `M` (multivariate all channels = `M`) |
| `--output_parent_path` | Root folder for checkpoints / logs (`args.output_parent_path`) |
| `--graph_mode` | `auto` (default): graph on multivariate tasks; `none`: TEM/EBM only path for graph-off |
| `--ebm_mode` | `auto` (default): train EBM + TEM; `none`: graph-only backbone path, skip EBM |
| `--is_test_mode` | `1` uses fewer epochs/iterations for smoke tests |

Many hyperparameters (`train_epochs`, `batch_size`, `ebm_epochs`, GPU, checkpoints dir, …) live in `get_default_args()` and dataset/model helpers inside `scripts/run_ebmExp.py`.

### Checkpoints and warm-start

- Checkpoints live under `./checkpoints/<setting>/…` plus EBM paths derived from `get_full_ebm_path`, etc.
- If **`checkpoint*.pth`** exists for the resolved `setting`, and (**`ebm_mode=none`** *or* **`full_ebm.pth`** exists when training EBM), and you did **not** set `force_retrain_orig_model` / `force_retrain_y_enc` / `force_retrain_xy_dec`, the script prints **`[WarmStart]`**, loads the backbone, and **skips** `train()` / `train_energy()`.
- To force a full retrain, use the `force_retrain_*` flags or remove the relevant files.

---

## Offline comparison (`compare_ebm_graph_fusion`)

Aggregates selective-inference curves from runs that contain **`args.csv`** and **`result_objects/`** (NPZ caches from the TEM pipeline).

```bash
python scripts/compare_ebm_graph_fusion.py \
  --checkpoints-root ./checkpoints \
  --only-model FEDformer \
  --only-dataset traffic \
  --output-dir ./exports/compare_out
```

- **Default**: test-split metrics use thresholds from **test** energies so empirical coverage matches targets.
- **`--no-per-split-coverage`**: use **validation** thresholds on **test** (legacy / transductive calibration style).

Optional baselines:

- `--include-error-predictor` (with `--error-predictor-fit-on {train,val}`).
- `--include-mc-dropout` (with `--mc-dropout-passes N`; slow, GPU recommended).

---

## Batch grid over models × datasets

```bash
python scripts/run_baseline_compare_grid.py \
  --checkpoints-root ./checkpoints \
  --merged-output-dir ./baseline_compare_grid_merged \
  --models "PatchTST,Autoformer,FEDformer" \
  --datasets "ETTh2,electricity,weather"
```

By default it forwards error-predictor (train-fit) + MC dropout (20 passes) to each subprocess unless you pass `--no-default-baselines`. Optional `--run-and-pause` runs `run_and_pause.py` after the grid.

---

## Other scripts (under `scripts/`)

| Script | Purpose |
|--------|---------|
| `plot_ebm_graph_fusion_curves.py` | Plot MSE vs coverage from compare outputs |
| `export_*_latex.py` / `export_ebm_graph_fusion_unified_table.py` | LaTeX tables from CSVs |
| `graph_only_selective_analysis.py` | Graph-only selective analysis over a checkpoint tree |
| `summarize_mse_changes.py` | Summarise MSE deltas |

Generated paper-style fragments may live under `docs/` (`.tex`).

---

## Repository layout (high level)

| Path | Content |
|------|---------|
| `exp/` | `Exp_Main_Energy`, training / validation / EBM loops |
| `energy/` | NeoEBM modules (concat / transformer-style heads) |
| `models/` | Backbone implementations |
| `data_provider/` | Loaders, `ExperimentData` |
| `tem_inferencer.py` | TEM / `test_adhoc_energy`, NPZ-friendly outputs |
| `utils/graph_energy_gate.py` | Graph energy, calibration, evaluation hooks |
| `utils/selective_baselines.py` | Error predictor, MC dropout helpers |
| `analysis/` | Selective metrics helpers used by compare |
| `scripts/` | CLI entry points |
| `dataset/` | Input CSVs (not always versioned) |

---

## Notes

- **Multivariate (`M`)**: graph regularisation and graph-gate paths expect multivariate structure; MC dropout in compare currently requires `features == "M"` in `args.csv`.
- **Logging**: experiments can integrate **ClearML** (`clearml` in requirements); disable or use test mode if you do not need remote logging.
- **Security**: review any local helpers (e.g. pause / automation scripts) for hard-coded tokens before sharing or running on shared machines.

---

## Citation

If you use this codebase in research, cite the relevant paper or technical report for your project and list this repository URL.
