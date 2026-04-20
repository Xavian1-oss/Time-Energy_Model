#!/usr/bin/env python3
"""Run ``compare_ebm_graph_fusion.py`` on a small fixed grid of models × datasets.

Default grid (edit ``MODELS`` / ``DATASETS`` in this file if needed):
  - Models: PatchTST, Autoformer
  - Datasets: ETTh2, electricity, weather

Each (model, dataset) pair is processed in isolation (matching compare's single-filter
CLI), then validation/test CSVs are concatenated under ``--merged-output-dir``.

Usage (from repo root)::

  python scripts/run_baseline_compare_grid.py --checkpoints-root ./checkpoints

By default each compare subprocess gets the same baseline flags as::

  --include-error-predictor --error-predictor-fit-on train \\
  --include-mc-dropout --mc-dropout-passes 20

Match your shell habit (log + pause)::

  ( python scripts/run_baseline_compare_grid.py --checkpoints-root ./checkpoints \\
      --run-and-pause ) 2>&1 | tee run.log

Unknown CLI flags are appended after those defaults (later flags win in compare).

Note: this script always passes ``--output-dir`` last per combo under
``<merged-output-dir>/combo_<model>_<dataset>/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMPARE_SCRIPT = Path(__file__).resolve().parent / "compare_ebm_graph_fusion.py"

MODELS = ["PatchTST", "Autoformer"]
DATASETS = ["ETTh2", "electricity", "weather"]

# Same defaults as: compare_ebm_graph_fusion.py ... --include-error-predictor
# --error-predictor-fit-on train --include-mc-dropout --mc-dropout-passes 20
COMPARE_DEFAULT_EXTRA: list[str] = [
    "--include-error-predictor",
    "--error-predictor-fit-on",
    "train",
    "--include-mc-dropout",
    "--mc-dropout-passes",
    "20",
]


def _run_one_combo(
    *,
    python: str,
    checkpoints_root: Path,
    model: str,
    dataset: str,
    combo_out: Path,
    extra_compare_args: list[str],
) -> int:
    combo_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(_COMPARE_SCRIPT),
        "--checkpoints-root",
        str(checkpoints_root),
        "--only-model",
        model,
        "--only-dataset",
        dataset,
    ]
    cmd.extend(extra_compare_args)
    # Last --output-dir wins in compare's argparse; keep per-combo paths for merging.
    cmd.extend(["--output-dir", str(combo_out)])
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=_REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-call compare_ebm_graph_fusion for PatchTST/Autoformer × "
            "ETTh2/electricity/weather and merge CSVs."
        )
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="./checkpoints",
        help="Same as compare's --checkpoints-root (default: ./checkpoints).",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="./baseline_compare_grid_merged",
        help="Directory for per-combo subfolders and merged ebm_graph_fusion_*.csv.",
    )
    parser.add_argument(
        "--no-default-baselines",
        action="store_true",
        help=(
            "Do not prepend error-predictor (train-fit) + MC-dropout (20 passes); "
            "only pass unknown args to compare."
        ),
    )
    parser.add_argument(
        "--run-and-pause",
        action="store_true",
        help="After the grid finishes, run python run_and_pause.py from repo root.",
    )
    cli, extra = parser.parse_known_args()

    checkpoints_root = Path(cli.checkpoints_root).expanduser().resolve()
    merged_root = Path(cli.merged_output_dir).expanduser().resolve()
    merged_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    compare_tail = list(extra)
    if not cli.no_default_baselines:
        compare_tail = COMPARE_DEFAULT_EXTRA + compare_tail

    codes: list[int] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for model in MODELS:
        for dataset in DATASETS:
            safe = f"{model}_{dataset}".replace("/", "_")
            combo_out = merged_root / f"combo_{safe}"
            code = _run_one_combo(
                python=python,
                checkpoints_root=checkpoints_root,
                model=model,
                dataset=dataset,
                combo_out=combo_out,
                extra_compare_args=compare_tail,
            )
            codes.append(code)

            val_path = combo_out / "ebm_graph_fusion_val_metrics.csv"
            test_path = combo_out / "ebm_graph_fusion_test_metrics.csv"
            if val_path.is_file():
                val_parts.append(pd.read_csv(val_path))
            else:
                print(f"[WARN] Missing {val_path} (no runs or compare failed).", flush=True)
            if test_path.is_file():
                test_parts.append(pd.read_csv(test_path))
            else:
                print(f"[WARN] Missing {test_path} (no runs or compare failed).", flush=True)

    if val_parts:
        val_merged = pd.concat(val_parts, ignore_index=True)
        out_val = merged_root / "ebm_graph_fusion_val_metrics.csv"
        val_merged.to_csv(out_val, index=False)
        print(f"\nMerged val metrics -> {out_val}", flush=True)
    else:
        print("\n[WARN] No val CSVs to merge.", flush=True)

    if test_parts:
        test_merged = pd.concat(test_parts, ignore_index=True)
        out_test = merged_root / "ebm_graph_fusion_test_metrics.csv"
        test_merged.to_csv(out_test, index=False)
        print(f"Merged test metrics -> {out_test}", flush=True)
    else:
        print("[WARN] No test CSVs to merge.", flush=True)

    if cli.run_and_pause:
        pause_script = _REPO_ROOT / "run_and_pause.py"
        if not pause_script.is_file():
            print(f"[WARN] Missing {pause_script}; skip --run-and-pause.", flush=True)
        else:
            pause_cmd = [python, str(pause_script)]
            print(f"\n>>> {' '.join(pause_cmd)}\n", flush=True)
            subprocess.call(pause_cmd, cwd=_REPO_ROOT)

    worst = max(codes) if codes else 0
    if worst != 0:
        sys.exit(worst)


if __name__ == "__main__":
    main()
