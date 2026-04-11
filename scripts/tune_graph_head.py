#!/usr/bin/env python3
"""
Grid search for graph-head weights `gate_graph_loss_weight` and `gate_graph_align_weight`
given a backbone (`--model`) and dataset (`--data_path`).

Each trial runs `run_ebmExp.py` with `--features M`, a unique `--des`, and a dedicated
`--output_parent_path`. After training + graph gate evaluation, validation metrics are read
from `graph_val_metrics_filtered.csv` (under `analysis_dir` in `run_summary.txt`).

By default the score is validation `split_mse_selected` at the coverage row closest to
`--target_coverage` (graph selective prediction). Lower is better.

Example (graph-only, same spirit as RUN_MODE=graph_only in scripts/run_tem_graph_joint_M.sh):

  python scripts/tune_graph_head.py \\
    --model Autoformer \\
    --data_path ETTh1.csv \\
    --loss-grid 0.05,0.1,0.2 \\
    --align-grid 0.02,0.05,0.1 \\
    --ebm_mode none
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [float(p) for p in parts if p]


def _des_token(w_loss: float, w_align: float) -> str:
    """Filesystem-safe unique tag for checkpoint `des` (no dots)."""
    g = str(w_loss).replace(".", "p").replace("-", "m")
    a = str(w_align).replace(".", "p").replace("-", "m")
    return f"tune_gloss_{g}_galign_{a}"


def _read_analysis_dir(run_summary: Path) -> Optional[Path]:
    if not run_summary.is_file():
        return None
    text = run_summary.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^analysis_dir:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    return Path(m.group(1).strip())


def _score_split_metrics(
    metrics_csv: Path,
    split_name: str,
    target_coverage: float,
    use_selected: bool,
) -> Optional[float]:
    if not metrics_csv.is_file():
        return None
    col = "split_mse_selected" if use_selected else "split_mse_orig"
    rows: List[dict] = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or col not in reader.fieldnames:
            return None
        for row in reader:
            if str(row.get("split", "")) != split_name:
                continue
            try:
                cov = float(row["target_coverage"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((cov, row))
    if not rows:
        return None
    _, best_row = min(rows, key=lambda t: abs(t[0] - float(target_coverage)))
    try:
        return float(best_row[col])
    except (TypeError, ValueError):
        return None


def _run_one(
    python: str,
    repo: Path,
    model: str,
    data_path: str,
    out_dir: Path,
    w_loss: float,
    w_align: float,
    extra_args: List[str],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    des = _des_token(w_loss, w_align)
    cmd = [
        python,
        str(repo / "scripts" / "run_ebmExp.py"),
        "--model",
        model,
        "--data_path",
        data_path,
        "--features",
        "M",
        "--inference_strategy",
        "noise",
        "--output_parent_path",
        str(out_dir),
        "--is_test_mode",
        "0",
        "--graph_mode",
        "auto",
        "--gate_graph_loss_weight",
        str(w_loss),
        "--gate_graph_align_weight",
        str(w_align),
        "--des",
        des,
    ]
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=str(repo))
    return int(proc.returncode)


def main() -> None:
    repo = _repo_root()
    parser = argparse.ArgumentParser(
        description="Grid search gate_graph_loss_weight x gate_graph_align_weight for graph head."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="./graph_head_tune_sweeps",
        help="Parent directory; each trial uses a subfolder under this path.",
    )
    parser.add_argument(
        "--loss-grid",
        type=str,
        required=True,
        help="Comma-separated list of gate_graph_loss_weight values, e.g. 0.05,0.1,0.2",
    )
    parser.add_argument(
        "--align-grid",
        type=str,
        required=True,
        help="Comma-separated list of gate_graph_align_weight values, e.g. 0.02,0.05,0.1",
    )
    parser.add_argument(
        "--ebm_mode",
        type=str,
        default="none",
        choices=["auto", "none"],
        help="Default 'none' matches graph-only training (no EBM); use 'auto' for full joint training.",
    )
    parser.add_argument(
        "--target_coverage",
        type=float,
        default=0.9,
        help="Pick validation row with target_coverage closest to this value.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mse_selected",
        choices=["mse_selected", "mse_orig"],
        help="Column in graph_val_metrics_filtered.csv (split=val).",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If set, continue the grid after a failed run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned runs and exit without executing.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed to run_ebmExp.py (prefix with -- if needed).",
    )

    args = parser.parse_args()
    loss_weights = _parse_float_list(args.loss_grid)
    align_weights = _parse_float_list(args.align_grid)
    if not loss_weights or not align_weights:
        parser.error("Both --loss-grid and --align-grid must contain at least one number.")

    use_selected = args.metric == "mse_selected"
    extra: List[str] = list(args.extra)
    if "--ebm_mode" not in extra:
        extra.extend(["--ebm_mode", args.ebm_mode])

    stem = Path(args.data_path).stem.replace(".", "_")
    sweep_name = f"{args.model}_{stem}"
    output_root = (repo / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    sweep_dir = output_root / sweep_name

    trials: List[Tuple[float, float, Path]] = []
    for wl in loss_weights:
        for wa in align_weights:
            tag = _des_token(wl, wa)
            trials.append((wl, wa, sweep_dir / tag))

    if args.dry_run:
        print(f"Planned {len(trials)} runs under {sweep_dir}")
        for wl, wa, od in trials:
            print(f"  gate_graph_loss_weight={wl}  gate_graph_align_weight={wa}  -> {od}")
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    rows: List[dict] = []

    for wl, wa, trial_dir in trials:
        rc = _run_one(
            python_exe,
            repo,
            args.model,
            args.data_path,
            trial_dir,
            wl,
            wa,
            extra,
        )
        summary = trial_dir / "run_summary.txt"
        analysis = _read_analysis_dir(summary)
        val_csv = (analysis / "graph_val_metrics_filtered.csv") if analysis else None
        score = (
            _score_split_metrics(val_csv, "val", args.target_coverage, use_selected)
            if val_csv
            else None
        )
        test_csv = (analysis / "graph_test_metrics_filtered.csv") if analysis else None
        test_score = (
            _score_split_metrics(test_csv, "test", args.target_coverage, use_selected)
            if test_csv
            else None
        )

        rows.append(
            {
                "gate_graph_loss_weight": wl,
                "gate_graph_align_weight": wa,
                "output_parent_path": str(trial_dir),
                "returncode": rc,
                "val_score": score,
                "test_score_at_same_cov": test_score,
                "analysis_dir": str(analysis) if analysis else "",
            }
        )
        if rc != 0 and not args.continue_on_error:
            print(f"[tune_graph_head] Stopping after failed run (returncode={rc}).", file=sys.stderr)
            break

    results_csv = sweep_dir / "tune_graph_head_results.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "gate_graph_loss_weight",
        "gate_graph_align_weight",
        "output_parent_path",
        "returncode",
        "val_score",
        "test_score_at_same_cov",
        "analysis_dir",
    ]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["returncode"] == 0 and r["val_score"] is not None]
    if ok:
        best = min(ok, key=lambda r: float(r["val_score"]))
        print("\n=== Best (min val_score) ===")
        print(
            f"  gate_graph_loss_weight={best['gate_graph_loss_weight']}, "
            f"gate_graph_align_weight={best['gate_graph_align_weight']}"
        )
        print(f"  val_score={best['val_score']:.6g}  (metric={args.metric}, cov≈{args.target_coverage})")
        if best.get("test_score_at_same_cov") is not None:
            print(f"  test_score at same coverage row: {best['test_score_at_same_cov']:.6g}")
        print(f"  output_parent_path={best['output_parent_path']}")
    else:
        print(
            "\n[tune_graph_head] No successful run with val metrics; see tune_graph_head_results.csv for details.",
            file=sys.stderr,
        )

    print(f"\nWrote {results_csv}")


if __name__ == "__main__":
    main()
