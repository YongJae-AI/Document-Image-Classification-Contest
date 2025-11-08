#!/usr/bin/env python
"""
Aggregate experiments and produce PPT-ready summary artifacts.

Outputs
-------
- reports/summary/<ts>/experiments_summary.csv
- reports/summary/<ts>/f1_by_epoch.png
- reports/summary/<ts>/best_f1_bar.png
- reports/summary/<ts>/val_loss_by_epoch.png
- reports/summary/<ts>/README.md (markdown with embedded charts)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def scan_runs(outputs_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    runs_dir = outputs_root / "runs"
    for run in sorted(runs_dir.glob("*")):
        metrics_path = run / "metrics.jsonl"
        config_path = run / "config.yaml"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_json(metrics_path, lines=True)
        except ValueError:
            continue
        if df.empty:
            continue
        best_row = df.sort_values("f1", ascending=False).iloc[0]
        rows.append(
            {
                "run_name": run.name,
                "epochs": int(df["epoch"].max()),
                "best_epoch": int(best_row["epoch"]),
                "best_f1": float(best_row["f1"]),
                "final_f1": float(df.iloc[-1]["f1"]),
                "final_val_loss": float(df.iloc[-1]["val_loss"]),
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
            }
        )
    return pd.DataFrame(rows)


def plot_series(run_paths: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load all series
    series = []
    for run in run_paths:
        df = pd.read_json(run / "metrics.jsonl", lines=True)
        series.append((run.name, df))

    # F1 by epoch
    plt.figure(figsize=(10, 6))
    for name, df in series:
        plt.plot(df["epoch"], df["f1"], label=name)
    plt.xlabel("epoch"); plt.ylabel("F1 (val)"); plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    f1_path = out_dir / "f1_by_epoch.png"
    plt.savefig(f1_path, dpi=180)
    plt.close()

    # Val loss by epoch
    plt.figure(figsize=(10, 6))
    for name, df in series:
        plt.plot(df["epoch"], df["val_loss"], label=name)
    plt.xlabel("epoch"); plt.ylabel("Val Loss"); plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    vl_path = out_dir / "val_loss_by_epoch.png"
    plt.savefig(vl_path, dpi=180)
    plt.close()

    # Best F1 bar
    names = [n for n, _ in series]
    bests = [float(df["f1"].max()) for _, df in series]
    plt.figure(figsize=(10, 6))
    plt.barh(names, bests)
    plt.xlabel("Best F1 (val)")
    plt.tight_layout()
    bf_path = out_dir / "best_f1_bar.png"
    plt.savefig(bf_path, dpi=180)
    plt.close()


def write_markdown(df: pd.DataFrame, out_dir: Path) -> None:
    md = out_dir / "README.md"
    lines = [
        "# Experiments Summary",
        "",
        "## Charts",
        f"![F1 by epoch](f1_by_epoch.png)",
        f"\n![Val loss by epoch](val_loss_by_epoch.png)",
        f"\n![Best F1](best_f1_bar.png)",
        "",
        "## Runs",
    ]
    for _, r in df.sort_values("best_f1", ascending=False).iterrows():
        lines.append(
            f"- {r['run_name']}: best_f1={r['best_f1']:.4f} @epoch {int(r['best_epoch'])} | "
            f"final_f1={r['final_f1']:.4f}, final_val_loss={r['final_val_loss']:.4f}"
        )
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs", type=str)
    ap.add_argument("--dest", default=None, type=str)
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    dest_root = Path(args.dest) if args.dest else Path("reports/summary")
    ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_dir = dest_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_runs(outputs_root)
    csv_path = out_dir / "experiments_summary.csv"
    df.to_csv(csv_path, index=False)

    # select top-K latest or all
    run_paths = [outputs_root / "runs" / n for n in df["run_name"].tolist()]
    plot_series(run_paths, out_dir)
    write_markdown(df, out_dir)
    print(f"Wrote summary to {out_dir}")


if __name__ == "__main__":
    main()

