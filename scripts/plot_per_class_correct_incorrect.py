#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_per_class(run_dir: Path) -> Dict | None:
    for fname in ["per_class_best.json", "per_class_latest.json"]:
        p = run_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def plot_bar(run_name: str, stats: Dict, out_dir: Path) -> Path:
    totals = stats.get("per_class_total", [])
    corrects = stats.get("per_class_correct", [])
    totals = np.array(totals, dtype=float)
    corrects = np.array(corrects, dtype=float)
    incorrects = totals - corrects

    idx = np.arange(len(totals))
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(idx - width/2, corrects / np.clip(totals, 1, None), width, label="correct", color="#4caf50")
    plt.bar(idx + width/2, incorrects / np.clip(totals, 1, None), width, label="incorrect", color="#f44336")
    plt.xlabel("class id"); plt.ylabel("ratio")
    plt.title(f"Per-class correct/incorrect ratio — {run_name}")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_class_correct_incorrect__{run_name}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="outputs/runs", type=str)
    ap.add_argument("--dest", default="reports/per_class", type=str)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    dest_root = Path(args.dest)
    generated: List[Path] = []
    for run in sorted(runs_root.glob("*")):
        stats = load_per_class(run)
        if not stats:
            continue
        generated.append(plot_bar(run.name, stats, dest_root))

    if generated:
        print("Saved:")
        for p in generated:
            print(" -", p)
    else:
        print("No per-class stats found. Re-run training with tensorboard enabled (already added) to record per-class stats.")


if __name__ == "__main__":
    main()

